import pandas as pd
import numpy as np
import time
import pickle
import argparse
from pathlib import Path
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from unicodedata import normalize


def preprocesamiento_texto(text):
    """Preprocesamiento robusto para títulos de productos"""
    if pd.isna(text):
        return ""
    
    text = str(text)
    text = normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def procesar_similitud(csv_path, output_path=None, top_n=5, modelo_path='models/similarity_model.pkl'):
    """
    Procesa un CSV de productos y genera similitudes
    
    Args:
        csv_path: Ruta al CSV con columna ITE_ITEM_TITLE
        output_path: Ruta de salida (opcional, por defecto output/resultados_similitud.csv)
        top_n: Número de productos similares a retornar
        modelo_path: Ruta al modelo TF-IDF entrenado
    """
    
    print("="*80)
    print("PROCESAMIENTO DE SIMILITUD DE PRODUCTOS")
    print("="*80)
    print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"CSV de entrada: {csv_path}")
    print("="*80)
    
    # Cargar datos
    print("\n[1/6] Cargando datos...")
    t0 = time.time()
    df = pd.read_csv(csv_path)
    t_carga = time.time() - t0
    
    if 'ITE_ITEM_TITLE' not in df.columns:
        raise ValueError("El CSV debe contener la columna 'ITE_ITEM_TITLE'")
    
    print(f"      Productos cargados: {len(df):,}")
    print(f"      Tiempo: {t_carga:.2f}s")
    
    # Cargar modelo
    print("\n[2/6] Cargando modelo TF-IDF...")
    t0 = time.time()
    
    if not Path(modelo_path).exists():
        raise FileNotFoundError(
            f"Modelo no encontrado: {modelo_path}\n"
            "Ejecutar primero: python train_model.py"
        )
    
    with open(modelo_path, 'rb') as f:
        model_data = pickle.load(f)
        vectorizer = model_data['vectorizer']
    
    t_modelo = time.time() - t0
    print(f"      Modelo cargado")
    print(f"      Tiempo: {t_modelo:.2f}s")
    
    # Preprocesamiento
    print("\n[3/6] Preprocesamiento de texto...")
    t0 = time.time()
    df['title_fix'] = df['ITE_ITEM_TITLE'].apply(preprocesamiento_texto)
    t_preproceso = time.time() - t0
    print(f"      Tiempo: {t_preproceso:.2f}s ({t_preproceso/len(df)*1000:.2f}ms por producto)")
    
    # Vectorización
    print("\n[4/6] Vectorización TF-IDF...")
    t0 = time.time()
    X = vectorizer.transform(df['title_fix'])
    t_vectorizacion = time.time() - t0
    print(f"      Tiempo: {t_vectorizacion:.2f}s ({t_vectorizacion/len(df)*1000:.2f}ms por producto)")
    print(f"      Dimensión de vectores: {X.shape}")
    print(f"      Sparsity: {(1 - X.nnz / (X.shape[0] * X.shape[1])) * 100:.2f}%")
    
    # Cálculo de similitud
    print("\n[5/6] Cálculo de similitud coseno...")
    t0 = time.time()
    similarity_matrix = cosine_similarity(X, X)
    t_similitud = time.time() - t0
    print(f"      Tiempo: {t_similitud:.2f}s")
    print(f"      Matriz de similitud: {similarity_matrix.shape}")
    
    # Extracción de top-N similares
    print(f"\n[6/6] Extracción de top-{top_n} productos similares...")
    t0 = time.time()
    
    resultados_lista = []
    for idx in range(len(df)):
        titulo_original = df.iloc[idx]['ITE_ITEM_TITLE']
        
        sim_scores = list(enumerate(similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[0:top_n]
        
        titulos_similares = [df.iloc[i]['ITE_ITEM_TITLE'] for i, _ in sim_scores]
        scores_similares = [score for _, score in sim_scores]
        
        resultados_lista.append({
            'title': titulo_original,
            'similar_titles': titulos_similares,
            'similarity_scores': scores_similares
        })
    
    t_extraccion = time.time() - t0
    print(f"      Tiempo: {t_extraccion:.2f}s ({t_extraccion/len(df)*1000:.2f}ms por producto)")
    
    # Guardar resultados
    df_resultados = pd.DataFrame(resultados_lista)
    
    if output_path is None:
        output_path = 'output/resultados_similitud.csv'
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_resultados.to_csv(output_path, index=False)
    
    # Resumen
    t_total = t_carga + t_modelo + t_preproceso + t_vectorizacion + t_similitud + t_extraccion
    
    print("\n" + "="*80)
    print("RESUMEN DE PERFORMANCE")
    print("="*80)
    print(f"Tiempo total:           {t_total:.2f}s")
    print(f"Productos procesados:   {len(df):,}")
    print(f"Tiempo por producto:    {t_total/len(df)*1000:.2f}ms")
    print(f"Throughput:             {len(df)/t_total:.2f} productos/segundo")
    print(f"\nArchivo generado:       {output_path}")
    print(f"Registros:              {len(df_resultados):,}")
    print("="*80)
    
    return df_resultados


def main():
    parser = argparse.ArgumentParser(
        description='Procesar CSV de productos y generar similitudes'
    )
    parser.add_argument(
        'csv_path',
        type=str,
        help='Ruta al CSV con columna ITE_ITEM_TITLE'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Ruta de salida (default: output/resultados_similitud.csv)'
    )
    parser.add_argument(
        '-n', '--top-n',
        type=int,
        default=5,
        help='Número de productos similares (default: 5)'
    )
    parser.add_argument(
        '-m', '--modelo',
        type=str,
        default='models/similarity_model.pkl',
        help='Ruta al modelo TF-IDF (default: models/similarity_model.pkl)'
    )
    
    args = parser.parse_args()
    
    try:
        procesar_similitud(
            csv_path=args.csv_path,
            output_path=args.output,
            top_n=args.top_n,
            modelo_path=args.modelo
        )
    except Exception as e:
        print(f"\nError: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())


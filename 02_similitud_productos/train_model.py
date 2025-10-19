import argparse
import traceback
from pathlib import Path
import time

import pandas as pd

from similarity_engine import SimilarityEngine


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description='Entrenar modelo de similitud TF-IDF')
    parser.add_argument('--train', type=str, default='data/items_titles.csv',
                        help='Dataset para entrenar vocabulario TF-IDF')
    parser.add_argument('--catalog', type=str, default='data/items_titles_test.csv',
                        help='Dataset para catálogo de búsqueda')
    parser.add_argument('--output', type=str, default='models/similarity_model.pkl')
    
    args = parser.parse_args()
    
    train_path = Path(args.train)
    catalog_path = Path(args.catalog)
    
    if not train_path.exists():
        print(f"Error: Archivo de entrenamiento no encontrado: {train_path}")
        return 1
    
    if not catalog_path.exists():
        print(f"Error: Archivo de catálogo no encontrado: {catalog_path}")
        return 1
    
    print("="*80)
    print("ENTRENAMIENTO DE MODELO DE SIMILITUD")
    print("="*80)
    print(f"\nDatos de entrenamiento (vocabulario): {train_path}")
    print(f"Catálogo de búsqueda: {catalog_path}")
    print(f"Salida: {args.output}\n")
    
    try:
        df_train = pd.read_csv(train_path)
        df_catalog = pd.read_csv(catalog_path)
        
        print(f"Entrenando vocabulario con {len(df_train):,} productos...")
        engine = SimilarityEngine()
        engine.fit(df_train)
        
        print(f"Estableciendo catálogo con {len(df_catalog):,} productos...")
        engine.set_catalog(df_catalog)
        
        # Guardar modelo
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        engine.save(str(output_path))
        
        print(f"\nModelo guardado en: {output_path}")
        print(f"Vocabulario: {len(engine.vectorizer.vocabulary_):,} términos")
        print(f"Catálogo: {len(engine.items_df):,} productos")
        
        print("\n" + "="*80)
        print("MODELO ENTRENADO EXITOSAMENTE")
        print("="*80)
        
        # Test
        test_query = "tenis nike masculino"
        print(f"\nTest query: '{test_query}'")
        results = engine.find_similar(test_query, top_k=3)
        
        print("Top 3 resultados del catálogo:")
        for i, result in enumerate(results, 1):
            score = result['similarity_score']
            title = result['title']
            print(f"  {i}. [{score:.4f}] {title}")
        
        end_time = time.time()
        print(f"Tiempo de ejecución todo el pipeline: {end_time - start_time:.2f} segundos")
        
        return 0

    except Exception as e:
        print(f"\nError: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

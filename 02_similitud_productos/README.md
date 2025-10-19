# Similitud de Productos

Sistema de búsqueda de productos similares basado en análisis de títulos usando TF-IDF y similitud coseno.

## Estructura

```
02_similitud_productos/
├── data/
│   ├── items_titles.csv          # 30,000 títulos (portugués)
│   └── items_titles_test.csv     # Conjunto de prueba
├── notebook/
│   └── similitud_models.ipynb    # Exploración y desarrollo
├── models/
│   └── similarity_model.pkl      # Modelo entrenado
├── output/
│   ├── df_resultados_cos.csv     # Resultados de similitud
│   ├── reporte_escalabilidad.txt
├── similarity_engine.py          # Motor de similitud
├── train_model.py                # Entrenamiento
├── procesar_similitud.py         # Script de procesamiento
├── api.py                        # API REST
├── Dockerfile
└── requirements.txt
```

## Solución Implementada

### Enfoque Técnico

**Vectorización:** TF-IDF con n-gramas de palabras (1,2)
**Similitud:** Coseno entre vectores normalizados L2
**Preprocesamiento:** Normalización Unicode, lowercase, limpieza de caracteres especiales

### Restricciones Cumplidas

- No utiliza modelos pre-entrenados
- Implementación desde cero con scikit-learn
- Análisis de escalabilidad incluido

### Desarrollo en Notebook

El notebook `similitud_models.ipynb` contiene:

- Exploración del dataset (portugués brasileño)
- Pruebas de preprocesamiento
- Experimentos con TF-IDF (word-level y character-level)
- Comparación de configuraciones de n-gramas
- Análisis de resultados y visualizaciones

## Uso

### 1. Entrenamiento (Requerido)

El modelo debe entrenarse antes de usar la API:

```bash
python train_model.py
```

Esto:

- Entrena vocabulario TF-IDF con `data/items_titles.csv` (30,000 productos)
- Establece catálogo de búsqueda con `data/items_titles_test.csv` (10,000 productos)
- Genera `models/similarity_model.pkl`

**Separación train/catalog:**

- **Train (30K):** Vocabulario rico para TF-IDF
- **Catalog (10K):** Productos sobre los que se busca

**Nota:** La API NO entrena automáticamente. Debes ejecutar este paso primero.

### 2. Procesamiento de CSV

Para procesar un CSV y generar similitudes:

```bash
python procesar_similitud.py <ruta_csv> [opciones]
```

**Opciones:**

```bash
-o, --output PATH    Ruta de salida (default: output/resultados_similitud.csv)
-n, --top-n N        Número de similares (default: 5)
-m, --modelo PATH    Ruta al modelo (default: models/similarity_model.pkl)
```

**Ejemplo:**

```bash
python procesar_similitud.py data/items_titles_test.csv -n 5 -o output/resultados.csv
```

**Salida:**

```

[1/6] Cargando datos...
      Productos cargados: 10,000
[2/6] Cargando modelo TF-IDF...
[3/6] Preprocesamiento de texto...
[4/6] Vectorización TF-IDF...
[5/6] Cálculo de similitud coseno...
[6/6] Extracción de top-5 productos similares...

RESUMEN DE PERFORMANCE
Tiempo total:           35.14s
Productos procesados:   10,000
Tiempo por producto:    3.51ms
Throughput:             284.60 productos/segundo
```

**Requisitos del CSV:**

- Debe contener columna `ITE_ITEM_TITLE`
- Formato: CSV estándar con headers

### 3. API REST (Demostración)

API simple para comparar similitud entre dos títulos.

**Iniciar API:**

```bash
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

**Endpoint:**

```bash
POST /comparar
```

**Ejemplo:**

```bash
curl -X POST "http://localhost:8000/comparar" \
  -H "Content-Type: application/json" \
  -d '{
    "titulo1": "Tenis Nike Air Max",
    "titulo2": "Tenis Nike Air Force"
  }'
```

**Respuesta:**

```json
{
  "titulo1": "Tenis Nike Air Max",
  "titulo2": "Tenis Nike Air Force",
  "similitud": 0.5705,
  "interpretacion": "Moderadamente similares"
}
```

**Interpretación de scores:**

- `1.0` = Idénticos
- `0.7-0.9` = Muy similares
- `0.5-0.7` = Similares
- `0.3-0.5` = Algo similares
- `<0.3` = Diferentes

**Documentación interactiva:**

http://localhost:8000/docs

## Análisis de Escalabilidad

**Performance (10,000 productos en test):**

- Tiempo de entrenamiento: ~1-2s
- Latencia de búsqueda: ~20-50ms
- Throughput: ~20-50 queries/s
- Memoria: ~500MB
- Fórmula estimada: tiempo = 0.003477 * tamaño + -6.5811

**Limitaciones:**

- Búsqueda exhaustiva crece linealmente con el catálogo
- Todos los vectores se mantienen en memoria

Ver `output/reporte_escalabilidad.txt` para análisis detallado.

## Configuración Técnica

### TF-IDF

```python
TfidfVectorizer(
    analyzer='word',
    ngram_range=(1, 2),
    min_df=2,
    strip_accents='unicode',
    lowercase=True
)
```

### Preprocesamiento

```python
def preprocess_text(text):
    text = normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
```

## Resultados

**Vocabulario:** 30,000 productos (24,498 términos únicos)
**Catálogo de búsqueda:** 10,000 productos en portugués brasileño

**Observaciones del notebook:**

- TF-IDF word-level captura similitud semántica básica
- N-gramas (1,2) mejoran detección de marcas y modelos
- Character n-gramas útiles para variaciones ortográficas
- Normalización Unicode crítica para portugués

**Arquitectura:**

- Vocabulario entrenado con dataset completo (30K) para mayor cobertura
- Búsquedas realizadas sobre catálogo reducido (10K) para performance

**Formato de salida:**

```csv
item_id_1,title_1,item_id_2,title_2,similarity_score
123,Produto A,456,Produto B,0.95
```

## Dependencias

```
pandas
numpy
scikit-learn
fastapi
uvicorn
pydantic
```

## Notas

- Los datos están en portugués brasileño
- La normalización L2 permite cálculo eficiente de similitud coseno
- El modelo se entrena automáticamente al iniciar la API si no existe
- El sistema preserva números en títulos (relevantes para productos)

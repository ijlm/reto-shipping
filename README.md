# Data Science Challenge

Solución técnica para el desafío de Data & Analytics.

## Estructura del Proyecto

```
test_shitment/
├── 01_ofertas_relampago/    # Análisis exploratorio de ofertas relampago 
│   ├── data/
│   └── README.md
├── 02_similitud_productos/   # Sistema de similitud de titlle de productos
│   ├── data/
│   ├── models/
│   ├── output/
│   ├── api.py
│   ├── similarity_engine.py
│   ├── train_model.py
│   ├── Dockerfile
│   └── README.md
├── 03_prevision_falla/       # Modelo predictivo de fallas
│   ├── data/
│   └── README.md
└── requirements.txt
```

## Configuración del Entorno

### Requisitos Previos

- Python 3.12+
- Docker (opcional, para deployment)

### Instalación

```bash
# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

## Ejercicio 1: Análisis de Ofertas Relámpago

**Ubicación:** `01_ofertas_relampago/`

### Resumen del Trabajo

Análisis exploratorio completo de ofertas relámpago para identificar factores de éxito.

**Hallazgos principales:**

- Análisis de 1,106 ofertas con métricas de ventas y stock
- Identificación de factores clave: ORIGIN, stock inicial, vertical, timing
- Ofertas con ORIGIN especial tienen 3x más ventas que ofertas regulares
- Verticals más exitosos: Beauty & Health, Phones, Electronics
- Mejor horario: 12-14h (almuerzo) y 20-22h (noche)
- 

**Insights de negocio:**

- Free shipping aumenta conversión significativamente
- Stock inicial debe balancearse con urgencia percibida
- Días laborales (martes-jueves) generan mejores resultados
- Categorías premium requieren mayor duración de oferta

Ver análisis completo en [01_ofertas_relampago/README.md](01_ofertas_relampago/README.md)

---

## Ejercicio 2: Similitud de Productos

**Ubicación:** `02_similitud_productos/`

### Resumen del Trabajo

Sistema de búsqueda de productos similares basado en NLP sin modelos pre-entrenados.

**Implementación:**

- TF-IDF con n-gramas de palabras (1-2)
- Similitud coseno para comparación de vectores
- Vocabulario entrenado en 30K productos (portugués)
- Script de procesamiento batch para generar CSV con similitudes
- API REST simple para demostración

**Componentes:**

- `procesar_similitud.py`: Procesa CSV y genera similitudes con métricas de performance
- `api.py`: API para comparar 2 títulos directamente
- `train_model.py`: Entrena modelo TF-IDF
- Notebook con análisis completo y pipeline de procesamiento

**Performance:**

- 10,000 productos procesados en 35s
- Throughput: ~295 productos/segundo
- Tiempo por producto: 3.39ms
- Producción-ready con Docker

### Ejecución

**1. Entrenar modelo:**

```bash
cd 02_similitud_productos
python train_model.py
```

**2. Procesar CSV:**

```bash
python procesar_similitud.py data/items_titles_test.csv -n 5
```

**3. API (opcional):**

```bash
python -m uvicorn api:app --host 0.0.0.0 --port 8000
curl -X POST "http://localhost:8000/comparar" \
  -H "Content-Type: application/json" \
  -d '{"titulo1": "Tenis Nike Air Max", "titulo2": "Tenis Nike Air Force"}'
```

Ver detalles técnicos en [02_similitud_productos/README.md](02_similitud_productos/README.md)

---

## Ejercicio 3: Predicción de Fallas en Dispositivos

**Ubicación:** `03_prevision_falla/`

### Resumen del Trabajo

Modelo predictivo para anticipar fallas en dispositivos con un día de antelación usando series temporales.

**Problema:**

- 1,169 dispositivos con features diarias
- Desbalance severo: 0.09% de eventos con falla (8-10% por dispositivo)
- Objetivo: predecir falla del día anterior con shift(-1)
- Datos temporales con dependencia histórica

**Feature Engineering:**

- 224 features temporales generadas a partir de 9 atributos
- Selección de 73 features con Information Value > 0.2
- Lags (1, 2, 3, 5, 7 días), rolling statistics, diferencias
- Ventanas disjuntas (1-2d, 3-4d, 4-7d, 8-14d)

**Experimentos realizados:**

1. **XGBoost + Optuna (seleccionado):** AUC-PR 0.22, Recall 0.65, Precision 0.15
2. **Random Survival Forest:** AUC-PR 0.21, Recall 0.30
3. **LSTM:** AUC-PR 0.32, Recall 0.53 (probabilidades  muy centradas)

**Modelo seleccionado: XGBoost + Optuna**

**Razones:**

- Mejor recall (65%): detecta la mayoría de fallas
- Balance óptimo entre precisión y recall
- Interpretable: feature importance clara
- Simple de desplegar y mantener
- En mantenimiento predictivo, minimizar False Negatives es crítico

**Features más importantes:** lags y promedios de attribute5, attribute2, attribute4

### Ejecución

**Local:**

```bash
cd 03_prevision_falla
source ../.venv_py312/bin/activate
python train_model.py
python -m uvicorn api:app --host 0.0.0.0 --port 8001
```

**Docker:**

```bash
cd 03_prevision_falla
docker build -t failure-api .
docker run -d -p 8001:8001 --name failure-api failure-api
```

**Endpoint:** `POST /predict` (recibe 73 features pre-calculadas)

**Documentación:** http://localhost:8001/docs

Ver análisis completo en [03_prevision_falla/README.md](03_prevision_falla/README.md)

## Tecnologías

**Core:**

- Python 3.12
- scikit-learn 1.7.2
- pandas, numpy
- XGBoost
- TensorFlow (LSTM)

**APIs:**

- FastAPI

**Deployment:**

- Docker

**Optimización:**

- Optuna (hyperparameter tuning)

## Datasets

Los datos se encuentran en las carpetas `data/` de cada ejercicio:

1. `01_ofertas_relampago/data/ofertas_relampago.csv` - 1,106 ofertas
2. `02_similitud_productos/data/items_titles.csv` - 30K productos (train)
3. `02_similitud_productos/data/items_titles_test.csv` - 10K productos (test)
4. `03_prevision_falla/data/full_devices.csv` - 1,169 dispositivos

## Performance

**Ejercicio 2 - Similitud:**

- Tiempo: <100ms para top-5 similares
- Escalable hasta 100K productos

**Ejercicio 3 - Predicción:**

- Recall: 65% (detecta mayoría de fallas)
- Precision: 15% (trade-off aceptable para mantenimiento)
- Inferencia: <50ms por dispositivo

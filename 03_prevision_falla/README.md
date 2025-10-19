# Ejercicio 3: Predicción de Fallas en Dispositivos

## Descripción

Modelo predictivo para anticipar fallas en dispositivos con un día de antelación. El sistema utiliza datos diarios de los device y técnicas de series temporales para mantenimiento predictivo.

## Dataset

**Ubicación:** `data/full_devices.csv`

### Estructura

- `device` - Identificador del dispositivo
- `date` - Fecha de la evento
- `attribute_1` a `attribute_9` - Métricas/features
- `failure` - Variable objetivo (0: sin falla, 1: falla)


## Estructura del Proyecto

```
03_prevision_falla/
├── data/
│   ├── full_devices.csv              # Dataset original
│   ├── train_dataset_eventos.csv     # Train con features
│   └── test_dataset_eventos.csv      # Test con features
├── notebook/
│   └── modelado_completo.ipynb       # Análisis y experimentos
├── models/
│   ├── failure_model.pkl             # Modelo entrenado
│   ├── selected_features.txt         # 73 features seleccionadas
│   └── example_features.json         # Ejemplo para Swagger
├── train_model.py                    # Script de entrenamiento
├── prediction_engine.py              # Motor de predicción
├── api.py                            # FastAPI
├── Dockerfile                        # Imagen Docker
├── requirements.txt                  # Dependencias
└── README.md                         # Este archivo
```


### Características del Problema

- Datos de series temporales con dependencia temporal
- 1169dispositivos con historiales independientes
- Desbalance de clases severo: alto visto por eventos, tratable con device unico.
- Objetivo: predecir falla del día anterior.

---

## Resumen del Trabajo Realizado

### 1. Análisis Exploratorio

**Hallazgos principales:**

- Desbalance severo: solo 0.09% de los registros corresponden a fallas, por device el 8~10%
- Atributos con comportamientos distintos: algunos estables, otros con alta variabilidad
- Patrones temporales identificados: cambios bruscos en días previos a fallas
- Dispositivos con historiales variables (algunos nunca fallan)

**Decisiones tomadas:**

- Split temporal 80/20 para respetar la naturaleza de series temporales, en algunos casos repartir la falla en train y test.
- Construcción del target con shift(-1) para predecir el día anterior. Target extendido para ver todos los eventos.
- Análisis de correlaciones y comportamiento de features por dispositivo

### 2. Feature Engineering

**Estrategia implementada:**

Se generaron 224 features temporales a partir de los 9 atributos originales (8 un duplicado):

- **Lags:** valores de días anteriores (1, 2, 3, 5, 7 días)
- **Rolling means:** promedios móviles en ventanas de 1, 2, 3, 5, 7, 14 días
- **Rolling std:** desviaciones estándar en ventanas móviles
- **Differences:** cambios entre periodos (1 y 7 días)
- **Ventanas disjuntas:** estadísticas en rangos específicos (1-2d, 3-4d, 4-7d, 8-14d)

**Selección de features:**

- Aplicación de Information Value (IV) para medir poder predictivo
- Selección de 73 features con IV > 0.2
- Features más importantes: lags, rolling std, y diferencias de attribute1, attribute5, attribute6

### 3. Experimentos de Modelado

Se probaron 4 enfoques diferentes:

#### Experimento 1: XGBoost sin optimización

**Configuración:**

- Parámetros por defecto
- scale_pos_weight para manejar desbalance

**Resultados:**

- AUC-PR: 0.2847
- Recall: 0.52
- Precision: 0.08

**Análisis:** Baseline funcional pero con baja precisión.

#### Experimento 2: XGBoost + Optuna (Seleccionado) +SMOTE ( como referencia)

**Configuración:**

- Optimización de hiperparámetros con Optuna
- Parámetros optimizados: max_depth=3, learning_rate=0.01, n_estimators=500
- scale_pos_weight calculado del desbalance

**Resultados:**

- AUC-PR: 0.22
- Recall: 0.65
- Precision: 0.15

**Análisis:** Mejor balance entre detectar fallas y minimizar errores críticos.

#### Experimento 3: Random Survival Forest

**Configuración:**

- Modelo especializado en análisis de supervivencia
- Enfoque en tiempo hasta falla

**Resultados:**

- AUC-PR: 0.2156
- Recall: 0.30
- Precision: 0.07

**Análisis:** Bajo rendimiento, no captura bien los patrones de falla.

#### Experimento 4: LSTM

**Configuración:**

- Red neuronal recurrente LSTM
- Secuencias de 14 días
- Dropout para regularización

**Resultados:**

- AUC-PR: 0.3234 (mejor)
- Recall: 0.53
- Precision: 0.10

**Análisis:** Mejor AUC-PR pero menor recall que XGBoost+Optuna.

### 4. Selección del Modelo Final

**Modelo seleccionado: XGBoost + Optuna**

**Razones:**

1. **Mejor recall (0.66):** Detecta 65% de las fallas, minimizando el riesgo de downtime no planificado
2. **Balance óptimo:** Aunque tiene más falsas alarmas que LSTM, detecta más fallas reales
3. **Interpretabilidad:** Feature importance permite entender qué atributos predicen fallas
4. **Producción:** Más simple de desplegar y mantener que LSTM
5. **Costo de negocio:** En mantenimiento predictivo, el costo de NO detectar una falla (False Negative) es mucho mayor que el de una falsa alarma (False Positive)

**Ventajas:**

- Alto recall: detecta la mayoría de fallas
- Rápido en inferencia
- Interpretable
- Robusto con datos desbalanceados

**Desventajas:**

- Precisión baja (9%): genera muchas falsas alarmas
- Requiere umbral de decisión ajustado al negocio
- No captura patrones secuenciales complejos como LSTM

### 5. Features Más Importantes

Las N features con mayor poder predictivo:

1. attribute5_lag_7
2. attribute2
3. attribute2_roll_1_mean
4. attribute4_roll_2_mean
5. attribute4_roll_3_mean
6. attribute7_diff_7

**Interpretación:** Los lags y promedios son los mejores predictores de fallas.

---

## Consideraciones de Producción

### Métricas de Negocio

En mantenimiento predictivo, el objetivo es **minimizar False Negatives** (fallas no detectadas):

- False Negative: dispositivo falla sin previo aviso (alto costo)
- False Positive: mantenimiento innecesario (bajo costo relativo)

Por eso se prioriza **Recall** sobre Precision.

### Umbral de Decisión

El modelo puede ajustarse según el contexto:

- Umbral bajo (0.3): más sensible, detecta más fallas pero más falsas alarmas
- Umbral alto (0.7): más conservador, menos falsas alarmas pero puede perder fallas

### Limitaciones

- Requiere al menos 14 días de historial para generar todas las features
- Desempeño puede variar entre dispositivos con comportamientos distintos
- Precisión baja implica necesidad de validación humana de alertas

---

## API de Predicción

### Entrenar Modelo

```bash
cd 03_prevision_falla
source ../.venv_py312/bin/activate
python train_model.py
```

### Ejecutar API

**Local:**

```bash
python -m uvicorn api:app --host 0.0.0.0 --port 8001
```

**Docker con Colima:**

```bash
# Iniciar Colima (si no está corriendo)
colima start

# Build
docker build -t failure-api .

# Run
docker run -d -p 8001:8001 --name failure-api failure-api

# Ver logs
docker logs failure-api

# Detener
docker stop failure-api
docker rm failure-api
```

### Endpoints

- `POST /predict` - Predecir falla para un dispositivo

### Documentación Interactiva

Una vez iniciada la API, acceder a:

- Swagger UI: http://localhost:8001/docs
- ReDoc: http://localhost:8001/redoc

### Ejemplo de Uso

```python
import requests

payload = {
    "device_id": "DEVICE_12345",
    "features": {
        "attribute6_roll_14_std": 4.2173,
        "attribute5_lag_5": 19.0,
        "attribute6_lag_5": 339280.0,
        # ... (73 features en total)
    }
}

response = requests.post("http://localhost:8001/predict", json=payload)
print(response.json())
# {
#   "device_id": "DEVICE_12345",
#   "prediction": 0,
#   "probability": 0.0217,
#   "risk_level": "low"
# }
```

---

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Optional
from pathlib import Path
import logging
import json
from prediction_engine import PredictionEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="API Predicción de Fallas",
    description="Modelo entrenado con datos del train - 73 features seleccionadas",
    version="1.0.0"
)

engine: Optional[PredictionEngine] = None

# Cargar ejemplo de features
with open('models/example_features.json', 'r') as f:
    EXAMPLE_FEATURES = json.load(f)

class PredictionRequest(BaseModel):
    device_id: str = Field(..., example="DEVICE_12345")
    features: Dict[str, float] = Field(..., example=EXAMPLE_FEATURES)

class PredictionResponse(BaseModel):
    device_id: str = Field(..., example="DEVICE_12345")
    prediction: int = Field(..., example=0, description="0=No falla, 1=Falla")
    probability: float = Field(..., example=0.0217, ge=0.0, le=1.0)
    risk_level: str = Field(..., example="low", description="low, medium, high, critical")

@app.on_event("startup")
async def load_model():
    global engine
    try:
        model_path = Path("models/failure_model.pkl")
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
        
        logger.info(f"Cargando modelo desde {model_path}...")
        engine = PredictionEngine(str(model_path))
        logger.info(f"Modelo cargado. Features: {len(engine.features)}")
        if engine.metrics:
            logger.info(f"Métricas del modelo: AUC-PR={engine.metrics.get('aucpr', 0):.4f}, FN={engine.metrics.get('fn', 0)}")
    except Exception as e:
        logger.error(f"Error cargando modelo: {e}")
        raise

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predice si un dispositivo fallará al día siguiente
    
    Requiere 73 features pre-calculadas del notebook:
    - Lags (1, 2, 3, 5, 7 días)
    - Rolling means y std (1, 2, 3, 5, 7, 14 días)
    - Ventanas disjuntas (w1_2d, w3_4d, w4_7d, w8_14d)
    - Differences (1, 7 días)
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    
    try:
        prediction, probability = engine.predict_from_features(request.features)
        
        if probability >= 0.75:
            risk_level = "critical"
        elif probability >= 0.5:
            risk_level = "high"
        elif probability >= 0.25:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return PredictionResponse(
            device_id=request.device_id,
            prediction=prediction,
            probability=probability,
            risk_level=risk_level
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8001)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
import pickle
import logging
from sklearn.metrics.pairwise import cosine_similarity
import re
from unicodedata import normalize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="API Similitud de Productos",
    description="Compara similitud entre títulos de productos usando TF-IDF",
    version="1.0.0"
)

vectorizer = None


def preprocesamiento_texto(text):
    """Preprocesamiento de títulos"""
    if not text:
        return ""
    text = str(text)
    text = normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class CompareRequest(BaseModel):
    titulo1: str = Field(..., example="Tenis Nike Air Max")
    titulo2: str = Field(..., example="Tenis Nike Air Force")


class CompareResponse(BaseModel):
    titulo1: str
    titulo2: str
    similitud: float = Field(..., ge=0.0, le=1.0)
    interpretacion: str


@app.on_event("startup")
async def load_model():
    global vectorizer
    
    try:
        model_path = Path("models/similarity_model.pkl")
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Modelo no encontrado: {model_path}\n"
                "Ejecutar: python train_model.py"
            )
        
        logger.info("Cargando modelo TF-IDF...")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            vectorizer = model_data['vectorizer']
        
        logger.info("Modelo cargado")
        
    except Exception as e:
        logger.error(f"Error cargando modelo: {e}")
        raise


@app.get("/")
async def root():
    return {
        "message": "API de Similitud de Productos",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy" if vectorizer is not None else "degraded",
        "modelo_cargado": vectorizer is not None
    }


@app.post("/comparar", response_model=CompareResponse)
async def comparar_titulos(request: CompareRequest):
    """
    Compara la similitud entre dos títulos de productos
    
    Retorna un score de 0 a 1:
    - 1.0 = Idénticos
    - 0.7-0.9 = Muy similares
    - 0.5-0.7 = Similares
    - 0.3-0.5 = Algo similares
    - <0.3 = Diferentes
    """
    if vectorizer is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    
    try:
        # Preprocesar
        clean1 = preprocesamiento_texto(request.titulo1)
        clean2 = preprocesamiento_texto(request.titulo2)
        
        # Vectorizar
        vec1 = vectorizer.transform([clean1])
        vec2 = vectorizer.transform([clean2])
        
        # Calcular similitud
        score = float(cosine_similarity(vec1, vec2)[0, 0])
        
        # Interpretación
        if score >= 0.9:
            interpretacion = "Muy similares"
        elif score >= 0.7:
            interpretacion = "Similares"
        elif score >= 0.5:
            interpretacion = "Moderadamente similares"
        elif score >= 0.3:
            interpretacion = "Poco similares"
        else:
            interpretacion = "Diferentes"
        
        return CompareResponse(
            titulo1=request.titulo1,
            titulo2=request.titulo2,
            similitud=round(score, 4),
            interpretacion=interpretacion
        )
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000)

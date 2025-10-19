FROM python:3.12-slim

LABEL description="API de Predicción de Fallas - 73 features del notebook"

WORKDIR /app

RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY prediction_engine.py .
COPY api.py .
COPY models/failure_model.pkl models/
COPY models/example_features.json models/

EXPOSE 8001

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001"]

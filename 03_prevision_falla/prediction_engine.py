import pandas as pd
import joblib
from typing import Tuple, Dict

class PredictionEngine:
    def __init__(self, model_path: str):
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.features = model_data['features']
        self.metrics = model_data.get('metrics', {})
    
    def predict_from_features(self, features_dict: Dict[str, float]) -> Tuple[int, float]:

        ##Predice desde features pre-calculadas (73 features del trabajo realizado en elnotebook)
      
        missing = set(self.features) - set(features_dict.keys())
        if missing:
            raise ValueError(f"Faltan {len(missing)} features: {list(missing)[:5]}")
        
        X = pd.DataFrame([features_dict])[self.features]
        
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0, 1]
        
        return int(prediction), float(probability)

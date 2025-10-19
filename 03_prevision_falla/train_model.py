import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, average_precision_score, confusion_matrix
from pathlib import Path
import logging
import joblib

logger = logging.getLogger(__name__)

def load_selected_features():
    """Carga las 73 features seleccionadas durante el entrenamiento"""
    features_file = Path("models/selected_features.txt")
    if not features_file.exists():
        raise FileNotFoundError(
            f"Archivo de features no encontrado: {features_file}\n"
            "Este archivo debe contener las 73 features seleccionadas del notebook"
        )
    
    with open(features_file, 'r') as f:
        features = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Features seleccionadas cargadas: {len(features)}")
    return features

def train_from_notebook_data():
    """entrena con datos generados en el notebook"""
    
    # Cargar datasets del notebook
    logger.info("\n1. Cargando datasets del notebook...")
    train_df = pd.read_csv('data/train_dataset_eventos.csv')
    test_df = pd.read_csv('data/test_dataset_eventos.csv')
    
    logger.info(f"   Train: {train_df.shape}")
    logger.info(f"   Test: {test_df.shape}")
    
    # Cargar features seleccionadas
    logger.info("\n2. Cargando features seleccionadas...")
    selected_features = load_selected_features()
    
    # Preparar datos
    logger.info("\n3. Preparando datos...")
    X_train = train_df[selected_features]
    y_train = train_df['extended_target']
    X_test = test_df[selected_features]
    y_test = test_df['extended_target']
    
    logger.info(f"   X_train: {X_train.shape}")
    logger.info(f"   y_train: {y_train.value_counts().to_dict()}")
    
    # Calcular scale_pos_weight
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    logger.info(f"\n4. Scale_pos_weight: {scale_pos_weight:.2f}")
    
    # Parámetros del notebook (XGBoost + Optuna)
    logger.info("\n5. Entrenando XGBoost con parámetros del de optuna")
    params = {
        'max_depth': 10,
        'learning_rate': 0.027745816713898846,
        'n_estimators': 377,
        'subsample': 0.8824698130887566,
        'colsample_bytree': 0.7869923236661704,
        'gamma': 1.3471642348257076e-05,
        'reg_alpha': 1.0649212529305546e-06,
        'reg_lambda': 1.8155856048269795e-08,
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'tree_method': 'auto',
        'use_label_encoder': False,
        'eval_metric': 'aucpr'
    }
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=False)
    
    logger.info("   Modelo entrenado")
    
    # Evaluar
    logger.info("\n6. Evaluando modelo...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    logger.info("\n" + "="*60)
    logger.info("RESULTADOS DEL MODELO")
    logger.info("="*60)
    logger.info(f"True Negatives: {tn}")
    logger.info(f"False Positives: {fp} (falsas alarmas)")
    logger.info(f"False Negatives: {fn} (fallas NO detectadas)")
    logger.info(f"True Positives: {tp} (fallas detectadas)")
    logger.info(f"Precision: {tp/(tp+fp) if (tp+fp) > 0 else 0:.4f}")
    logger.info(f"Recall: {tp/(tp+fn) if (tp+fn) > 0 else 0:.4f}")
    
    aucpr = average_precision_score(y_test, y_proba)
    logger.info(f"AUC-PR: {aucpr:.4f}")
    logger.info("="*60)
    
    # Guardar modelo
    logger.info("\n7. Guardando modelo...")
    output_path = Path("models/failure_model.pkl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    model_data = {
        'model': model,
        'features': selected_features,
        'params': params,
        'metrics': {
            'aucpr': float(aucpr),
            'fn': int(fn),
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn)
        }
    }
    joblib.dump(model_data, output_path)
    logger.info(f"   Modelo guardado en: {output_path}")
    
    logger.info("\nEntrenamiento completado")

if __name__ == "__main__":
    train_from_notebook_data()

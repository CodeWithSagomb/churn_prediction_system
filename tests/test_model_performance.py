# tests/test_model_performance.py

import pytest
import mlflow
import pandas as pd
from sklearn.metrics import f1_score

# Importer nos modules
from src.utils import load_config
from src.data_preprocessing import load_raw_data, preprocess_and_split_data

def test_model_performance_threshold():
    """
    Teste si le F1-score du modèle enregistré est supérieur à un seuil défini.
    C'est un test de non-régression de performance.
    """
    # 1. Arrange: Charger la configuration, les données et le modèle
    try:
        config = load_config("params.yml")
    except FileNotFoundError:
        # Si exécuté depuis un autre répertoire, ajuster le chemin
        config = load_config("../params.yml")

    # Re-générer les données de test pour être sûr qu'elles sont cohérentes
    raw_df = load_raw_data(config)
    _, X_test_transformed, _, y_test, _ = preprocess_and_split_data(raw_df, config)

    # Définir le seuil de performance attendu
    PERFORMANCE_THRESHOLD = 0.60  # Nous attendons un F1-score d'au moins 60%

    # ID du run du modèle XGBoost à tester (celui avec F1=0.6319)
    # Assurez-vous que cet ID est correct
    XGBOOST_RUN_ID = "598f4622dd314b39a163d8a208d1d993" # <--- METTEZ LE BON ID ICI

    # Charger le modèle depuis MLflow
    mlflow.set_tracking_uri("mlruns") # Indiquer où trouver les runs
    try:
        model_uri = f"runs:/{XGBOOST_RUN_ID}/model_XGBoost"
        model = mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        pytest.fail(f"Impossible de charger le modèle depuis MLflow. Vérifiez le RUN_ID. Erreur: {e}")

    # 2. Act: Faire des prédictions et calculer la performance
    y_pred = model.predict(X_test_transformed)
    current_f1_score = f1_score(y_test, y_pred)
    print(f"F1-score actuel du modèle chargé : {current_f1_score:.4f}")

    # 3. Assert: Vérifier que la performance dépasse le seuil
    assert current_f1_score > PERFORMANCE_THRESHOLD, \
        f"La performance du modèle ({current_f1_score:.4f}) est inférieure au seuil de {PERFORMANCE_THRESHOLD}"
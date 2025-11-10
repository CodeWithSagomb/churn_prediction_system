# production/api/main.py

"""
FastAPI pour le modèle de prédiction de churn.

MLOps Expert - Production Ready API
Version: 1.0.0
"""

from fastapi import FastAPI, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd
import sys
import os

# Ajouter le chemin parent pour importer les modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_object
from src.data_preprocessing import preprocess_data_for_prediction

# Configuration
API_VERSION = "1.0.0"
MODEL_VERSION = "1.0.0"

# Sécurité - API Keys (à stocker dans variables d'environnement en prod)
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
VALID_API_KEYS = {
    "demo-key-123": "demo",
    "prod-key-456": "production"
}

# Initialisation FastAPI
app = FastAPI(
    title="Churn Prediction API",
    description="API de prédiction de churn client - Télécommunications",
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Charger le modèle au démarrage
try:
    MODEL_PACKAGE = load_object('models/churn_model_v1.joblib')
    PREPROCESSOR = load_object('data/preprocessor.joblib')
    print(f"✅ Modèle chargé: F1={MODEL_PACKAGE['metrics']['test_f1']:.4f}")
except Exception as e:
    print(f"❌ Erreur chargement modèle: {e}")
    MODEL_PACKAGE = None
    PREPROCESSOR = None


# Modèles Pydantic
class Customer(BaseModel):
    """Données client pour prédiction"""
    customerID: Optional[str] = Field(None, description="ID client (optionnel)")
    gender: str = Field(..., description="Genre: Male/Female")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="0 ou 1")
    Partner: str = Field(..., description="Yes/No")
    Dependents: str = Field(..., description="Yes/No")
    tenure: int = Field(..., ge=0, description="Mois d'ancienneté")
    PhoneService: str = Field(..., description="Yes/No")
    MultipleLines: str = Field(..., description="Yes/No/No phone service")
    InternetService: str = Field(..., description="DSL/Fiber optic/No")
    OnlineSecurity: str = Field(..., description="Yes/No/No internet service")
    OnlineBackup: str = Field(..., description="Yes/No/No internet service")
    DeviceProtection: str = Field(..., description="Yes/No/No internet service")
    TechSupport: str = Field(..., description="Yes/No/No internet service")
    StreamingTV: str = Field(..., description="Yes/No/No internet service")
    StreamingMovies: str = Field(..., description="Yes/No/No internet service")
    Contract: str = Field(..., description="Month-to-month/One year/Two year")
    PaperlessBilling: str = Field(..., description="Yes/No")
    PaymentMethod: str = Field(..., description="Electronic check/Mailed check/Bank transfer/Credit card")
    MonthlyCharges: float = Field(..., ge=0, description="Charges mensuelles")
    TotalCharges: float = Field(..., ge=0, description="Charges totales")

    class Config:
        schema_extra = {
            "example": {
                "customerID": "7590-VHVEG",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85
            }
        }


class PredictionResponse(BaseModel):
    """Réponse de prédiction"""
    customerID: Optional[str]
    churn_probability: float = Field(..., description="Probabilité de churn (0-1)")
    churn_prediction: int = Field(..., description="Prédiction: 0=Non, 1=Oui")
    risk_level: str = Field(..., description="Low/Medium/High/Very High")
    recommended_action: str = Field(..., description="Action recommandée")
    confidence: float = Field(..., description="Confiance du modèle")
    model_version: str = Field(..., description="Version du modèle")


# Fonctions utilitaires
def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    """Vérifier la clé API"""
    if api_key is None or api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key"
        )
    return api_key


def get_recommended_action(risk_level: str) -> str:
    """Recommandations par niveau de risque"""
    actions = {
        "Low": "Maintenir relation standard, opportunités de cross-selling",
        "Medium": "Monitoring régulier, programme de fidélité, communication personnalisée",
        "High": "Campagne de rétention proactive, survey de satisfaction, proposition d'upgrade",
        "Very High": "Contact immédiat service rétention, offre spéciale -30%, appel personnalisé"
    }
    return actions.get(risk_level, "Contacter le service client")


def calculate_risk_level(probability: float) -> str:
    """Calculer le niveau de risque"""
    if probability < 0.3:
        return "Low"
    elif probability < 0.5:
        return "Medium"
    elif probability < 0.7:
        return "High"
    else:
        return "Very High"


def calculate_confidence(probability: float) -> float:
    """Calculer la confiance (distance au threshold)"""
    threshold = MODEL_PACKAGE['threshold']
    distance = abs(probability - threshold)
    return min(1.0, distance * 2)  # Normaliser


# Endpoints
@app.get("/", tags=["Info"])
async def root():
    """Page d'accueil de l'API"""
    return {
        "message": "Churn Prediction API",
        "version": API_VERSION,
        "model_version": MODEL_VERSION,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", tags=["Monitoring"])
async def health_check():
    """Health check pour monitoring"""
    if MODEL_PACKAGE is None or PREPROCESSOR is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    return {
        "status": "healthy",
        "api_version": API_VERSION,
        "model_version": MODEL_VERSION,
        "model_loaded": MODEL_PACKAGE is not None
    }


@app.get("/metrics", tags=["Monitoring"])
async def get_model_metrics():
    """Métriques du modèle"""
    if MODEL_PACKAGE is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    return {
        "model_version": MODEL_VERSION,
        "metrics": MODEL_PACKAGE['metrics'],
        "threshold": MODEL_PACKAGE['threshold'],
        "models": MODEL_PACKAGE['models'],
        "weights": MODEL_PACKAGE['weights']
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_churn(
    customer: Customer,
    api_key: str = Security(verify_api_key)
):
    """
    Prédire le risque de churn pour un client.

    **Niveau de risque:**
    - Low (<30%): Relation standard
    - Medium (30-50%): Monitoring régulier
    - High (50-70%): Rétention proactive
    - Very High (>70%): Action immédiate
    """
    try:
        # Vérifier que le modèle est chargé
        if MODEL_PACKAGE is None or PREPROCESSOR is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not ready"
            )

        # Conversion en DataFrame
        customer_dict = customer.dict()
        customer_id = customer_dict.pop('customerID', None)
        df = pd.DataFrame([customer_dict])

        # Preprocessing
        X_processed = preprocess_data_for_prediction(df, PREPROCESSOR)

        # Prédiction
        model = MODEL_PACKAGE['ensemble']
        threshold = MODEL_PACKAGE['threshold']

        probability = float(model.predict_proba(X_processed)[0, 1])
        prediction = int(probability >= threshold)

        # Calculs
        risk_level = calculate_risk_level(probability)
        recommended_action = get_recommended_action(risk_level)
        confidence = calculate_confidence(probability)

        return PredictionResponse(
            customerID=customer_id,
            churn_probability=probability,
            churn_prediction=prediction,
            risk_level=risk_level,
            recommended_action=recommended_action,
            confidence=confidence,
            model_version=MODEL_VERSION
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(
    customers: list[Customer],
    api_key: str = Security(verify_api_key)
):
    """
    Prédictions batch pour plusieurs clients.
    Limite: 1000 clients par requête.
    """
    if len(customers) > 1000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 1000 customers per batch"
        )

    predictions = []
    for customer in customers:
        pred = await predict_churn(customer, api_key)
        predictions.append(pred)

    return {
        "total": len(predictions),
        "predictions": predictions
    }


# Lancer avec: uvicorn api.main:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# MLOps Roadmap - Churn Prediction System

## Expert MLOps - Plan d'Action Complet

---

## 🎯 PHASE 1: CONTAINERISATION & API (Semaine 1)

### 1.1 Créer une API REST avec FastAPI

**Priorité:** 🔴 CRITIQUE

**Objectif:** Exposer le modèle via API pour intégration système

**Actions:**

```python
# production/api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from src.utils import load_object
from src.data_preprocessing import preprocess_data_for_prediction

app = FastAPI(title="Churn Prediction API", version="1.0.0")

class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.post("/predict")
async def predict_churn(customer: Customer):
    # Conversion en DataFrame
    df = pd.DataFrame([customer.dict()])

    # Preprocessing
    preprocessor = load_object('data/preprocessor.joblib')
    X = preprocess_data_for_prediction(df, preprocessor)

    # Prédiction
    model_pkg = load_object('models/churn_model_v1.joblib')
    proba = model_pkg['ensemble'].predict_proba(X)[0, 1]
    pred = int(proba >= model_pkg['threshold'])

    # Risk level
    if proba < 0.3:
        risk = "Low"
    elif proba < 0.5:
        risk = "Medium"
    elif proba < 0.7:
        risk = "High"
    else:
        risk = "Very High"

    return {
        "churn_probability": float(proba),
        "churn_prediction": pred,
        "risk_level": risk,
        "recommended_action": get_action(risk)
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_version": "1.0.0"}

@app.get("/metrics")
async def model_metrics():
    model_pkg = load_object('models/churn_model_v1.joblib')
    return model_pkg['metrics']
```

**Fichiers à créer:**
- `production/api/main.py` - API FastAPI
- `production/api/requirements.txt` - Dépendances API
- `production/api/Dockerfile` - Container Docker

**Tests:**
```bash
# Lancer l'API
uvicorn api.main:app --reload

# Tester
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"gender":"Female", "SeniorCitizen":0, ...}'
```

---

### 1.2 Dockerisation

**Priorité:** 🔴 CRITIQUE

**Dockerfile:**

```dockerfile
# production/Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Installer dépendances système
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copier requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code
COPY . .

# Exposer le port
EXPOSE 8000

# Lancer l'API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**docker-compose.yml:**

```yaml
version: '3.8'

services:
  churn-api:
    build: ./production
    ports:
      - "8000:8000"
    environment:
      - MODEL_VERSION=1.0.0
      - LOG_LEVEL=INFO
    volumes:
      - ./production/models:/app/models:ro
      - ./production/data:/app/data:ro
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

**Commandes:**
```bash
# Build
docker build -t churn-prediction:v1.0.0 ./production

# Run
docker run -p 8000:8000 churn-prediction:v1.0.0

# Docker Compose
docker-compose up -d
```

---

## 🔄 PHASE 2: CI/CD PIPELINE (Semaine 2)

### 2.1 GitHub Actions pour CI/CD

**Priorité:** 🟡 IMPORTANTE

**.github/workflows/ci-cd.yml:**

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov

      - name: Run tests
        run: |
          pytest tests/ --cov=src --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml

  build-and-push:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3

      - name: Login to Docker Hub
        uses: docker/login-action@v2
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}

      - name: Build and push
        uses: docker/build-push-action@v4
        with:
          context: ./production
          push: true
          tags: |
            ${{ secrets.DOCKER_USERNAME }}/churn-prediction:latest
            ${{ secrets.DOCKER_USERNAME }}/churn-prediction:${{ github.sha }}

  deploy:
    needs: build-and-push
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to production
        run: |
          # SSH to server and update container
          ssh ${{ secrets.PROD_SERVER }} \
            'docker pull your-repo/churn-prediction:latest && \
             docker-compose up -d'
```

---

## 📊 PHASE 3: MONITORING & OBSERVABILITÉ (Semaine 3)

### 3.1 Prometheus + Grafana

**Priorité:** 🟡 IMPORTANTE

**prometheus.yml:**

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'churn-api'
    static_configs:
      - targets: ['churn-api:8000']
```

**Métriques à tracker:**

```python
# production/api/metrics.py
from prometheus_client import Counter, Histogram, Gauge
from prometheus_fastapi_instrumentator import Instrumentator

# Compteurs
predictions_total = Counter(
    'churn_predictions_total',
    'Total predictions made',
    ['prediction', 'risk_level']
)

# Histogrammes
prediction_latency = Histogram(
    'churn_prediction_duration_seconds',
    'Time to make prediction'
)

# Jauges
model_f1_score = Gauge(
    'churn_model_f1_score',
    'Current model F1 score'
)

# Ajouter à l'API
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)
```

**Dashboard Grafana:**
- Nombre de prédictions/minute
- Latence moyenne
- Distribution des risques
- Taux d'erreurs
- Drift des features

---

### 3.2 Logging Structuré

**Priorité:** 🟡 IMPORTANTE

**production/api/logging_config.py:**

```python
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName
        }

        if hasattr(record, 'prediction'):
            log_data['prediction'] = record.prediction

        return json.dumps(log_data)

# Configuration
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/var/log/churn-api/predictions.log')
    ]
)

logger = logging.getLogger(__name__)
logger.handlers[0].setFormatter(JSONFormatter())
```

**Utilisation:**

```python
@app.post("/predict")
async def predict_churn(customer: Customer):
    logger.info(
        "Prediction request received",
        extra={'customer_id': customer.customerID}
    )

    # ... prédiction ...

    logger.info(
        "Prediction completed",
        extra={
            'prediction': pred,
            'probability': proba,
            'risk': risk
        }
    )
```

---

## 🔍 PHASE 4: MODEL MONITORING (Semaine 4)

### 4.1 Détection de Data Drift

**Priorité:** 🔴 CRITIQUE

**production/monitoring/drift_detector.py:**

```python
from scipy.stats import ks_2samp
import pandas as pd
import numpy as np

class DriftDetector:
    def __init__(self, reference_data, threshold=0.05):
        self.reference_data = reference_data
        self.threshold = threshold

    def detect_drift(self, new_data):
        """Détecte le drift pour chaque feature"""
        drift_report = {}

        for col in self.reference_data.columns:
            if col in new_data.columns:
                # KS test pour features numériques
                if pd.api.types.is_numeric_dtype(self.reference_data[col]):
                    statistic, pvalue = ks_2samp(
                        self.reference_data[col],
                        new_data[col]
                    )

                    drift_report[col] = {
                        'drift_detected': pvalue < self.threshold,
                        'p_value': pvalue,
                        'statistic': statistic
                    }

        return drift_report

    def alert_if_drift(self, drift_report):
        """Envoie alerte si drift détecté"""
        drifted_features = [
            f for f, r in drift_report.items()
            if r['drift_detected']
        ]

        if len(drifted_features) > 0:
            # Envoyer alerte (email, Slack, etc.)
            print(f"⚠️ DRIFT DETECTED: {drifted_features}")
            return True
        return False
```

---

### 4.2 Performance Monitoring

**Priorité:** 🔴 CRITIQUE

**production/monitoring/performance_tracker.py:**

```python
from sklearn.metrics import f1_score, precision_score, recall_score
import pandas as pd
from datetime import datetime

class PerformanceTracker:
    def __init__(self, db_connection):
        self.db = db_connection

    def track_prediction(self, customer_id, prediction, probability, timestamp):
        """Enregistre la prédiction"""
        self.db.execute("""
            INSERT INTO predictions (customer_id, prediction, probability, timestamp)
            VALUES (?, ?, ?, ?)
        """, (customer_id, prediction, probability, timestamp))

    def update_ground_truth(self, customer_id, actual_churn):
        """Met à jour avec la réalité"""
        self.db.execute("""
            UPDATE predictions
            SET actual_churn = ?, evaluated_at = ?
            WHERE customer_id = ?
        """, (actual_churn, datetime.now(), customer_id))

    def calculate_monthly_metrics(self):
        """Calcule les métriques du mois"""
        df = pd.read_sql("""
            SELECT prediction, actual_churn
            FROM predictions
            WHERE evaluated_at IS NOT NULL
            AND evaluated_at >= DATE('now', '-30 days')
        """, self.db)

        if len(df) > 0:
            metrics = {
                'f1_score': f1_score(df['actual_churn'], df['prediction']),
                'precision': precision_score(df['actual_churn'], df['prediction']),
                'recall': recall_score(df['actual_churn'], df['prediction']),
                'n_samples': len(df)
            }

            # Alerte si F1 < 60%
            if metrics['f1_score'] < 0.60:
                self.send_alert(f"⚠️ F1-Score dropped to {metrics['f1_score']:.2%}")

            return metrics
        return None
```

---

## 🗄️ PHASE 5: MODEL VERSIONING & REGISTRY (Semaine 5)

### 5.1 MLflow Model Registry

**Priorité:** 🟡 IMPORTANTE

**production/mlflow_registry.py:**

```python
import mlflow
from mlflow.tracking import MlflowClient

class ModelRegistry:
    def __init__(self):
        mlflow.set_tracking_uri("http://mlflow-server:5000")
        self.client = MlflowClient()

    def register_model(self, model, model_name, metrics):
        """Enregistre un nouveau modèle"""
        with mlflow.start_run():
            # Log métriques
            mlflow.log_metrics(metrics)

            # Log modèle
            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name=model_name
            )

            # Ajouter tags
            mlflow.set_tag("stage", "staging")
            mlflow.set_tag("f1_score", metrics['test_f1'])

    def promote_to_production(self, model_name, version):
        """Promouvoir en production"""
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production"
        )

    def get_production_model(self, model_name):
        """Récupère le modèle en production"""
        model_uri = f"models:/{model_name}/Production"
        return mlflow.sklearn.load_model(model_uri)
```

---

## 🔐 PHASE 6: SÉCURITÉ & GOVERNANCE (Semaine 6)

### 6.1 Authentification API

**Priorité:** 🔴 CRITIQUE

```python
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    if api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )
    return api_key

@app.post("/predict")
async def predict_churn(
    customer: Customer,
    api_key: str = Security(verify_api_key)
):
    # ... prédiction ...
```

---

### 6.2 Rate Limiting

**Priorité:** 🟡 IMPORTANTE

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/predict")
@limiter.limit("100/minute")
async def predict_churn(request: Request, customer: Customer):
    # ... prédiction ...
```

---

## 📈 PHASE 7: SCALABILITÉ (Semaine 7-8)

### 7.1 Kubernetes Deployment

**Priorité:** 🟢 BONNE PRATIQUE

**k8s/deployment.yaml:**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: churn-prediction
spec:
  replicas: 3
  selector:
    matchLabels:
      app: churn-prediction
  template:
    metadata:
      labels:
        app: churn-prediction
    spec:
      containers:
      - name: api
        image: your-repo/churn-prediction:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: churn-prediction-service
spec:
  selector:
    app: churn-prediction
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: churn-prediction-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: churn-prediction
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

---

## 🧪 PHASE 8: TESTING AVANCÉ (Continu)

### 8.1 Tests de Performance

**tests/performance/load_test.py:**

```python
from locust import HttpUser, task, between

class ChurnPredictionUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def predict(self):
        payload = {
            "gender": "Female",
            "SeniorCitizen": 0,
            # ... autres champs
        }

        self.client.post(
            "/predict",
            json=payload,
            headers={"X-API-Key": "test-key"}
        )
```

**Commandes:**
```bash
# Load test
locust -f tests/performance/load_test.py --host=http://localhost:8000

# Target: 100 RPS sans dégradation
```

---

### 8.2 Tests d'Intégration

**tests/integration/test_api.py:**

```python
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_predict_endpoint():
    response = client.post("/predict", json={
        "gender": "Female",
        "SeniorCitizen": 0,
        # ... données valides
    })

    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert 0 <= data["churn_probability"] <= 1

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
```

---

## 📋 PRIORISATION DES TÂCHES

### Semaine 1-2: DÉPLOIEMENT MINIMAL VIABLE
1. ✅ API FastAPI
2. ✅ Dockerisation
3. ✅ CI/CD basique
4. ✅ Logging

### Semaine 3-4: MONITORING
5. ✅ Prometheus + Grafana
6. ✅ Performance tracking
7. ✅ Drift detection

### Semaine 5-6: PRODUCTION-READY
8. ✅ Model registry
9. ✅ Sécurité (API keys, rate limiting)
10. ✅ Tests automatisés

### Semaine 7-8: SCALABILITÉ
11. ✅ Kubernetes
12. ✅ Auto-scaling
13. ✅ Load balancing

---

## 🎯 MÉTRIQUES DE SUCCÈS MLOPS

| Métrique | Cible | Actuel |
|----------|-------|--------|
| Uptime | >99.9% | - |
| Latence P95 | <100ms | - |
| Throughput | >1000 RPS | - |
| Déploiement | <10 min | - |
| Rollback | <2 min | - |
| MTTR | <30 min | - |

---

## 🛠️ STACK TECHNOLOGIQUE RECOMMANDÉ

**Infrastructure:**
- ☁️ Cloud: AWS/GCP/Azure
- 🐳 Containers: Docker + Kubernetes
- 🔄 CI/CD: GitHub Actions
- 📊 Monitoring: Prometheus + Grafana
- 📝 Logging: ELK Stack (Elasticsearch, Logstash, Kibana)

**ML:**
- 🧪 Experiment Tracking: MLflow
- 📦 Model Registry: MLflow / AWS SageMaker
- 🔍 Feature Store: Feast (optionnel)
- 📈 Drift Detection: Evidently AI

**API:**
- ⚡ Framework: FastAPI
- 🔐 Auth: JWT / API Keys
- 🚦 Rate Limiting: SlowAPI
- 📖 Documentation: Swagger (auto-généré)

---

## 📞 PROCHAINES ACTIONS IMMÉDIATES

**AUJOURD'HUI:**
1. Créer `production/api/main.py`
2. Tester l'API localement
3. Créer Dockerfile

**CETTE SEMAINE:**
1. Déployer sur serveur de test
2. Configurer CI/CD GitHub Actions
3. Implémenter logging basique

**CE MOIS:**
1. Monitoring Prometheus/Grafana
2. Performance tracking
3. Drift detection

---

**MLOps Expert Contact:** votre-email@example.com
**Documentation:** docs.mlops-project.com
**Status Dashboard:** grafana.mlops-project.com

---

**Version:** 1.0
**Date:** 2025-11-10
**Statut:** 🚀 Ready to Scale

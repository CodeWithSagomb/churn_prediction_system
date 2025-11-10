# Quick Start - API FastAPI

## 🚀 Démarrage en 5 Minutes

### Option 1: Docker (RECOMMANDÉ)

```bash
# 1. Build et lancer avec Docker Compose
docker-compose up -d

# 2. Vérifier que l'API fonctionne
curl http://localhost:8000/health

# 3. Accéder à la documentation
open http://localhost:8000/docs
```

**Résultat:** API + Prometheus + Grafana démarrés!

---

### Option 2: Local (Pour développement)

```bash
# 1. Installer les dépendances
cd production
pip install -r requirements.txt

# 2. Lancer l'API
uvicorn api.main:app --reload --port 8000

# 3. Tester
curl http://localhost:8000/health
```

---

## 📡 Endpoints Disponibles

### 1. Health Check

```bash
curl http://localhost:8000/health
```

**Réponse:**
```json
{
  "status": "healthy",
  "api_version": "1.0.0",
  "model_version": "1.0.0",
  "model_loaded": true
}
```

---

### 2. Prédiction Simple

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "X-API-Key: demo-key-123" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

**Réponse:**
```json
{
  "customerID": null,
  "churn_probability": 0.7234,
  "churn_prediction": 1,
  "risk_level": "Very High",
  "recommended_action": "Contact immédiat service rétention, offre spéciale -30%, appel personnalisé",
  "confidence": 0.95,
  "model_version": "1.0.0"
}
```

---

### 3. Métriques du Modèle

```bash
curl "http://localhost:8000/metrics"
```

**Réponse:**
```json
{
  "model_version": "1.0.0",
  "metrics": {
    "test_f1": 0.6181,
    "test_precision": 0.5556,
    "test_recall": 0.6964,
    "test_roc_auc": 0.8278
  },
  "threshold": 0.550,
  "models": ["RandomForest", "XGBoost", "CatBoost", "LightGBM"],
  "weights": [0.16, 0.84, 0.04, 2.96]
}
```

---

### 4. Prédiction Batch (Plusieurs clients)

```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "X-API-Key: demo-key-123" \
  -H "Content-Type: application/json" \
  -d '[
    {
      "gender": "Male",
      "SeniorCitizen": 1,
      ...
    },
    {
      "gender": "Female",
      "SeniorCitizen": 0,
      ...
    }
  ]'
```

---

## 🔐 Sécurité

### API Keys

L'API utilise des clés d'authentification.

**Clés de démo:**
- `demo-key-123` - Pour tests
- `prod-key-456` - Pour production (à changer!)

**En-tête requis:**
```
X-API-Key: demo-key-123
```

**Erreur si clé manquante:**
```json
{
  "detail": "Invalid or missing API Key"
}
```

---

## 📊 Monitoring

### Accès aux Dashboards

**Grafana:** http://localhost:3000
- Login: admin
- Password: admin123

**Prometheus:** http://localhost:9090

### Métriques Disponibles

- Nombre de prédictions total
- Latence des prédictions
- Distribution des niveaux de risque
- Taux d'erreurs

---

## 🧪 Tests

### Test Python

```python
import requests

# URL de l'API
API_URL = "http://localhost:8000"
API_KEY = "demo-key-123"

# Données client
customer = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 89.95,
    "TotalCharges": 1079.40
}

# Requête
response = requests.post(
    f"{API_URL}/predict",
    json=customer,
    headers={"X-API-Key": API_KEY}
)

# Résultat
result = response.json()
print(f"Churn Probability: {result['churn_probability']:.2%}")
print(f"Risk Level: {result['risk_level']}")
print(f"Action: {result['recommended_action']}")
```

---

## 📖 Documentation Interactive

L'API génère automatiquement une documentation interactive Swagger:

**Swagger UI:** http://localhost:8000/docs

**ReDoc:** http://localhost:8000/redoc

**Fonctionnalités:**
- Tester tous les endpoints
- Voir les schémas de données
- Exemples de requêtes/réponses
- Télécharger le schéma OpenAPI

---

## 🛠️ Commandes Utiles

### Docker

```bash
# Démarrer
docker-compose up -d

# Arrêter
docker-compose down

# Logs
docker-compose logs -f churn-api

# Rebuild
docker-compose up --build -d

# Redémarrer un service
docker-compose restart churn-api
```

### Développement Local

```bash
# Lancer avec reload auto
uvicorn api.main:app --reload --port 8000

# Lancer en mode debug
uvicorn api.main:app --reload --log-level debug

# Spécifier l'hôte
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

---

## 🔧 Troubleshooting

### Erreur: "Model not loaded"

**Solution:**
```bash
# Vérifier que les fichiers existent
ls production/models/churn_model_v1.joblib
ls production/data/preprocessor.joblib

# Vérifier les permissions
chmod +r production/models/churn_model_v1.joblib
```

### Erreur: Port 8000 déjà utilisé

**Solution:**
```bash
# Trouver le processus
lsof -i :8000

# Tuer le processus
kill -9 <PID>

# Ou utiliser un autre port
uvicorn api.main:app --port 8001
```

### Performance lente

**Diagnostics:**
```bash
# Vérifier les ressources Docker
docker stats

# Vérifier les logs
docker-compose logs churn-api

# Tester la latence
time curl -X POST http://localhost:8000/predict \
  -H "X-API-Key: demo-key-123" \
  -H "Content-Type: application/json" \
  -d @test_customer.json
```

---

## 📈 Performance Attendue

| Métrique | Valeur |
|----------|--------|
| Latence P50 | <50ms |
| Latence P95 | <100ms |
| Latence P99 | <200ms |
| Throughput | >100 RPS |

---

## 🚀 Prochaines Étapes

1. **Tester l'API** avec vos données
2. **Configurer monitoring** Grafana
3. **Déployer en production** (Kubernetes)
4. **Implémenter logging** avancé
5. **Ajouter rate limiting**

---

## 📞 Support

**Documentation complète:** `MLOPS_ROADMAP.md`
**API Docs:** http://localhost:8000/docs
**Issues:** GitHub Issues

---

**Version:** 1.0.0
**Date:** 2025-11-10
**Statut:** ✅ Production Ready

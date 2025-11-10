# Guide d'Utilisation - Churn Prediction System

## 1. API (http://localhost:8000)

### Swagger UI (Interface Interactive)
1. Ouvrir: http://localhost:8000/docs
2. Tester l'endpoint `/predict`:
   - Cliquer "Try it out"
   - API Key: `demo-key-123`
   - Utiliser le JSON de test ci-dessous
   - Cliquer "Execute"

### Exemple de Client (JSON)
```json
{
  "customerID": "TEST-001",
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
```

### Tests avec curl
```bash
# Health check
curl http://localhost:8000/health

# Métriques du modèle
curl http://localhost:8000/metrics

# Prédiction
curl -X POST http://localhost:8000/predict \
  -H "X-API-Key: demo-key-123" \
  -H "Content-Type: application/json" \
  -d @tests/fixtures/test_customer.json
```

### Tests Python
```bash
cd tests/fixtures
PYTHONIOENCODING=utf-8 python test_api.py
```

---

## 2. Prometheus (http://localhost:9090)

### Vérifier les Métriques
1. Ouvrir: http://localhost:9090
2. Aller dans "Status" → "Targets"
3. Vérifier que `churn-prediction-api` est "UP"

### Requêtes PromQL Utiles

**Statut des services:**
```promql
up{job="churn-prediction-api"}
```

**Requêtes HTTP par seconde:**
```promql
rate(http_requests_total[1m])
```

**Latence moyenne (ms):**
```promql
rate(http_request_duration_seconds_sum[1m]) / rate(http_request_duration_seconds_count[1m]) * 1000
```

**Total de prédictions:**
```promql
churn_predictions_total
```

**Ratio de churn détecté:**
```promql
churn_predictions_total{prediction="1"} / sum(churn_predictions_total)
```

### Créer un Graphique
1. Taper une requête (ex: `up`)
2. Cliquer "Execute"
3. Onglet "Graph" pour visualiser
4. Ajuster la période (Last 1h, Last 6h, etc.)

---

## 3. Grafana (http://localhost:3000)

### Connexion
```
Username: admin
Password: admin123
```

### Configuration Initiale

**1. Ajouter Prometheus comme source:**
- Menu ☰ → Connections → Data sources
- "Add data source" → Prometheus
- URL: `http://prometheus:9090`
- "Save & Test" (doit être vert ✅)

**2. Créer un Dashboard:**
- Menu ☰ → Dashboards → New → New Dashboard
- "Add visualization"

### Panneaux Recommandés

**Panel 1: API Status**
- Query: `up{job="churn-prediction-api"}`
- Visualization: Stat
- Title: "API Status"

**Panel 2: Request Rate**
- Query: `rate(http_requests_total[5m])`
- Visualization: Time series
- Title: "Requests/sec"

**Panel 3: Latency**
- Query:
  ```promql
  rate(http_request_duration_seconds_sum[5m]) /
  rate(http_request_duration_seconds_count[5m]) * 1000
  ```
- Visualization: Time series
- Title: "Latency (ms)"
- Unit: milliseconds (ms)

**Panel 4: Churn Predictions**
- Query: `rate(churn_predictions_total{prediction="1"}[5m])`
- Visualization: Time series
- Title: "Churn Predictions/sec"

**Panel 5: Risk Distribution**
- Query: `churn_predictions_by_risk`
- Visualization: Pie chart
- Title: "Risk Level Distribution"

### Alertes (Optionnel)

**Alerte: API Down**
- Condition: `up{job="churn-prediction-api"} == 0`
- For: 1m
- Alert: Envoyer notification

**Alerte: High Latency**
- Condition: `avg_latency > 200`
- For: 5m
- Alert: Performance dégradée

---

## 4. Workflow Complet

### Scénario: Prédire le Churn d'un Client

1. **Préparer les données du client** (JSON)
2. **Appeler l'API:**
   ```bash
   curl -X POST http://localhost:8000/predict \
     -H "X-API-Key: demo-key-123" \
     -H "Content-Type: application/json" \
     -d '{...customer data...}'
   ```
3. **Analyser la réponse:**
   ```json
   {
     "customerID": "TEST-001",
     "churn_probability": 0.69,
     "churn_prediction": 1,
     "risk_level": "High",
     "recommended_action": "Campagne de rétention...",
     "confidence": 0.29
   }
   ```
4. **Vérifier les métriques dans Prometheus:**
   - Aller sur http://localhost:9090
   - Query: `churn_predictions_total`
5. **Visualiser dans Grafana:**
   - Voir le dashboard en temps réel
   - Analyser les tendances

---

## 5. Commandes Utiles

### Docker
```bash
# Voir les logs
docker-compose logs -f churn-api

# Redémarrer un service
docker-compose restart churn-api

# Arrêter tout
docker-compose down

# Redémarrer tout
docker-compose up -d
```

### Tests
```bash
# Test complet de l'API
PYTHONIOENCODING=utf-8 python tests/fixtures/test_api.py

# Test d'un endpoint spécifique
curl http://localhost:8000/health
```

### Monitoring
```bash
# Vérifier le statut des containers
docker-compose ps

# Métriques système
docker stats
```

---

## 6. Troubleshooting

**Problème: API ne répond pas**
```bash
docker-compose logs churn-api
docker-compose restart churn-api
```

**Problème: Prometheus ne collecte pas**
- Vérifier: http://localhost:9090/targets
- Vérifier que l'API expose `/metrics`

**Problème: Grafana ne se connecte pas à Prometheus**
- URL correcte: `http://prometheus:9090` (pas localhost)
- Vérifier la source de données: Connections → Data sources

---

## 7. Ressources

- **API Docs**: http://localhost:8000/docs
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000
- **Tests**: `tests/fixtures/test_api.py`
- **Documentation complète**: `docs/QUICK_START_API.md`

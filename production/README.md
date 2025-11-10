# Churn Prediction System - Production Package

## Modele de Production

**Version:** 1.0
**Date:** 2025-11-10
**Performance Test:** F1-Score = 0.6181

### Metriques du Modele

| Metrique | Valeur |
|----------|--------|
| F1-Score | 0.6181 |
| Precision | 0.5556 |
| Recall | 0.6964 |
| ROC-AUC | 0.8278 |
| Threshold | 0.550 |

### Description du Modele

- **Type:** Voting Ensemble (Soft Voting)
- **Modeles inclus:** RandomForest, XGBoost, CatBoost, LightGBM
- **Poids:** [0.16, 0.84, 0.04, 2.96]
- **Features:** 68 (avec feature engineering)
- **Techniques:** SMOTE + Feature Engineering + Threshold Optimization

### Structure du Package

```
production/
├── models/
│   └── churn_model_v1.joblib          # Modele de production
├── data/
│   └── preprocessor.joblib             # Pipeline de preprocessing
├── src/
│   ├── utils.py                        # Utilitaires
│   ├── data_preprocessing.py           # Preprocessing
│   └── feature_engineering.py          # Feature engineering
├── predict.py                          # Script de prediction
├── params.yml                          # Configuration
└── README.md                           # Cette documentation

```

### Installation

```bash
pip install pandas numpy scikit-learn xgboost catboost lightgbm imbalanced-learn joblib pyyaml
```

### Utilisation

#### Prediction sur nouvelles donnees

```bash
python predict.py --input data/new_customers.csv --output predictions.csv
```

#### En Python

```python
from src.utils import load_object

# Charger le modele
model_package = load_object('models/churn_model_v1.joblib')
model = model_package['ensemble']
threshold = model_package['threshold']

# Predire
probabilities = model.predict_proba(X_processed)[:, 1]
predictions = (probabilities >= threshold).astype(int)
```

### Format des Donnees d'Entree

Le fichier CSV doit contenir les colonnes suivantes:

**Colonnes requises:**
- customerID (optionnel)
- gender
- SeniorCitizen
- Partner
- Dependents
- tenure
- PhoneService
- MultipleLines
- InternetService
- OnlineSecurity
- OnlineBackup
- DeviceProtection
- TechSupport
- StreamingTV
- StreamingMovies
- Contract
- PaperlessBilling
- PaymentMethod
- MonthlyCharges
- TotalCharges

### Interpretation des Predictions

- **churn_probability:** Probabilite de churn (0.0 - 1.0)
- **churn_prediction:** Prediction binaire (0 = Non, 1 = Oui)
- **churn_risk:** Niveau de risque
  - Low: probabilite < 30%
  - Medium: 30% <= probabilite < 50%
  - High: 50% <= probabilite < 70%
  - Very High: probabilite >= 70%

### Actions Recommandees par Niveau de Risque

**Very High Risk (>70%):**
- Contact immediat du service retention
- Offre speciale personnalisee
- Remise importante

**High Risk (50-70%):**
- Campagne de retention proactive
- Survey de satisfaction
- Proposition d'upgrade

**Medium Risk (30-50%):**
- Monitoring regulier
- Programme de fidelite
- Communication personnalisee

**Low Risk (<30%):**
- Maintenance relation standard
- Cross-selling opportunites

### Performance et Limites

**Points forts:**
- Excellent ROC-AUC (82.8%) - bonne discrimination
- Recall de 69.6% - capture bien les churners
- Modele stable (pas d'overfitting)

**Limites:**
- Precision moderee (55.6%) - ~45% de faux positifs
- Performance plafonnee a 62% F1 avec ce dataset
- Sensible au threshold (optimizer selon le cout metier)

### Maintenance

**Re-entrainement recommande:**
- Tous les 3 mois minimum
- Apres changements majeurs dans l'offre
- Si performance < 60% F1

**Monitoring:**
- Suivre le taux de churn reel vs predit
- Calculer le F1-Score mensuel
- Analyser les faux positifs/negatifs

### Contact

Pour questions ou support: votre-email@example.com

---

**Version History:**
- v1.0 (2025-11-10): Version initiale de production

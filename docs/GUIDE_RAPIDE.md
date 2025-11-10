# Guide Rapide - Churn Prediction System

## Démarrage Rapide (5 minutes)

### 1. Installation

```bash
# Cloner le projet (ou télécharger)
cd churn_prediction_system

# Créer environnement virtuel
python -m venv .venv
.venv\Scripts\activate  # Windows

# Installer dépendances
pip install -r production/requirements.txt
```

### 2. Prédiction Immédiate

```bash
cd production
python predict.py --input ../data/WA_Fn-UseC_-Telco-Customer-Churn.csv --output predictions.csv
```

Résultat: Fichier `predictions.csv` avec probabilités et prédictions de churn

## Commandes Essentielles

### Production

```bash
# Prédiction sur nouvelles données
cd production
python predict.py --input data.csv --output results.csv

# Avec threshold personnalisé
python predict.py --input data.csv --output results.csv --threshold 0.6
```

### Développement

```bash
# Entraîner nouveaux modèles
python src/model_training_v2.py

# Tests
pytest tests/

# MLflow UI (voir expériences)
mlflow ui
# Ouvrir http://localhost:5000
```

## Structure Minimale pour Production

Pour déployer, copier seulement le dossier `production/`:

```
production/
├── models/churn_model_v1.joblib    # Modèle
├── data/preprocessor.joblib         # Preprocessing
├── src/                             # Code
├── predict.py                       # Script principal
├── requirements.txt                 # Dépendances
└── README.md                        # Doc
```

## Exemples d'Utilisation

### Python Script

```python
from src.utils import load_object
from src.data_preprocessing import preprocess_data_for_prediction
import pandas as pd

# Charger
model_pkg = load_object('production/models/churn_model_v1.joblib')
preprocessor = load_object('production/data/preprocessor.joblib')

# Données
df = pd.read_csv('nouveaux_clients.csv')

# Prédire
X = preprocess_data_for_prediction(df, preprocessor)
proba = model_pkg['ensemble'].predict_proba(X)[:, 1]
pred = (proba >= model_pkg['threshold']).astype(int)

# Résultats
df['churn_proba'] = proba
df['churn_pred'] = pred
```

### Jupyter Notebook

```python
%load_ext autoreload
%autoreload 2

from src.utils import load_object
import pandas as pd

# Charger modèle
model = load_object('production/models/churn_model_v1.joblib')

# Analyse
print(f"F1-Score: {model['metrics']['test_f1']:.4f}")
print(f"Threshold: {model['threshold']:.3f}")
print(f"Modèles: {model['models']}")
print(f"Poids: {model['weights']}")
```

## Format des Données d'Entrée

**Colonnes requises (20):**

```csv
customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,
MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,
TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,
PaymentMethod,MonthlyCharges,TotalCharges
```

**Exemple de ligne:**
```
7590-VHVEG,Female,0,Yes,No,1,No,No phone service,DSL,No,Yes,No,No,No,No,Month-to-month,Yes,Electronic check,29.85,29.85
```

## Interprétation des Résultats

### Colonnes de Sortie

| Colonne | Description | Valeurs |
|---------|-------------|---------|
| `churn_probability` | Probabilité de churn | 0.0 - 1.0 |
| `churn_prediction` | Prédiction binaire | 0 (Non) / 1 (Oui) |
| `churn_risk` | Niveau de risque | Low / Medium / High / Very High |

### Actions Recommandées

**Very High Risk (>70%)**
- 🚨 Contact immédiat service rétention
- 💰 Offre spéciale -30% minimum
- 📞 Appel téléphonique personnalisé

**High Risk (50-70%)**
- ⚠️ Campagne email de rétention
- 📊 Survey de satisfaction
- 🎁 Proposition upgrade/bundle

**Medium Risk (30-50%)**
- 📧 Email personnalisé
- 🎯 Programme de fidélité
- 📈 Monitoring mensuel

**Low Risk (<30%)**
- ✅ Relation standard
- 🛍️ Cross-selling opportunités

## Métriques de Performance

### Modèle de Production (v1.0)

```
F1-Score:    61.81% ⭐
Precision:   55.56% ⚠️  (45% faux positifs)
Recall:      69.64% ✅  (70% churners détectés)
ROC-AUC:     82.78% ⭐  (excellente discrimination)
Threshold:   0.550
```

### Benchmark

| Modèle | F1-Score |
|--------|----------|
| RandomForest (individuel) | 59.5% |
| XGBoost (individuel) | 59.9% |
| CatBoost (individuel) | 60.2% |
| LightGBM (individuel) | 59.7% |
| **Ensemble Optimisé** | **61.8%** ✅ |

## Troubleshooting

### Erreur: Module 'src' not found

```bash
# Ajouter le dossier au PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:."  # Linux/Mac
set PYTHONPATH=%PYTHONPATH%;.        # Windows
```

### Erreur: Colonnes manquantes

Vérifier que votre CSV contient TOUTES les 20 colonnes requises

### Performance dégradée

1. Vérifier distribution des données (drift?)
2. Re-entraîner avec nouvelles données
3. Ajuster threshold selon coût métier

### Temps de prédiction lent

```python
# Utiliser batch prediction
batch_size = 1000
for i in range(0, len(df), batch_size):
    batch = df[i:i+batch_size]
    # predict...
```

## Performance Attendue

### Temps d'Exécution

- **Preprocessing**: ~2 secondes pour 1000 clients
- **Prédiction**: ~1 seconde pour 1000 clients
- **Total**: ~3 secondes pour 1000 clients

### Ressources

- **RAM**: ~500 MB
- **CPU**: 4 cores recommandés
- **Stockage**: ~50 MB (modèle + data)

## Maintenance

### Monitoring

```python
# Calculer métriques en production
from sklearn.metrics import f1_score

# Comparer prédictions vs réalité (après 1 mois)
y_true = df_real['Churn']
y_pred = df_predictions['churn_prediction']

f1_prod = f1_score(y_true, y_pred)
print(f"F1 en production: {f1_prod:.4f}")

# Alerte si < 60%
if f1_prod < 0.60:
    print("⚠️ PERFORMANCE DEGRADEE - Re-entraînement requis!")
```

### Re-entraînement

```bash
# 1. Collecter nouvelles données
# 2. Ajouter au dataset
# 3. Re-entraîner
python src/model_training_v2.py

# 4. Comparer performances
# 5. Déployer si meilleur
```

## Support

### Documentation

- README principal: `README.md`
- Doc production: `production/README.md`
- Ce guide: `GUIDE_RAPIDE.md`

### Ressources

- Dataset: [Kaggle Telco Churn](https://www.kaggle.com/blastchar/telco-customer-churn)
- MLflow: [Documentation](https://mlflow.org/docs/latest/index.html)
- Scikit-learn: [User Guide](https://scikit-learn.org/stable/user_guide.html)

### Contact

Pour questions: votre-email@example.com

---

**Dernière mise à jour:** 2025-11-10
**Version:** 1.0

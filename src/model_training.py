# src/model_training.py

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.pipeline import Pipeline as ImbPipeline # Renommer pour éviter confusion avec pipeline sklearn
from imblearn.over_sampling import SMOTE
import warnings

# Importer nos modules personnalisés
from src.utils import load_config, load_object
from src.data_preprocessing import load_raw_data, preprocess_and_split_data

# Ignorer les avertissements futurs de scikit-learn pour un output plus propre
warnings.filterwarnings("ignore", category=FutureWarning)

def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> dict:
    """
    Calcule et retourne un dictionnaire de métriques d'évaluation.
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'roc_auc_score': roc_auc_score(y_true, y_pred_proba)
    }
    return metrics

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> str:
    """
    Crée et sauvegarde la matrice de confusion en tant qu'image.
    Retourne le chemin du fichier sauvegardé.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Churn', 'Churn'],
                yticklabels=['Not Churn', 'Churn'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    
    # Sauvegarder la figure dans un fichier temporaire
    file_path = f"confusion_matrix_{model_name}.png"
    plt.savefig(file_path)
    plt.close() # Fermer la figure pour libérer la mémoire
    return file_path

def train_and_evaluate():
    """
    Fonction principale pour orchestrer l'entraînement, l'évaluation
    et le suivi des modèles avec MLflow.
    """
    print("--- Démarrage du processus d'entraînement et d'évaluation ---")
    
    # 1. Charger la configuration
    config = load_config()
    mlflow.set_experiment(config['training']['experiment_name'])
    
    # 2. Charger et préparer les données
    raw_df = load_raw_data(config)
    # Nous n'avons besoin que des données divisées, pas du préprocesseur ici, car il est déjà sauvegardé.
    X_train_transformed, X_test_transformed, y_train, y_test, _ = preprocess_and_split_data(raw_df, config)
    print("Données chargées et prétraitées avec succès.")
    
    # 3. Définir les modèles et leurs grilles d'hyperparamètres
    models = {
        'RandomForest': RandomForestClassifier(random_state=config['preprocessing']['random_state']),
        'XGBoost': XGBClassifier(random_state=config['preprocessing']['random_state'], use_label_encoder=False, eval_metric='logloss'),
        'MLPClassifier': MLPClassifier(random_state=config['preprocessing']['random_state'], max_iter=1000) # Augmenter max_iter pour convergence
    }
    
    param_grids = {
        'RandomForest': config['models']['RandomForest']['param_grid'],
        'XGBoost': config['models']['XGBoost']['param_grid'],
        'MLPClassifier': config['models']['MLPClassifier']['param_grid']
    }


    # 4. Boucle d'entraînement pour chaque modèle
    for model_name, model in models.items():
        print(f"\n--- Entraînement du modèle : {model_name} ---")
        
        with mlflow.start_run(run_name=f"{model_name}_Training"):
            
            # A. Enregistrer le nom du modèle comme tag pour une identification facile
            mlflow.set_tag("model_name", model_name)
            
            # B. Créer le pipeline complet : SMOTE + Modèle
            # C'est la clé pour éviter le data leakage avec SMOTE
            pipeline = ImbPipeline(steps=[
                ('smote', SMOTE(random_state=config['preprocessing']['random_state'])),
                ('model', model)
            ])
            
            # C. Configurer et exécuter RandomizedSearchCV
            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=param_grids[model_name],
                n_iter=config['training']['hyperparameter_tuning']['n_iter'],
                cv=config['training']['hyperparameter_tuning']['cv'],
                scoring=config['training']['hyperparameter_tuning']['scoring'],
                random_state=config['preprocessing']['random_state'],
                n_jobs=-1, # Utiliser tous les cœurs de CPU disponibles
                verbose=1
            )
            
            print("Lancement de la recherche d'hyperparamètres...")
            search.fit(X_train_transformed, y_train)
            
            # D. Obtenir le meilleur pipeline (SMOTE + meilleur modèle)
            best_pipeline = search.best_estimator_
            
            # E. Faire des prédictions sur l'ensemble de test (non vu par SMOTE)
            y_pred = best_pipeline.predict(X_test_transformed)
            y_pred_proba = best_pipeline.predict_proba(X_test_transformed)[:, 1]
            
            # F. Évaluer les métriques
            metrics = evaluate_metrics(y_test, y_pred, y_pred_proba)
            
            print(f"Résultats pour {model_name}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")

            # G. Enregistrer tout avec MLflow
            print("Enregistrement des résultats avec MLflow...")
            mlflow.log_params(search.best_params_)
            mlflow.log_metrics(metrics)
            
            # Sauvegarder et enregistrer la matrice de confusion
            cm_path = plot_confusion_matrix(y_test, y_pred, model_name)
            mlflow.log_artifact(cm_path, "confusion_matrix")
            
            # Enregistrer le meilleur pipeline (qui inclut SMOTE et le modèle)
            # C'est ce pipeline qui sera utilisé pour l'inférence
            mlflow.sklearn.log_model(
                sk_model=best_pipeline,
                artifact_path=f"model_{model_name}",
                registered_model_name=f"{model_name}_Churn_Model" # Optionnel : pour le Model Registry
            )

            # Enregistrer le préprocesseur utilisé pour la traçabilité complète
            mlflow.log_artifact(config['data']['preprocessor_path'], "preprocessor")
            
            print(f"Run MLflow pour {model_name} terminé avec succès.")

    print("\n--- Processus d'entraînement complet terminé ---")


if __name__ == "__main__":
    train_and_evaluate()
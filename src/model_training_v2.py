# src/model_training_v2.py

"""
Module d'entrainement de modeles pour la prediction du churn - Version 2.0

Ameliorations:
- Utilisation du validation set (70/15/15 split)
- Evaluation sur train/val/test pour detecter l'overfitting
- Support de LightGBM et CatBoost
- Hyperparameter tuning ameliore (50 iterations)
- Logs detailles et visualisations avancees
- MLflow tracking complet

Auteur: Data Science Team
Version: 2.0 - Real-Time Churn Prediction System
Date: 2025
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import warnings
from typing import Dict, Tuple, Any
import os

# Importer nos modules personnalises
from src.utils import load_config, load_object, save_object
from src.data_preprocessing import load_raw_data, preprocess_and_split_data

# Ignorer les avertissements
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
    """
    Calcule et retourne un dictionnaire complet de metriques d'evaluation.

    Args:
        y_true: Vraies etiquettes
        y_pred: Predictions binaires
        y_pred_proba: Probabilites de la classe positive

    Returns:
        Dict contenant accuracy, precision, recall, f1, roc_auc
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc_score': roc_auc_score(y_true, y_pred_proba)
    }
    return metrics


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, dataset_name: str = "test") -> str:
    """
    Cree et sauvegarde la matrice de confusion.

    Args:
        y_true: Vraies etiquettes
        y_pred: Predictions
        model_name: Nom du modele
        dataset_name: Nom du dataset (train/val/test)

    Returns:
        Chemin du fichier sauvegarde
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Churn', 'Churn'],
                yticklabels=['Not Churn', 'Churn'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name} ({dataset_name})')

    # Sauvegarder dans models/
    os.makedirs('models', exist_ok=True)
    file_path = f"models/confusion_matrix_{model_name}_{dataset_name}.png"
    plt.savefig(file_path)
    plt.close()

    return file_path


def plot_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray, model_name: str) -> str:
    """
    Cree et sauvegarde la courbe ROC.

    Args:
        y_true: Vraies etiquettes
        y_pred_proba: Probabilites predictions
        model_name: Nom du modele

    Returns:
        Chemin du fichier sauvegarde
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    os.makedirs('models', exist_ok=True)
    file_path = f"models/roc_curve_{model_name}.png"
    plt.savefig(file_path)
    plt.close()

    return file_path


def plot_precision_recall_curve(y_true: np.ndarray, y_pred_proba: np.ndarray, model_name: str) -> str:
    """
    Cree et sauvegarde la courbe Precision-Recall.

    Args:
        y_true: Vraies etiquettes
        y_pred_proba: Probabilites predictions
        model_name: Nom du modele

    Returns:
        Chemin du fichier sauvegarde
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.grid(alpha=0.3)

    os.makedirs('models', exist_ok=True)
    file_path = f"models/precision_recall_curve_{model_name}.png"
    plt.savefig(file_path)
    plt.close()

    return file_path


def print_metrics_comparison(train_metrics: Dict, val_metrics: Dict, test_metrics: Dict, model_name: str):
    """
    Affiche une comparaison des metriques sur train/val/test.

    Aide a detecter l'overfitting.
    """
    print(f"\n{'='*70}")
    print(f"  RESULTATS DETAILLES - {model_name}")
    print(f"{'='*70}")

    print(f"\n{'Metric':<15} {'Train':<12} {'Validation':<12} {'Test':<12} {'Overfitting':<12}")
    print("-" * 70)

    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc_score']:
        train_val = train_metrics[metric]
        val_val = val_metrics[metric]
        test_val = test_metrics[metric]

        # Detecter overfitting (train >> val)
        overfit_indicator = "OK" if (train_val - val_val) < 0.05 else "WARNING!"

        print(f"{metric:<15} {train_val:<12.4f} {val_val:<12.4f} {test_val:<12.4f} {overfit_indicator:<12}")

    print("-" * 70)

    # Resume
    avg_train = np.mean([train_metrics[m] for m in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc_score']])
    avg_val = np.mean([val_metrics[m] for m in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc_score']])
    avg_test = np.mean([test_metrics[m] for m in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc_score']])

    print(f"\nAverage:        {avg_train:<12.4f} {avg_val:<12.4f} {avg_test:<12.4f}")
    print(f"{'='*70}\n")


def get_active_models(config: Dict) -> Dict[str, Any]:
    """
    Retourne uniquement les modeles actifs selon la configuration.

    Args:
        config: Configuration chargee depuis params.yml

    Returns:
        Dictionnaire des modeles actifs
    """
    random_state = config['preprocessing']['random_state']
    models = {}

    # RandomForest
    if config['models']['RandomForest'].get('active', True):
        models['RandomForest'] = RandomForestClassifier(random_state=random_state, n_jobs=-1)

    # XGBoost
    if config['models']['XGBoost'].get('active', True):
        models['XGBoost'] = XGBClassifier(
            random_state=random_state,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1
        )

    # MLPClassifier
    if config['models']['MLPClassifier'].get('active', True):
        models['MLPClassifier'] = MLPClassifier(random_state=random_state, max_iter=1000)

    # LightGBM
    if config['models'].get('LightGBM', {}).get('active', False):
        models['LightGBM'] = LGBMClassifier(random_state=random_state, n_jobs=-1, verbose=-1)

    # CatBoost
    if config['models'].get('CatBoost', {}).get('active', False):
        models['CatBoost'] = CatBoostClassifier(random_state=random_state, verbose=0)

    return models


def train_and_evaluate():
    """
    Fonction principale pour orchestrer l'entrainement et l'evaluation.

    Processus:
    1. Charger config et donnees
    2. Pour chaque modele actif:
       - Hyperparameter tuning sur train avec validation
       - Evaluation sur train/val/test
       - Logging MLflow complet
       - Visualisations (confusion matrix, ROC, PR curves)
    3. Comparaison des modeles
    """
    print("\n" + "="*70)
    print("  PHASE 2: ENTRAINEMENT ET OPTIMISATION DES MODELES")
    print("="*70)

    # 1. Charger la configuration
    print("\n[1/6] Chargement de la configuration...")
    config = load_config()
    mlflow.set_experiment(config['training']['experiment_name'])
    print(f"  [OK] Experience MLflow: {config['training']['experiment_name']}")

    # 2. Charger et preparer les donnees
    print("\n[2/6] Chargement et preprocessing des donnees...")
    raw_df = load_raw_data(config)

    # IMPORTANT: Maintenant preprocess_and_split_data retourne 7 valeurs (avec validation set)
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = preprocess_and_split_data(raw_df, config)

    print(f"  [OK] Donnees pretres et divisees")
    print(f"    - Train: {X_train.shape[0]} samples")
    print(f"    - Val:   {X_val.shape[0]} samples")
    print(f"    - Test:  {X_test.shape[0]} samples")

    # 3. Obtenir les modeles actifs
    print("\n[3/6] Initialisation des modeles...")
    models = get_active_models(config)
    print(f"  [OK] {len(models)} modeles actifs: {list(models.keys())}")

    # 4. Extraire les param grids
    param_grids = {}
    for model_name in models.keys():
        if model_name in config['models']:
            param_grids[model_name] = config['models'][model_name]['param_grid']

    # 5. Parametres de tuning
    n_iter = config['training']['hyperparameter_tuning']['n_iter']
    cv = config['training']['hyperparameter_tuning']['cv']
    scoring = config['training']['hyperparameter_tuning']['scoring']
    random_state = config['preprocessing']['random_state']

    print(f"  [OK] Hyperparameter tuning config:")
    print(f"    - n_iter: {n_iter}")
    print(f"    - cv: {cv}")
    print(f"    - scoring: {scoring}")

    # 6. Entrainement de chaque modele
    print("\n[4/6] Entrainement des modeles...")
    print("="*70)

    best_models = {}
    all_results = []

    for model_name, model in models.items():
        print(f"\n>>> MODELE: {model_name}")
        print("-" * 70)

        with mlflow.start_run(run_name=f"{model_name}_V2_Training"):

            # Tags MLflow
            mlflow.set_tag("model_name", model_name)
            mlflow.set_tag("version", "2.0")
            mlflow.set_tag("features", "advanced_engineering")

            # Creer le pipeline SMOTE + Modele
            pipeline = ImbPipeline(steps=[
                ('smote', SMOTE(random_state=random_state)),
                ('model', model)
            ])

            # Hyperparameter tuning avec RandomizedSearchCV
            print(f"  Lancement RandomizedSearchCV ({n_iter} iterations)...")

            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=param_grids.get(model_name, {}),
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                random_state=random_state,
                n_jobs=-1,
                verbose=1
            )

            search.fit(X_train, y_train)
            best_pipeline = search.best_estimator_

            print(f"  [OK] Meilleur score CV: {search.best_score_:.4f}")
            print(f"  [OK] Meilleurs parametres: {search.best_params_}")

            # Predictions sur train/val/test
            print("\n  Evaluation sur les 3 datasets...")

            # Train
            y_train_pred = best_pipeline.predict(X_train)
            y_train_proba = best_pipeline.predict_proba(X_train)[:, 1]
            train_metrics = evaluate_metrics(y_train, y_train_pred, y_train_proba)

            # Validation
            y_val_pred = best_pipeline.predict(X_val)
            y_val_proba = best_pipeline.predict_proba(X_val)[:, 1]
            val_metrics = evaluate_metrics(y_val, y_val_pred, y_val_proba)

            # Test
            y_test_pred = best_pipeline.predict(X_test)
            y_test_proba = best_pipeline.predict_proba(X_test)[:, 1]
            test_metrics = evaluate_metrics(y_test, y_test_pred, y_test_proba)

            # Afficher la comparaison
            print_metrics_comparison(train_metrics, val_metrics, test_metrics, model_name)

            # Logger les metriques dans MLflow
            for metric_name, value in train_metrics.items():
                mlflow.log_metric(f"train_{metric_name}", value)
            for metric_name, value in val_metrics.items():
                mlflow.log_metric(f"val_{metric_name}", value)
            for metric_name, value in test_metrics.items():
                mlflow.log_metric(f"test_{metric_name}", value)

            # Logger les parametres
            mlflow.log_params(search.best_params_)
            mlflow.log_param("cv_score", search.best_score_)

            # Creer et logger les visualisations
            print("\n  Creation des visualisations...")

            # Confusion matrices (train, val, test)
            cm_train = plot_confusion_matrix(y_train, y_train_pred, model_name, "train")
            cm_val = plot_confusion_matrix(y_val, y_val_pred, model_name, "val")
            cm_test = plot_confusion_matrix(y_test, y_test_pred, model_name, "test")

            mlflow.log_artifact(cm_train, "confusion_matrices")
            mlflow.log_artifact(cm_val, "confusion_matrices")
            mlflow.log_artifact(cm_test, "confusion_matrices")

            # Courbes ROC et Precision-Recall (sur test)
            roc_path = plot_roc_curve(y_test, y_test_proba, model_name)
            pr_path = plot_precision_recall_curve(y_test, y_test_proba, model_name)

            mlflow.log_artifact(roc_path, "curves")
            mlflow.log_artifact(pr_path, "curves")

            # Sauvegarder le modele
            mlflow.sklearn.log_model(
                sk_model=best_pipeline,
                artifact_path=f"model_{model_name}",
                registered_model_name=f"{model_name}_Churn_V2"
            )

            # Sauvegarder localement aussi
            model_path = f"models/{model_name}_best_model.joblib"
            save_object(best_pipeline, model_path)

            print(f"  [OK] Modele sauvegarde: {model_path}")
            print(f"  [OK] Run MLflow termine\n")

            # Stocker pour comparaison finale
            best_models[model_name] = best_pipeline
            all_results.append({
                'model': model_name,
                'val_f1': val_metrics['f1_score'],
                'test_f1': test_metrics['f1_score'],
                'val_roc_auc': val_metrics['roc_auc_score'],
                'test_roc_auc': test_metrics['roc_auc_score']
            })

    # 7. Comparaison finale
    print("\n[5/6] COMPARAISON FINALE DES MODELES")
    print("="*70)

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('test_f1', ascending=False)

    print("\n", results_df.to_string(index=False))

    # Identifier le meilleur modele
    best_model_name = results_df.iloc[0]['model']
    best_test_f1 = results_df.iloc[0]['test_f1']

    print(f"\n{'='*70}")
    print(f"  MEILLEUR MODELE: {best_model_name}")
    print(f"  Test F1-Score: {best_test_f1:.4f}")

    if best_test_f1 >= 0.80:
        print(f"  [SUCCESS] OBJECTIF ATTEINT! (F1 >= 0.80)")
    else:
        print(f"  [INFO] Progression: {best_test_f1:.1%} / 80%")

    print(f"{'='*70}\n")

    # 8. Sauvegarde du resume
    print("[6/6] Sauvegarde du resume...")
    results_df.to_csv('models/model_comparison_results.csv', index=False)
    print("  [OK] Resultats sauvegardes: models/model_comparison_results.csv")

    print("\n" + "="*70)
    print("  PHASE 2 TERMINEE!")
    print("="*70 + "\n")

    return best_models, results_df


if __name__ == "__main__":
    best_models, results = train_and_evaluate()

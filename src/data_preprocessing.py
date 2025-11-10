# src/data_preprocessing.py

"""
Module de prétraitement des données pour la prédiction du churn client.

Ce module gère :
- Chargement et nettoyage des données brutes
- Feature engineering avancé
- Split train/validation/test stratifié
- Création et fit du pipeline de transformation
- Sauvegarde du preprocessor pour production

Auteur: Data Science Team
Version: 2.0 - Avec validation set et feature engineering avancé
Date: 2025
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
import os
from typing import Tuple

# Importer nos modules personnalisés
from src.utils import load_config, save_object, load_object
from src.feature_engineering import ChurnFeatureEngineer


# --- 1. Définition des constantes et des colonnes ---
TARGET_COLUMN = 'Churn'
ID_COLUMN = 'customerID'

# Features de base (avant feature engineering)
BASE_BINARY_COLS = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'SeniorCitizen']
BASE_SERVICE_COLS = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                     'TechSupport', 'StreamingTV', 'StreamingMovies', 'InternetService']
BASE_MULTI_CLASS_COLS = ['Contract', 'PaymentMethod']
BASE_NUMERICAL_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']


def clean_initial_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Effectue le nettoyage initial du DataFrame brut.

    Opérations :
    1. Suppression de customerID (identifiant non-prédictif)
    2. Conversion de TotalCharges en numérique (gestion des espaces vides)
    3. Suppression des lignes avec TotalCharges manquant
    4. Harmonisation des colonnes de services ("No internet service" -> "No")
    5. Mapping de la cible Churn (Yes/No -> 1/0)

    Args:
        df (pd.DataFrame): DataFrame brut chargé depuis le CSV

    Returns:
        pd.DataFrame: DataFrame nettoyé prêt pour feature engineering
    """
    df_cleaned = df.copy()

    # 1. Supprimer customerID
    if ID_COLUMN in df_cleaned.columns:
        df_cleaned.drop(columns=[ID_COLUMN], inplace=True)
        print(f"  [OK] Colonne '{ID_COLUMN}' supprimée")

    # 2. Nettoyer TotalCharges (espaces -> NaN -> numeric)
    if 'TotalCharges' in df_cleaned.columns:
        initial_count = len(df_cleaned)
        df_cleaned['TotalCharges'] = df_cleaned['TotalCharges'].replace(' ', np.nan)
        df_cleaned['TotalCharges'] = pd.to_numeric(df_cleaned['TotalCharges'], errors='coerce')
        df_cleaned.dropna(subset=['TotalCharges'], inplace=True)
        dropped_count = initial_count - len(df_cleaned)
        print(f"  [OK] TotalCharges converti en numérique ({dropped_count} lignes supprimées)")

    # 3. Harmoniser les colonnes de services
    for col in BASE_SERVICE_COLS:
        if col in df_cleaned.columns:
            if col == 'InternetService':
                df_cleaned[col] = df_cleaned[col].replace('No internet service', 'No')
            elif col == 'MultipleLines':
                df_cleaned[col] = df_cleaned[col].replace('No phone service', 'No')
            else:
                df_cleaned[col] = df_cleaned[col].replace('No internet service', 'No')

    print(f"  [OK] Colonnes de services harmonisées")

    # 4. Mapper la cible Churn
    if TARGET_COLUMN in df_cleaned.columns:
        df_cleaned[TARGET_COLUMN] = df_cleaned[TARGET_COLUMN].map({'Yes': 1, 'No': 0})
        print(f"  [OK] Cible '{TARGET_COLUMN}' mappée (Yes->1, No->0)")

    # 5. S'assurer que SeniorCitizen est de type int
    if 'SeniorCitizen' in df_cleaned.columns:
        df_cleaned['SeniorCitizen'] = df_cleaned['SeniorCitizen'].astype(int)

    print(f"  [OK] Nettoyage terminé: {df_cleaned.shape[0]} lignes x {df_cleaned.shape[1]} colonnes")

    return df_cleaned




def create_preprocessing_pipeline(feature_engineer: ChurnFeatureEngineer, df_sample: pd.DataFrame) -> Pipeline:
    """
    Crée le pipeline de prétraitement scikit-learn.

    Le pipeline utilise ColumnTransformer pour appliquer différentes transformations :
    - StandardScaler sur les features numériques
    - OrdinalEncoder sur les features binaires
    - OneHotEncoder sur les features catégorielles multi-classes

    Args:
        feature_engineer (ChurnFeatureEngineer): Instance du feature engineer
        df_sample (pd.DataFrame): DataFrame d'exemple pour identifier les features

    Returns:
        Pipeline: Pipeline scikit-learn configuré
    """
    print("\n  Création du pipeline de prétraitement...")

    # Obtenir les listes de features depuis le feature engineer
    numerical_features, categorical_features, binary_features = feature_engineer.get_feature_names(df_sample)

    print(f"    • Features numériques: {len(numerical_features)}")
    print(f"    • Features catégorielles: {len(categorical_features)}")
    print(f"    • Features binaires: {len(binary_features)}")

    # Séparer les binaires avec encodage différent
    gender_features = [f for f in binary_features if f == 'gender']
    yes_no_binary_features = [f for f in binary_features if f != 'gender' and f in df_sample.columns]

    # Filtrer les features catégorielles pour exclure tenure_group et value_segment (qui seront encodées)
    categorical_to_encode = [f for f in categorical_features
                            if f not in binary_features
                            and f in df_sample.columns
                            and f != 'gender']

    # Mappings pour OrdinalEncoder
    gender_mapping = ['Female', 'Male']
    binary_yes_no_mapping = ['No', 'Yes']

    # Créer les transformateurs
    transformers = []

    # 1. Scaler pour features numériques
    if numerical_features:
        transformers.append(
            ('numerical_scaler', StandardScaler(), numerical_features)
        )

    # 2. Ordinal encoder pour gender
    if gender_features and 'gender' in df_sample.columns:
        transformers.append(
            ('gender_encoder', OrdinalEncoder(categories=[gender_mapping]), ['gender'])
        )

    # 3. Ordinal encoder pour features binaires Yes/No
    yes_no_binary_in_df = [f for f in yes_no_binary_features if f in df_sample.columns]
    if yes_no_binary_in_df:
        yes_no_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
        yes_no_to_encode = [f for f in yes_no_binary_in_df if f in yes_no_cols]

        if yes_no_to_encode:
            transformers.append(
                ('binary_yes_no_encoder',
                 OrdinalEncoder(categories=[binary_yes_no_mapping] * len(yes_no_to_encode)),
                 yes_no_to_encode)
            )

        # Features binaires 0/1 (créées par feature engineering) : passthrough
        binary_0_1_cols = [f for f in yes_no_binary_in_df if f not in yes_no_cols]
        if binary_0_1_cols:
            transformers.append(
                ('binary_passthrough', 'passthrough', binary_0_1_cols)
            )

    # 4. OneHotEncoder pour features catégorielles
    if categorical_to_encode:
        transformers.append(
            ('onehot_encoder',
             OneHotEncoder(handle_unknown='ignore', sparse_output=False),
             categorical_to_encode)
        )

    # Créer le ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop',
        verbose_feature_names_out=False
    )

    # Créer le pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor)
    ])

    print("  [OK] Pipeline créé avec succès")
    return pipeline


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    config: dict
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Divise les données en ensembles train/validation/test de manière stratifiée.

    Split: 70% train, 15% validation, 15% test

    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Cible
        config (dict): Configuration (contient random_state)

    Returns:
        Tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    random_state = config['preprocessing']['random_state']

    # Premier split: 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.30,
        random_state=random_state,
        stratify=y
    )

    # Deuxième split: 50% de 30% = 15% validation, 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        random_state=random_state,
        stratify=y_temp
    )

    print(f"\n  Split des données (stratifié):")
    print(f"    • Train:      {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"    • Validation: {len(X_val):,} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"    • Test:       {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")

    print(f"\n  Distribution du churn:")
    print(f"    • Train:      {y_train.mean()*100:.2f}% churn")
    print(f"    • Validation: {y_val.mean()*100:.2f}% churn")
    print(f"    • Test:       {y_test.mean()*100:.2f}% churn")

    return X_train, X_val, X_test, y_train, y_val, y_test


def preprocess_and_split_data(
    df: pd.DataFrame,
    config: dict
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series, pd.Series, pd.Series, Pipeline]:
    """
    Pipeline complet de prétraitement et séparation des données.

    Étapes:
    1. Nettoyage initial
    2. Feature engineering
    3. Séparation features/cible
    4. Split train/validation/test (70/15/15)
    5. Création et fit du pipeline sur train
    6. Transformation de tous les sets
    7. Sauvegarde du pipeline

    Args:
        df (pd.DataFrame): DataFrame brut
        config (dict): Configuration chargée depuis params.yml

    Returns:
        Tuple: (X_train, X_val, X_test, y_train, y_val, y_test, preprocessor_fitted)
    """
    print("\n" + "="*70)
    print("  PIPELINE DE PRÉTRAITEMENT DES DONNÉES")
    print("="*70)

    # 1. Nettoyage initial
    print("\n[1/7] Nettoyage des données...")
    df_cleaned = clean_initial_data(df.copy())

    # 2. Feature Engineering
    print("\n[2/7] Feature Engineering avancé...")
    feature_engineer = ChurnFeatureEngineer()
    df_enhanced = feature_engineer.create_all_features(df_cleaned)
    print(f"  [OK] {df_enhanced.shape[1] - df_cleaned.shape[1]} nouvelles features créées")
    print(f"  [OK] Total features: {df_enhanced.shape[1] - 1} (sans la cible)")

    # 3. Séparer features et cible
    print("\n[3/7] Séparation features/cible...")
    X = df_enhanced.drop(columns=[TARGET_COLUMN])
    y = df_enhanced[TARGET_COLUMN]
    print(f"  [OK] Features: {X.shape[1]} colonnes")
    print(f"  [OK] Cible: {y.name}")

    # 4. Split train/validation/test
    print("\n[4/7] Split des données (70% train / 15% val / 15% test)...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, config)

    # 5. Créer le pipeline de prétraitement
    print("\n[5/7] Création du pipeline de transformation...")
    preprocessor_pipeline = create_preprocessing_pipeline(feature_engineer, X_train)

    # 6. Fit sur train et transform tous les sets
    print("\n[6/7] Fit et transformation des données...")
    print("  • Fitting sur train set...")
    preprocessor_pipeline.fit(X_train)
    print("  [OK] Pipeline fitted")

    print("  • Transformation de tous les sets...")
    X_train_transformed = preprocessor_pipeline.transform(X_train)
    X_val_transformed = preprocessor_pipeline.transform(X_val)
    X_test_transformed = preprocessor_pipeline.transform(X_test)
    print(f"  [OK] Transformations effectuées")
    print(f"    • Train: {X_train_transformed.shape}")
    print(f"    • Val:   {X_val_transformed.shape}")
    print(f"    • Test:  {X_test_transformed.shape}")

    # 7. Sauvegarder le pipeline
    print("\n[7/7] Sauvegarde du pipeline...")
    preprocessor_path = config['data']['preprocessor_path']
    save_object(preprocessor_pipeline, preprocessor_path)
    print(f"  [OK] Pipeline sauvegardé: {preprocessor_path}")

    print("\n" + "="*70)
    print("  PRÉTRAITEMENT TERMINÉ AVEC SUCCÈS [SUCCESS]")
    print("="*70 + "\n")

    return X_train_transformed, X_val_transformed, X_test_transformed, y_train, y_val, y_test, preprocessor_pipeline


def load_raw_data(config: dict) -> pd.DataFrame:
    """
    Charge les données brutes depuis le chemin spécifié dans la configuration.

    Args:
        config (dict): Dictionnaire de configuration chargé depuis params.yml

    Returns:
        pd.DataFrame: DataFrame des données brutes

    Raises:
        FileNotFoundError: Si le fichier n'est pas trouvé
    """
    file_path = config['data']['raw_data_path']

    try:
        df = pd.read_csv(file_path)
        print(f"\n[OK] Données chargées depuis: {file_path}")
        print(f"  • Shape: {df.shape[0]:,} lignes x {df.shape[1]} colonnes")
        return df
    except FileNotFoundError:
        print(f"[ERROR] ERREUR: Fichier non trouvé: {file_path}")
        raise


# --- Exemple d'utilisation (pour tester le module) ---
if __name__ == "__main__":
    print("\n" + "="*70)
    print("  TEST DU MODULE data_preprocessing.py")
    print("="*70)

    # Charger la configuration
    config = load_config()

    # Charger les données brutes
    raw_df = load_raw_data(config)

    # Prétraiter et diviser les données
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = preprocess_and_split_data(
        raw_df,
        config
    )

    # Vérifier que le pipeline peut être chargé
    print("\n" + "="*70)
    print("  TEST DE CHARGEMENT DU PIPELINE")
    print("="*70)

    preprocessor_path = config['data']['preprocessor_path']
    loaded_preprocessor = load_object(preprocessor_path)
    print(f"[OK] Pipeline chargé avec succès depuis: {preprocessor_path}")

    print("\n" + "="*70)
    print("  TOUS LES TESTS RÉUSSIS [SUCCESS]")
    print("="*70 + "\n")
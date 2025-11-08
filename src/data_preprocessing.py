# src/data_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
import os

# Importer nos nouvelles fonctions utilitaires
from src.utils import load_config, save_object, load_object

# --- 1. Définition des constantes et des colonnes ---
# ... (cette section reste identique) ...
TARGET_COLUMN = 'Churn'
ID_COLUMN = 'customerID'
BINARY_COLS = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
SERVICE_RELATED_COLS = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'InternetService']
MULTI_CLASS_COLS = ['Contract', 'PaymentMethod']
NUMERICAL_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']
FEATURE_COLS = [col for col in BINARY_COLS + SERVICE_RELATED_COLS + MULTI_CLASS_COLS + NUMERICAL_COLS if col not in [ID_COLUMN, TARGET_COLUMN]]


# --- 2. Fonction de nettoyage initial (SRP) ---
# ... (cette fonction reste identique) ...
def clean_initial_data(df: pd.DataFrame) -> pd.DataFrame:
    df_cleaned = df.copy()
    if ID_COLUMN in df_cleaned.columns:
        df_cleaned.drop(columns=[ID_COLUMN], inplace=True)
    df_cleaned['TotalCharges'] = df_cleaned['TotalCharges'].replace(' ', np.nan)
    df_cleaned['TotalCharges'] = pd.to_numeric(df_cleaned['TotalCharges'], errors='coerce')
    df_cleaned.dropna(subset=['TotalCharges'], inplace=True)
    for col in SERVICE_RELATED_COLS:
        if col == 'InternetService':
            df_cleaned[col] = df_cleaned[col].replace('No internet service', 'No')
        elif col == 'MultipleLines':
            df_cleaned[col] = df_cleaned[col].replace('No phone service', 'No')
        else:
            df_cleaned[col] = df_cleaned[col].replace('No internet service', 'No')
    if TARGET_COLUMN in df_cleaned.columns:
        df_cleaned[TARGET_COLUMN] = df_cleaned[TARGET_COLUMN].map({'Yes': 1, 'No': 0})
    return df_cleaned


# --- 3. Création du pipeline de prétraitement (DRY, SRP, OCP, DIP) ---
# ... (cette fonction reste identique) ...
def create_preprocessing_pipeline() -> Pipeline:
    gender_mapping = ['Female', 'Male']
    binary_mapping = ['No', 'Yes']
    binary_cols_yes_no = [col for col in BINARY_COLS if col != 'gender' and col != 'SeniorCitizen']
    binary_cols_gender = ['gender']
    preprocessor = ColumnTransformer(
        transformers=[
            ('numerical_scaler', StandardScaler(), NUMERICAL_COLS),
            ('binary_yes_no_encoder', OrdinalEncoder(categories=[binary_mapping] * len(binary_cols_yes_no)), binary_cols_yes_no),
            ('gender_encoder', OrdinalEncoder(categories=[gender_mapping]), binary_cols_gender),
            ('onehot_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False), MULTI_CLASS_COLS + SERVICE_RELATED_COLS)
        ],
        remainder='passthrough'
    )
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor)
    ])
    return pipeline

# --- 4. Fonction principale de prétraitement et de préparation (SRP) - MISE À JOUR ---
def preprocess_and_split_data(
    df: pd.DataFrame,
    config: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Pipeline]:
    """
    Prétraite le DataFrame, le divise en ensembles d'entraînement/test
    et sauvegarde le pipeline de prétraitement en utilisant la configuration fournie.

    Args:
        df (pd.DataFrame): DataFrame brut à prétraiter.
        config (dict): Dictionnaire de configuration chargé depuis params.yml.

    Returns:
        tuple: (X_train, X_test, y_train, y_test, preprocessor_pipeline_fitted)
    """
    # Lire les paramètres depuis le dictionnaire de configuration
    test_size = config['preprocessing']['test_size']
    random_state = config['preprocessing']['random_state']
    preprocessor_path = config['data']['preprocessor_path']

    # Nettoyage initial
    df_cleaned = clean_initial_data(df.copy())

    # Séparer les caractéristiques (X) de la cible (y)
    X = df_cleaned.drop(columns=[TARGET_COLUMN])
    y = df_cleaned[TARGET_COLUMN]

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Data split into training ({X_train.shape[0]} samples) and test ({X_test.shape[0]} samples) sets.")

    # Créer le pipeline de prétraitement
    preprocessor_pipeline = create_preprocessing_pipeline()

    # Entraîner le pipeline de prétraitement sur les données d'entraînement
    print("Fitting preprocessing pipeline on training data...")
    preprocessor_pipeline.fit(X_train)
    print("Preprocessing pipeline fitted.")

    # Transformer les données
    X_train_processed = preprocessor_pipeline.transform(X_train)
    X_test_processed = preprocessor_pipeline.transform(X_test)
    print("Data transformed successfully.")
    
    # Sauvegarder le pipeline de prétraitement entraîné en utilisant notre fonction utilitaire
    save_object(preprocessor_pipeline, preprocessor_path)

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor_pipeline


# --- 5. Fonction utilitaire pour charger les données brutes (SRP) - MISE À JOUR ---
def load_raw_data(config: dict) -> pd.DataFrame:
    """
    Charge les données brutes depuis le chemin spécifié dans la configuration.

    Args:
        config (dict): Dictionnaire de configuration chargé depuis params.yml.

    Returns:
        pd.DataFrame: DataFrame des données brutes.
    """
    file_path = config['data']['raw_data_path']
    try:
        df = pd.read_csv(file_path)
        print(f"Raw data loaded from {file_path}. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        raise

# --- 6. Exemple d'utilisation (pour tester le module) - MISE À JOUR ---
if __name__ == "__main__":
    print("--- Testing data_preprocessing.py module with config ---")
    
    # Charger la configuration
    config = load_config()

    # Charger les données brutes en utilisant la configuration
    raw_df = load_raw_data(config)

    # Prétraiter et diviser les données en utilisant la configuration
    X_train_transformed, X_test_transformed, y_train, y_test, preprocessor_fitted = preprocess_and_split_data(
        raw_df,
        config
    )

    print("\n--- Preprocessing & Splitting Results ---")
    print(f"X_train_transformed shape: {X_train_transformed.shape}")
    print(f"X_test_transformed shape: {X_test_transformed.shape}")

    # Vérifier que le pipeline peut être chargé et utilisé
    preprocessor_path = config['data']['preprocessor_path']
    loaded_preprocessor = load_object(preprocessor_path)

    # Test de transformation sur une petite partie du df brut avec le pipeline chargé
    sample_X_raw = raw_df.sample(5, random_state=42).drop(columns=[TARGET_COLUMN])
    sample_X_processed_from_loaded = loaded_preprocessor.transform(sample_X_raw)
    
    print(f"\nSample X_raw (input to loaded preprocessor) shape: {sample_X_raw.shape}")
    print(f"Sample X_processed_from_loaded (output) shape: {sample_X_processed_from_loaded.shape}")
    print("Preprocessing module test completed successfully!")
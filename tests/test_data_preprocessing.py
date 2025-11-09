# tests/test_data_preprocessing.py (Version Finale)

import pytest
import pandas as pd
import numpy as np

from src.data_preprocessing import clean_initial_data

@pytest.fixture
def sample_raw_dataframe() -> pd.DataFrame:
    """
    Crée et retourne un DataFrame de test réutilisable avec des cas typiques.
    La 3ème ligne contient maintenant un TotalCharges valide pour ne pas être supprimée.
    """
    data = {
        'customerID': ['123-ABC', '456-DEF', '789-GHI', '101-JKL'],
        'gender': ['Female', 'Male', 'Female', 'Male'],
        'Partner': ['Yes', 'No', 'No', 'Yes'],
        'Dependents': ['No', 'Yes', 'No', 'No'],
        'tenure': [1, 24, 72, 5],
        'PhoneService': ['No', 'Yes', 'Yes', 'Yes'],
        'MultipleLines': ['No phone service', 'No', 'Yes', 'No'],
        'InternetService': ['DSL', 'Fiber optic', 'No', 'DSL'],
        'OnlineSecurity': ['No', 'Yes', 'No internet service', 'No'],
        'OnlineBackup': ['Yes', 'No', 'No internet service', 'No'],
        'DeviceProtection': ['No', 'Yes', 'No internet service', 'No'],
        'TechSupport': ['No', 'No', 'No internet service', 'No'],
        'StreamingTV': ['No', 'Yes', 'No internet service', 'No'],
        'StreamingMovies': ['No', 'Yes', 'No internet service', 'No'],
        'Contract': ['Month-to-month', 'One year', 'Two year', 'Month-to-month'],
        'PaperlessBilling': ['Yes', 'No', 'Yes', 'No'],
        'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Mailed check'],
        'MonthlyCharges': [29.85, 70.70, 20.05, 30.20],
        'TotalCharges': ['29.85', '1889.5', '1400.55', ' '], # La 4ème ligne a maintenant le problème
        'Churn': ['No', 'Yes', 'No', 'Yes']
    }
    return pd.DataFrame(data)

def test_total_charges_conversion_and_na_drop(sample_raw_dataframe):
    input_df = sample_raw_dataframe
    cleaned_df = clean_initial_data(input_df)
    assert cleaned_df.shape[0] == 3
    assert cleaned_df['TotalCharges'].isnull().sum() == 0
    assert cleaned_df['TotalCharges'].dtype == 'float64'

def test_churn_column_mapping(sample_raw_dataframe):
    input_df = sample_raw_dataframe
    cleaned_df = clean_initial_data(input_df)
    expected_churn_values = [0, 1, 0]
    assert cleaned_df['Churn'].dtype == 'int64'
    assert all(cleaned_df['Churn'] == expected_churn_values)

def test_service_columns_harmonization(sample_raw_dataframe):
    input_df = sample_raw_dataframe
    cleaned_df = clean_initial_data(input_df)
    
    # La 4ème ligne est supprimée, les 3 premières restent.
    # La première ligne (iloc[0]) avait 'No phone service'
    assert cleaned_df.iloc[0]['MultipleLines'] == 'No'
    
    # La troisième ligne (iloc[2]) avait 'No internet service'
    assert cleaned_df.iloc[2]['OnlineSecurity'] == 'No'
    assert cleaned_df.iloc[2]['OnlineBackup'] == 'No'
    
    # Vérifier qu'une valeur qui n'était pas 'No...' n'a pas été changée
    assert cleaned_df.iloc[1]['OnlineSecurity'] == 'Yes'

def test_customerid_column_is_dropped(sample_raw_dataframe):
    input_df = sample_raw_dataframe
    cleaned_df = clean_initial_data(input_df)
    assert 'customerID' not in cleaned_df.columns
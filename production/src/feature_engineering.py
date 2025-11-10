# src/feature_engineering.py

"""
Module de Feature Engineering pour la prédiction du churn client.

Ce module contient toutes les transformations et créations de features
basées sur l'analyse exploratoire et les bonnes pratiques du domaine.

Auteur: Data Science Team
Date: 2025
"""

import pandas as pd
import numpy as np
from typing import Tuple


class ChurnFeatureEngineer:
    """
    Classe pour créer des features avancées pour la prédiction de churn.

    Cette classe implémente des transformations basées sur :
    - Domain knowledge (télécommunications)
    - Analyse exploratoire des données
    - Bonnes pratiques ML pour prédiction de churn
    """

    def __init__(self):
        """Initialise le feature engineer."""
        self.service_columns = [
            'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]

    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crée toutes les features engineered.

        Args:
            df (pd.DataFrame): DataFrame avec les features de base nettoyées

        Returns:
            pd.DataFrame: DataFrame avec toutes les features (originales + engineered)
        """
        df_enhanced = df.copy()

        # 1. Features de ratio et moyennes
        df_enhanced = self._create_charge_features(df_enhanced)

        # 2. Features de catégorisation temporelle
        df_enhanced = self._create_tenure_features(df_enhanced)

        # 3. Features de comptage de services
        df_enhanced = self._create_service_features(df_enhanced)

        # 4. Features d'interaction
        df_enhanced = self._create_interaction_features(df_enhanced)

        # 5. Features de segmentation client
        df_enhanced = self._create_customer_segment_features(df_enhanced)

        # 6. Features binaires avancées
        df_enhanced = self._create_advanced_binary_features(df_enhanced)

        return df_enhanced

    def _create_charge_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crée des features liées aux charges et aux montants.

        Features créées:
        - avg_monthly_charge: TotalCharges / tenure (moyenne réelle mensuelle)
        - charges_to_tenure_ratio: Ratio entre monthly et total charges
        - charge_variance: Écart entre MonthlyCharges et avg_monthly_charge
        """
        df = df.copy()

        # Moyenne mensuelle réelle (évite division par 0)
        df['avg_monthly_charge'] = np.where(
            df['tenure'] > 0,
            df['TotalCharges'] / df['tenure'],
            df['MonthlyCharges']
        )

        # Ratio charges (indique si le client paie plus/moins que la moyenne)
        df['charge_variance'] = df['MonthlyCharges'] - df['avg_monthly_charge']

        # Ratio tenure (combien de mois de charges accumulées)
        df['total_to_monthly_ratio'] = np.where(
            df['MonthlyCharges'] > 0,
            df['TotalCharges'] / df['MonthlyCharges'],
            0
        )

        return df

    def _create_tenure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crée des features de catégorisation de la durée client.

        Catégories basées sur l'analyse de churn par tenure:
        - Nouveau (0-12 mois): Taux de churn élevé
        - Régulier (13-24 mois): Taux de churn moyen
        - Fidèle (25-48 mois): Taux de churn faible
        - Ancien (48+ mois): Taux de churn très faible
        """
        df = df.copy()

        # Catégorisation de tenure
        df['tenure_group'] = pd.cut(
            df['tenure'],
            bins=[0, 12, 24, 48, 72],
            labels=['nouveau', 'regulier', 'fidele', 'ancien'],
            include_lowest=True
        )

        # Flags binaires pour segments critiques
        df['is_new_customer'] = (df['tenure'] <= 12).astype(int)
        df['is_very_loyal'] = (df['tenure'] > 48).astype(int)

        # Tenure normalisée (0-1)
        max_tenure = df['tenure'].max() if df['tenure'].max() > 0 else 72
        df['tenure_normalized'] = df['tenure'] / max_tenure

        return df

    def _create_service_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crée des features liées au nombre et type de services.

        Insight: Plus un client a de services, moins il est susceptible de partir
        (effet de lock-in et augmentation du switching cost)
        """
        df = df.copy()

        # Comptage total de services (convertir 'Yes' en 1)
        service_count = 0
        for col in self.service_columns:
            if col in df.columns:
                service_count += (df[col] == 'Yes').astype(int)

        df['service_count'] = service_count

        # Flags pour services spécifiques
        if 'InternetService' in df.columns:
            df['has_internet'] = (df['InternetService'] != 'No').astype(int)
            df['has_fiber_optic'] = (df['InternetService'] == 'Fiber optic').astype(int)

        # Compte des services de sécurité/support (souvent corrélés à la rétention)
        security_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
        security_count = 0
        for col in security_services:
            if col in df.columns:
                security_count += (df[col] == 'Yes').astype(int)

        df['security_services_count'] = security_count
        df['has_security_services'] = (security_count > 0).astype(int)

        # Services de streaming (entertainment)
        if 'StreamingTV' in df.columns and 'StreamingMovies' in df.columns:
            df['streaming_services_count'] = (
                (df['StreamingTV'] == 'Yes').astype(int) +
                (df['StreamingMovies'] == 'Yes').astype(int)
            )
            df['has_streaming'] = (df['streaming_services_count'] > 0).astype(int)

        return df

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crée des features d'interaction entre variables.

        Les interactions capturent des effets combinés importants
        pour la prédiction du churn.
        """
        df = df.copy()

        # Interaction famille (Partner + Dependents)
        if 'Partner' in df.columns and 'Dependents' in df.columns:
            df['has_family'] = (
                ((df['Partner'] == 'Yes') | (df['Dependents'] == 'Yes'))
            ).astype(int)

            df['family_with_dependents'] = (
                ((df['Partner'] == 'Yes') & (df['Dependents'] == 'Yes'))
            ).astype(int)

        # Interaction contrat et paiement automatique
        if 'Contract' in df.columns and 'PaymentMethod' in df.columns:
            df['auto_payment'] = df['PaymentMethod'].isin([
                'Bank transfer (automatic)',
                'Credit card (automatic)'
            ]).astype(int)

            # Long contrat + paiement auto = très faible risque churn
            df['long_contract_auto_pay'] = (
                (df['Contract'] != 'Month-to-month') &
                (df['auto_payment'] == 1)
            ).astype(int)

        # Senior avec famille (segment particulier)
        if 'SeniorCitizen' in df.columns and 'has_family' in df.columns:
            df['senior_with_family'] = (
                (df['SeniorCitizen'] == 1) &
                (df['has_family'] == 1)
            ).astype(int)

        # Interaction charges et tenure (lifetime value)
        if 'MonthlyCharges' in df.columns and 'tenure' in df.columns:
            df['lifetime_value_score'] = df['MonthlyCharges'] * df['tenure']

        return df

    def _create_customer_segment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crée des features de segmentation client.

        Segments basés sur profil de risque de churn.
        """
        df = df.copy()

        # Segment de valeur client (basé sur MonthlyCharges)
        if 'MonthlyCharges' in df.columns:
            df['value_segment'] = pd.cut(
                df['MonthlyCharges'],
                bins=[0, 35, 70, 200],
                labels=['low_value', 'medium_value', 'high_value'],
                include_lowest=True
            )

        # Profil de risque basique (combinaison de facteurs)
        risk_score = 0

        # Facteurs qui augmentent le risque
        if 'Contract' in df.columns:
            risk_score += (df['Contract'] == 'Month-to-month').astype(int) * 3

        if 'PaperlessBilling' in df.columns:
            risk_score += (df['PaperlessBilling'] == 'Yes').astype(int)

        if 'PaymentMethod' in df.columns:
            risk_score += (df['PaymentMethod'] == 'Electronic check').astype(int) * 2

        # Facteurs qui diminuent le risque
        if 'tenure' in df.columns:
            risk_score -= (df['tenure'] > 24).astype(int) * 2

        if 'service_count' in df.columns:
            risk_score -= (df['service_count'] > 3).astype(int)

        df['churn_risk_score'] = risk_score

        return df

    def _create_advanced_binary_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crée des features binaires avancées basées sur patterns de churn.
        """
        df = df.copy()

        # Contrat flexible (month-to-month)
        if 'Contract' in df.columns:
            df['has_flexible_contract'] = (df['Contract'] == 'Month-to-month').astype(int)
            df['has_long_contract'] = (df['Contract'].isin(['One year', 'Two year'])).astype(int)

        # Paperless billing (corrélé positivement au churn dans les données)
        if 'PaperlessBilling' in df.columns:
            df['uses_paperless'] = (df['PaperlessBilling'] == 'Yes').astype(int)

        # Electronic check (méthode la plus associée au churn)
        if 'PaymentMethod' in df.columns:
            df['uses_electronic_check'] = (
                df['PaymentMethod'] == 'Electronic check'
            ).astype(int)

        # Pas de services de protection
        if 'security_services_count' in df.columns:
            df['no_protection_services'] = (df['security_services_count'] == 0).astype(int)

        return df

    def get_feature_names(self, df: pd.DataFrame) -> Tuple[list, list, list]:
        """
        Retourne les listes de noms de features par type.

        Returns:
            Tuple contenant:
            - numerical_features: Features numériques
            - categorical_features: Features catégorielles
            - binary_features: Features binaires
        """
        numerical_features = [
            'tenure', 'MonthlyCharges', 'TotalCharges',
            'avg_monthly_charge', 'charge_variance', 'total_to_monthly_ratio',
            'tenure_normalized', 'service_count', 'security_services_count',
            'streaming_services_count', 'lifetime_value_score', 'churn_risk_score'
        ]

        categorical_features = [
            'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
            'PaperlessBilling', 'PaymentMethod', 'tenure_group', 'value_segment'
        ]

        binary_features = [
            'SeniorCitizen', 'is_new_customer', 'is_very_loyal', 'has_internet',
            'has_fiber_optic', 'has_security_services', 'has_streaming',
            'has_family', 'family_with_dependents', 'auto_payment',
            'long_contract_auto_pay', 'senior_with_family', 'has_flexible_contract',
            'has_long_contract', 'uses_paperless', 'uses_electronic_check',
            'no_protection_services'
        ]

        # Filtrer pour ne garder que les features présentes dans le DataFrame
        numerical_features = [f for f in numerical_features if f in df.columns]
        categorical_features = [f for f in categorical_features if f in df.columns]
        binary_features = [f for f in binary_features if f in df.columns]

        return numerical_features, categorical_features, binary_features


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fonction utilitaire pour créer toutes les features engineered.

    Args:
        df (pd.DataFrame): DataFrame avec features de base nettoyées

    Returns:
        pd.DataFrame: DataFrame avec toutes les features
    """
    engineer = ChurnFeatureEngineer()
    return engineer.create_all_features(df)


# --- Exemple d'utilisation ---
if __name__ == "__main__":
    print("--- Testing feature_engineering.py module ---")

    # Créer un DataFrame de test
    sample_data = {
        'tenure': [1, 24, 48, 72, 6],
        'MonthlyCharges': [29.85, 56.95, 42.30, 106.70, 70.70],
        'TotalCharges': [29.85, 1889.50, 1840.75, 7382.25, 151.65],
        'SeniorCitizen': [0, 0, 0, 1, 0],
        'Partner': ['Yes', 'No', 'No', 'Yes', 'No'],
        'Dependents': ['No', 'No', 'No', 'Yes', 'No'],
        'PhoneService': ['No', 'Yes', 'No', 'Yes', 'Yes'],
        'MultipleLines': ['No', 'No', 'No', 'Yes', 'No'],
        'InternetService': ['DSL', 'DSL', 'DSL', 'Fiber optic', 'Fiber optic'],
        'OnlineSecurity': ['No', 'Yes', 'Yes', 'Yes', 'No'],
        'OnlineBackup': ['Yes', 'No', 'No', 'Yes', 'No'],
        'DeviceProtection': ['No', 'Yes', 'Yes', 'Yes', 'No'],
        'TechSupport': ['No', 'No', 'No', 'Yes', 'No'],
        'StreamingTV': ['No', 'No', 'No', 'Yes', 'No'],
        'StreamingMovies': ['No', 'No', 'No', 'Yes', 'No'],
        'Contract': ['Month-to-month', 'One year', 'One year', 'Two year', 'Month-to-month'],
        'PaperlessBilling': ['Yes', 'No', 'No', 'No', 'Yes'],
        'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)',
                         'Credit card (automatic)', 'Electronic check']
    }

    df_test = pd.DataFrame(sample_data)

    print("\nDataFrame original shape:", df_test.shape)
    print("\nPremières lignes:")
    print(df_test.head(2))

    # Appliquer le feature engineering
    engineer = ChurnFeatureEngineer()
    df_enhanced = engineer.create_all_features(df_test)

    print("\n\nDataFrame avec features engineered shape:", df_enhanced.shape)
    print(f"\nNombre de features ajoutées: {df_enhanced.shape[1] - df_test.shape[1]}")

    # Afficher quelques nouvelles features
    new_features = [
        'avg_monthly_charge', 'tenure_group', 'service_count',
        'has_family', 'churn_risk_score', 'is_new_customer'
    ]

    print("\n\nQuelques nouvelles features créées:")
    print(df_enhanced[new_features].head())

    # Obtenir les listes de features par type
    num_feats, cat_feats, bin_feats = engineer.get_feature_names(df_enhanced)

    print(f"\n\n--- Résumé des Features ---")
    print(f"Features numériques: {len(num_feats)}")
    print(f"Features catégorielles: {len(cat_feats)}")
    print(f"Features binaires: {len(bin_feats)}")
    print(f"Total: {len(num_feats) + len(cat_feats) + len(bin_feats)}")

    print("\nTest du module feature_engineering.py réussi! ✅")

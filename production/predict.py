# predict.py

"""
Script de prediction en production pour le modele de churn.

Usage:
    python predict.py --input data.csv --output predictions.csv
"""

import pandas as pd
import argparse
from src.utils import load_object
from src.data_preprocessing import preprocess_data_for_prediction

def predict_churn(input_file, output_file=None, threshold=0.550):
    """
    Predit le churn pour de nouvelles donnees.

    Args:
        input_file: Chemin vers le fichier CSV d'entree
        output_file: Chemin vers le fichier de sortie (optionnel)
        threshold: Seuil de classification (default: 0.550)

    Returns:
        DataFrame avec predictions
    """
    print("\nChargement du modele...")
    model_package = load_object('models/churn_model_v1.joblib')
    model = model_package['ensemble']
    optimal_threshold = model_package['threshold']

    print(f"Modele charge: F1-Score = {model_package['metrics']['test_f1']:.4f}")
    print(f"Threshold utilise: {optimal_threshold:.3f}")

    print(f"\nChargement des donnees: {input_file}...")
    df = pd.read_csv(input_file)
    print(f"  {len(df)} lignes chargees")

    print("\nPretraitement...")
    preprocessor = load_object('data/preprocessor.joblib')
    X_processed = preprocess_data_for_prediction(df, preprocessor)

    print("\nPrediction...")
    probabilities = model.predict_proba(X_processed)[:, 1]
    predictions = (probabilities >= optimal_threshold).astype(int)

    # Creer DataFrame de resultats
    results = df.copy()
    results['churn_probability'] = probabilities
    results['churn_prediction'] = predictions
    results['churn_risk'] = pd.cut(
        probabilities,
        bins=[0, 0.3, 0.5, 0.7, 1.0],
        labels=['Low', 'Medium', 'High', 'Very High']
    )

    # Statistiques
    n_churners = predictions.sum()
    churn_rate = n_churners / len(predictions) * 100
    avg_prob = probabilities.mean()

    print(f"\nRESULTATS:")
    print(f"  Churners predits: {n_churners}/{len(predictions)} ({churn_rate:.1f}%)")
    print(f"  Probabilite moyenne: {avg_prob:.3f}")
    print(f"  Distribution risque:")
    print(f"    Low Risk:       {(results['churn_risk'] == 'Low').sum()}")
    print(f"    Medium Risk:    {(results['churn_risk'] == 'Medium').sum()}")
    print(f"    High Risk:      {(results['churn_risk'] == 'High').sum()}")
    print(f"    Very High Risk: {(results['churn_risk'] == 'Very High').sum()}")

    # Sauvegarder
    if output_file:
        results.to_csv(output_file, index=False)
        print(f"\nPredictions sauvegardees: {output_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict customer churn')
    parser.add_argument('--input', required=True, help='Input CSV file')
    parser.add_argument('--output', help='Output CSV file (optional)')
    parser.add_argument('--threshold', type=float, default=0.550, help='Classification threshold')

    args = parser.parse_args()

    results = predict_churn(args.input, args.output, args.threshold)

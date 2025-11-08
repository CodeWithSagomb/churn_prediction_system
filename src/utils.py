# src/utils.py

import yaml
import os
import joblib

def load_config(config_path: str = "params.yml") -> dict:
    """
    Charge la configuration depuis un fichier YAML.

    Args:
        config_path (str): Chemin d'accès au fichier de configuration.

    Returns:
        dict: Dictionnaire contenant la configuration.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        print(f"Configuration loaded successfully from {config_path}")
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        raise
    except Exception as e:
        print(f"An error occurred while loading the configuration: {e}")
        raise

def save_object(obj: object, file_path: str):
    """
    Sauvegarde un objet Python (ex: modèle, pipeline) dans un fichier.

    Args:
        obj (object): L'objet à sauvegarder.
        file_path (str): Chemin où sauvegarder l'objet.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(obj, file_path)
        print(f"Object saved successfully to {file_path}")
    except Exception as e:
        print(f"An error occurred while saving the object: {e}")
        raise

def load_object(file_path: str) -> object:
    """
    Charge un objet Python depuis un fichier.

    Args:
        file_path (str): Chemin du fichier à charger.

    Returns:
        object: L'objet chargé.
    """
    try:
        obj = joblib.load(file_path)
        print(f"Object loaded successfully from {file_path}")
        return obj
    except FileNotFoundError:
        print(f"Error: Object file not found at {file_path}")
        raise
    except Exception as e:
        print(f"An error occurred while loading the object: {e}")
        raise

# --- Exemple d'utilisation (pour tester le module) ---
if __name__ == "__main__":
    print("--- Testing utils.py module ---")
    
    # Tester le chargement de la configuration
    config = load_config()
    
    # Accéder à une valeur de configuration
    if config:
        print("\n--- Configuration Sample ---")
        print(f"Raw data path: {config['data']['raw_data_path']}")
        print(f"Random Forest n_estimators sample: {config['models']['RandomForest']['param_grid']['model__n_estimators']}")
        
        # Tester la sauvegarde et le chargement d'un objet simple
        print("\n--- Testing Object Save/Load ---")
        sample_object = {"key": "value", "number": 123}
        save_path = "models/temp_test_object.joblib"
        save_object(sample_object, save_path)
        loaded_object = load_object(save_path)
        print(f"Original object: {sample_object}")
        print(f"Loaded object: {loaded_object}")
        assert sample_object == loaded_object
        print("Object save/load test passed!")
        os.remove(save_path) # Nettoyer le fichier de test
        
    print("\nUtils module test completed successfully!")
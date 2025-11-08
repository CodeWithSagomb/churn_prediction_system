# .gitignore
.venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
/data/*.csv # Pversionner les données transformées, mais utile pour les brutes
/data/*.parquet
/mlruns/ #  tracking local pour MLflow
/models/*.joblib

# .gitignore

# Environnement virtuel Python
.venv/

# Données et artefacts MLflow
mlruns/

# Fichiers cache Python
__pycache__/
*.pyc
*.pyo
*.pyd

# Fichiers cache Jupyter Notebook
.ipynb_checkpoints/

# Fichiers de configuration IDE (VSCode)
.vscode/

# Fichiers temporaires
*.log
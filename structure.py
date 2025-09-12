import os

# Define the folder structure
folders = [
    "lending_club_policy_optimization/config",
    "lending_club_policy_optimization/data/raw",
    "lending_club_policy_optimization/notebooks",
    "lending_club_policy_optimization/src/data",
    "lending_club_policy_optimization/src/models",
    "lending_club_policy_optimization/src/evaluation",
    "lending_club_policy_optimization/src/utils",
    "lending_club_policy_optimization/results/figures",
    "lending_club_policy_optimization/results/models",
    "lending_club_policy_optimization/results/logs",
    "lending_club_policy_optimization/reports",
    "lending_club_policy_optimization/tests"
]

# Define the files to be created
files = [
    "lending_club_policy_optimization/README.md",
    "lending_club_policy_optimization/requirements.txt",
    "lending_club_policy_optimization/config/config.py",
    "lending_club_policy_optimization/data/raw/accepted_2007_to_2018.csv",
    "lending_club_policy_optimization/notebooks/01_exploratory_data_analysis.ipynb",
    "lending_club_policy_optimization/notebooks/02_data_preprocessing.ipynb",
    "lending_club_policy_optimization/notebooks/03_deep_learning_model.ipynb",
    "lending_club_policy_optimization/notebooks/04_offline_rl_agent.ipynb",
    "lending_club_policy_optimization/notebooks/05_analysis_comparison.ipynb",
    "lending_club_policy_optimization/src/__init__.py",
    "lending_club_policy_optimization/src/data/__init__.py",
    "lending_club_policy_optimization/src/data/data_loader.py",
    "lending_club_policy_optimization/src/data/preprocessor.py",
    "lending_club_policy_optimization/src/models/__init__.py",
    "lending_club_policy_optimization/src/models/deep_learning_classifier.py",
    "lending_club_policy_optimization/src/models/offline_rl_agent.py",
    "lending_club_policy_optimization/src/evaluation/__init__.py",
    "lending_club_policy_optimization/src/evaluation/metrics.py",
    "lending_club_policy_optimization/src/evaluation/visualizations.py",
    "lending_club_policy_optimization/src/utils/__init__.py",
    "lending_club_policy_optimization/src/utils/helpers.py",
    "lending_club_policy_optimization/reports/final_report.md",
    "lending_club_policy_optimization/tests/__init__.py",
    "lending_club_policy_optimization/tests/test_models.py"
]

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Create files
for file in files:
    if not os.path.exists(file):
        with open(file, "w") as f:
            pass  # Creates empty file

print("✅ Project structure created successfully!")

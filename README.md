# LendingClub Policy Optimization

A comprehensive solution for **financial decision-making** using the LendingClub loan dataset. This project leverages both **deep learning** and **offline reinforcement learning (RL)** to optimize loan approval policies, aiming to maximize financial returns and minimize risk.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
- [Key Features](#key-features)
- [Outputs](#outputs)
- [Requirements](#requirements)
- [License](#license)
- [Contact](#contact)

---

## Overview

This repository implements two main approaches for loan approval decisions:

1. **Deep Learning Classifier**: Predicts the probability of loan default using historical data.
2. **Offline RL Agent**: Learns an optimal approval policy directly from past decisions and outcomes using Conservative Q-Learning (CQL).

Both approaches are evaluated and compared to provide actionable insights for financial institutions.

---

## Project Structure

```
├── data/
│   ├── raw/         # Original LendingClub dataset
│   └── processed/   # Cleaned and feature-engineered data
├── src/
│   ├── models/      # Deep learning and RL agent implementations
│   ├── evaluation/  # Metrics and visualization tools
│   └── utils/       # Helper functions and logging
├── results/         # Model outputs, figures, and analysis
├── config/          # Configuration files for models and rewards
├── requirements.txt # Python dependencies
├── README.md        # Project documentation
└── notebooks/
    ├── 01_exploratory_data_analysiss.ipynb
    ├── 03_deep_learning_model.ipynb
    └── 04_offline_rl_agent.ipynb
```

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   ```

2. **Create and activate a Python environment:**
   ```bash
   conda create -p venv python=3.9 -y
   conda activate venv/
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Data Preparation

1. **Download the LendingClub dataset:**
   - Get `accepted_2007_to_2018.csv` from [Kaggle LendingClub Dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club).
   - Place it in the `data/raw/` directory.

2. **Run the EDA notebook:**
   - Open and execute `01_exploratory_data_analysiss.ipynb` to clean and preprocess the data.
   - Processed data will be saved in `data/processed/`.

---

## Usage

### 1. Exploratory Data Analysis
- Run `01_exploratory_data_analysiss.ipynb` to explore and prepare the data.

### 2. Deep Learning Model
- Run `03_deep_learning_model.ipynb` to train and evaluate a classifier for loan default prediction.

### 3. Offline RL Agent
- Run `04_offline_rl_agent.ipynb` to train an RL agent for optimal loan approval decisions using CQL.

### 4. Evaluation & Visualization
- Use the provided scripts and notebooks to compare model performance, visualize results, and analyze financial impact.

---

## Key Features

- **Supervised Learning**: Neural network classifier for default risk prediction.
- **Offline RL**: Conservative Q-Learning agent for policy optimization.
- **Custom Evaluation**: Metrics, confusion matrices, and financial analysis.
- **Visualization**: Distribution plots, policy comparison charts, and feature importance.
- **Modular Design**: Easily extendable codebase with clear separation of concerns.

---

## Outputs

- **Processed Data**: Cleaned datasets for modeling.
- **Model Results**: Performance metrics, trained models, and predictions.
- **Visualizations**: Plots and figures for analysis.
- **Policy Decisions**: CSV/JSON files with recommended actions and financial outcomes.

---

## Requirements

- Python 3.9
- See `requirements.txt` for all dependencies.

---

## License

This project is for research and educational purposes.  
See the repository for license details.

---

## Contact

For questions, issues, or collaboration, please open an issue or contact the maintainers.

---

**Summary:**  
This repository enables financial analysts and data scientists to compare deep learning and RL-based policy optimization for loan approval, with a focus on maximizing financial outcomes and understanding model decisions.

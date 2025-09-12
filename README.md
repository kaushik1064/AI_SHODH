# Policy Optimization for Financial Decision-Making

This project implements both supervised learning and offline reinforcement learning approaches for loan approval decisions using the LendingClub dataset.

## Project Overview

The goal is to develop an intelligent system that can decide whether to approve or deny loan applications to maximize financial return. We compare two approaches:

1. **Deep Learning Classifier**: Predicts probability of default
2. **Offline RL Agent**: Learns optimal approval policy directly

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd lending_club_policy_optimization
```

2. Create virtual environment:

```bash
conda create -p venv python==3.9 -y
conda activate venv/
```


3. Install dependencies:

```bash
pip install -r requirements.txt
```


4. Download the dataset:
   - Download `accepted_2007_to_2018.csv` from [Kaggle LendingClub Dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
   - Place it in `data/raw/` directory

## Usage

### 1. Exploratory Data Analysis

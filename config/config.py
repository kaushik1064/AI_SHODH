"""
Configuration file for the lending club policy optimization project.
"""
import os
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class DataConfig:
    """Data configuration parameters."""
    raw_data_path: str = "D:/AI_SHODH/data/raw/accepted_2007_to_2018.csv"
    processed_data_path: str = "data/processed/"
    sample_size: int = 100000  # For faster iteration
    test_size: float = 0.2
    val_size: float = 0.2
    random_state: int = 42

@dataclass
class ModelConfig:
    """Model configuration parameters."""
    # Deep Learning Model
    dl_hidden_layers: List[int] = None
    dl_dropout_rate: float = 0.3
    dl_learning_rate: float = 0.001
    dl_batch_size: int = 256
    dl_epochs: int = 100
    dl_early_stopping_patience: int = 10
    
    # Offline RL Model
    rl_algorithm: str = "CQL"  # Conservative Q-Learning
    rl_batch_size: int = 256
    rl_n_epochs: int = 100
    rl_learning_rate: float = 3e-4
    
    def __post_init__(self):
        if self.dl_hidden_layers is None:
            self.dl_hidden_layers = [256, 128, 64, 32]

@dataclass
class RewardConfig:
    """Reward structure for RL agent."""
    deny_reward: float = 0.0
    approve_paid_multiplier: float = 1.0  # loan_amnt * int_rate
    approve_default_multiplier: float = -1.0  # -loan_amnt

# Global configuration instances
DATA_CONFIG = DataConfig()
MODEL_CONFIG = ModelConfig()
REWARD_CONFIG = RewardConfig()

# Feature selection
NUMERICAL_FEATURES = [
    'loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'term', 'int_rate',
    'installment', 'annual_inc', 'dti', 'delinq_2yrs', 'fico_range_low',
    'fico_range_high', 'inq_last_6mths', 'mths_since_last_delinq',
    'mths_since_last_record', 'open_acc', 'pub_rec', 'revol_bal',
    'revol_util', 'total_acc', 'collections_12_mths_ex_med',
    'mths_since_last_major_derog', 'acc_now_delinq', 'tot_coll_amt',
    'tot_cur_bal', 'total_rev_hi_lim'
]

CATEGORICAL_FEATURES = [
    'grade', 'sub_grade', 'emp_title', 'emp_length', 'home_ownership',
    'verification_status', 'purpose', 'addr_state', 'initial_list_status',
    'application_type'
]

TARGET_MAPPING = {
    'Fully Paid': 0,
    'Charged Off': 1,
    'Default': 1,
    'Late (31-120 days)': 1,
    'Late (16-30 days)': 1
}

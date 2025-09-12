"""
Data loading and initial processing utilities.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging
from config.config import DATA_CONFIG, TARGET_MAPPING

logger = logging.getLogger(__name__)

class LendingClubDataLoader:
    """Data loader for LendingClub dataset."""
    
    def __init__(self, config: Optional[object] = None):
        self.config = config or DATA_CONFIG
        
    def load_raw_data(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        """Load raw LendingClub data."""
        try:
            logger.info(f"Loading data from {self.config.raw_data_path}")
            
            # Load data in chunks for memory efficiency
            chunk_size = 10000
            chunks = []
            
            for chunk in pd.read_csv(self.config.raw_data_path, 
                                   chunksize=chunk_size, 
                                   low_memory=False):
                chunks.append(chunk)
                if sample_size and len(pd.concat(chunks)) >= sample_size:
                    break
            
            df = pd.concat(chunks, ignore_index=True)
            
            if sample_size:
                df = df.sample(n=min(sample_size, len(df)), 
                             random_state=self.config.random_state)
            
            logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def create_binary_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create binary target variable from loan_status."""
        df = df.copy()
        
        # Filter for completed loans only
        completed_statuses = list(TARGET_MAPPING.keys())
        df = df[df['loan_status'].isin(completed_statuses)]
        
        # Create binary target
        df['target'] = df['loan_status'].map(TARGET_MAPPING)
        
        # Remove rows with unknown status
        df = df.dropna(subset=['target'])
        df['target'] = df['target'].astype(int)
        
        logger.info(f"Target distribution:\n{df['target'].value_counts()}")
        return df
    
    def get_feature_info(self, df: pd.DataFrame) -> dict:
        """Get information about features in the dataset."""
        info = {
            'total_features': len(df.columns),
            'numerical_features': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_features': df.select_dtypes(include=['object']).columns.tolist(),
            'missing_values': df.isnull().sum().sort_values(ascending=False),
            'target_distribution': df['target'].value_counts() if 'target' in df.columns else None
        }
        return info

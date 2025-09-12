"""
Helper utilities and functions.
"""
import os
import json
import pickle
import logging
from typing import Any, Dict, List
import pandas as pd
import numpy as np
from datetime import datetime

def setup_logging(log_level: str = 'INFO', log_file: str = None) -> None:
    """Setup logging configuration."""
    
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if log_file:
        logging.basicConfig(
            level=getattr(logging, log_level),
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(
            level=getattr(logging, log_level),
            format=log_format
        )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging setup complete. Level: {log_level}")

def create_directories() -> None:
    """Create necessary project directories."""
    
    directories = [
        'data/raw',
        'data/processed',
        'results/figures',
        'results/models',
        'results/logs',
        'reports'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def save_results(results: Dict[str, Any], filename: str, 
                results_dir: str = 'results') -> None:
    """Save results to file."""
    
    filepath = os.path.join(results_dir, filename)
    
    if filename.endswith('.json'):
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = convert_numpy_to_list(results)
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
    
    elif filename.endswith('.pkl'):
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
    
    elif filename.endswith('.csv'):
        if isinstance(results, pd.DataFrame):
            results.to_csv(filepath, index=False)
        else:
            pd.DataFrame([results]).to_csv(filepath, index=False)
    
    print(f"Results saved to: {filepath}")

def load_results(filename: str, results_dir: str = 'results') -> Any:
    """Load results from file."""
    
    filepath = os.path.join(results_dir, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if filename.endswith('.json'):
        with open(filepath, 'r') as f:
            return json.load(f)
    
    elif filename.endswith('.pkl'):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    elif filename.endswith('.csv'):
        return pd.read_csv(filepath)
    
    else:
        raise ValueError(f"Unsupported file format: {filename}")

def convert_numpy_to_list(obj: Any) -> Any:
    """Convert numpy arrays to lists recursively."""
    
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_list(item) for item in obj]
    elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
        return obj.item()
    else:
        return obj

def calculate_memory_usage(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate memory usage of DataFrame."""
    
    memory_usage = df.memory_usage(deep=True)
    total_mb = memory_usage.sum() / (1024 * 1024)
    
    return {
        'total_mb': total_mb,
        'per_column_mb': (memory_usage / (1024 * 1024)).to_dict(),
        'shape': df.shape
    }

def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame data types to reduce memory usage."""
    
    df_optimized = df.copy()
    
    for col in df_optimized.columns:
        col_type = df_optimized[col].dtype
        
        if col_type != 'object':
            c_min = df_optimized[col].min()
            c_max = df_optimized[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df_optimized[col] = df_optimized[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df_optimized[col] = df_optimized[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df_optimized[col] = df_optimized[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df_optimized[col] = df_optimized[col].astype(np.float32)
    
    return df_optimized

def print_data_summary(df: pd.DataFrame, title: str = "Data Summary") -> None:
    """Print comprehensive data summary."""
    
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
    
    print(f"\nData types:")
    print(df.dtypes.value_counts())
    
    print(f"\nMissing values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing': missing,
        'Percentage': missing_pct
    })
    print(missing_df[missing_df['Missing'] > 0].sort_values('Missing', ascending=False))
    
    print(f"\nNumerical columns summary:")
    print(df.describe())

def generate_timestamp() -> str:
    """Generate timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def get_model_filename(model_name: str, timestamp: str = None) -> str:
    """Generate standardized model filename."""
    if timestamp is None:
        timestamp = generate_timestamp()
    return f"{model_name}_{timestamp}.pkl"

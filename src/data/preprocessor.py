"""
Data preprocessing utilities.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Tuple, List, Dict, Any
import logging
from config.config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, DATA_CONFIG

logger = logging.getLogger(__name__)

class LendingClubPreprocessor:
    """Preprocessor for LendingClub dataset."""
    
    def __init__(self):
        self.preprocessor = None
        self.feature_names = None
        self.fitted = False
        
    def create_preprocessing_pipeline(self) -> ColumnTransformer:
        """Create preprocessing pipeline."""
        # Numerical pipeline
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical pipeline
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])
        
        # Combine pipelines
        preprocessor = ColumnTransformer([
            ('num', numerical_pipeline, NUMERICAL_FEATURES),
            ('cat', categorical_pipeline, CATEGORICAL_FEATURES)
        ])
        
        return preprocessor
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Fit preprocessor and transform data."""
        # Select features that exist in the dataset
        available_num_features = [f for f in NUMERICAL_FEATURES if f in df.columns]
        available_cat_features = [f for f in CATEGORICAL_FEATURES if f in df.columns]
        
        logger.info(f"Using {len(available_num_features)} numerical and {len(available_cat_features)} categorical features")
        
        # Update feature lists
        self.numerical_features = available_num_features
        self.categorical_features = available_cat_features
        
        # Create preprocessor with available features
        self.preprocessor = self._create_pipeline_with_features(
            available_num_features, available_cat_features
        )
        
        # Prepare features and target
        X = df[available_num_features + available_cat_features]
        y = df['target'].values
        
        # Fit and transform
        X_processed = self.preprocessor.fit_transform(X)
        self.fitted = True
        
        # Store feature names
        self._get_feature_names()
        
        logger.info(f"Processed data shape: {X_processed.shape}")
        return X_processed, y
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted preprocessor."""
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        X = df[self.numerical_features + self.categorical_features]
        return self.preprocessor.transform(X)
    
    def _create_pipeline_with_features(self, num_features: List[str], 
                                     cat_features: List[str]) -> ColumnTransformer:
        """Create pipeline with specific features."""
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])
        
        transformers = []
        if num_features:
            transformers.append(('num', numerical_pipeline, num_features))
        if cat_features:
            transformers.append(('cat', categorical_pipeline, cat_features))
        
        return ColumnTransformer(transformers)
    
    def _get_feature_names(self):
        """Get feature names after preprocessing."""
        feature_names = []
        
        # Numerical features
        if hasattr(self.preprocessor.named_transformers_, 'num'):
            feature_names.extend(self.numerical_features)
        
        # Categorical features
        if hasattr(self.preprocessor.named_transformers_, 'cat'):
            cat_encoder = self.preprocessor.named_transformers_['cat']['encoder']
            cat_feature_names = cat_encoder.get_feature_names_out(self.categorical_features)
            feature_names.extend(cat_feature_names)
        
        self.feature_names = feature_names
    
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Split data into train, validation, and test sets."""
        # First split: train+val, test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=DATA_CONFIG.test_size, 
            random_state=DATA_CONFIG.random_state, stratify=y
        )
        
        # Second split: train, val
        val_size_adjusted = DATA_CONFIG.val_size / (1 - DATA_CONFIG.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted,
            random_state=DATA_CONFIG.random_state, stratify=y_temp
        )
        
        logger.info(f"Data split - Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test

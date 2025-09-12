"""
Unit tests for model implementations.
"""
import unittest
import numpy as np
import pandas as pd
import tempfile
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.data.data_loader import LendingClubDataLoader
from src.data.preprocessor import LendingClubPreprocessor
from src.models.deep_learning_classifier import DeepLearningClassifier, LoanClassifierTrainer
from src.models.offline_rl_agent import OfflineRLAgent, LoanApprovalEnvironment
from src.evaluation.metrics import ModelEvaluator

class TestDataLoader(unittest.TestCase):
    """Test data loading functionality."""
    
    def setUp(self):
        self.loader = LendingClubDataLoader()
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'loan_amnt': [1000, 2000, 3000],
            'int_rate': [10.5, 15.2, 8.9],
            'loan_status': ['Fully Paid', 'Charged Off', 'Fully Paid'],
            'annual_inc': [50000, 60000, 70000]
        })
    
    def test_create_binary_target(self):
        """Test binary target creation."""
        result = self.loader.create_binary_target(self.sample_data)
        
        self.assertIn('target', result.columns)
        self.assertEqual(result['target'].iloc[0], 0)  # Fully Paid -> 0
        self.assertEqual(result['target'].iloc[1], 1)  # Charged Off -> 1
        self.assertEqual(result['target'].dtype, int)
    
    def test_get_feature_info(self):
        """Test feature information extraction."""
        info = self.loader.get_feature_info(self.sample_data)
        
        self.assertIn('total_features', info)
        self.assertIn('numerical_features', info)
        self.assertIn('categorical_features', info)
        self.assertIn('missing_values', info)

class TestPreprocessor(unittest.TestCase):
    """Test preprocessing functionality."""
    
    def setUp(self):
        self.preprocessor = LendingClubPreprocessor()
        
        # Create sample data with target
        self.sample_data = pd.DataFrame({
            'loan_amnt': [1000, 2000, 3000, 4000],
            'int_rate': [10.5, 15.2, 8.9, 12.1],
            'annual_inc': [50000, 60000, 70000, 80000],
            'grade': ['A', 'B', 'A', 'C'],
            'target': [0, 1, 0, 1]
        })
    
    def test_preprocessing_pipeline(self):
        """Test preprocessing pipeline creation."""
        pipeline = self.preprocessor.create_preprocessing_pipeline()
        self.assertIsNotNone(pipeline)
    
    def test_fit_transform(self):
        """Test fit and transform functionality."""
        X, y = self.preprocessor.fit_transform(self.sample_data)
        
        self.assertEqual(X.shape[0], len(self.sample_data))
        self.assertEqual(len(y), len(self.sample_data))
        self.assertTrue(self.preprocessor.fitted)

class TestDeepLearningClassifier(unittest.TestCase):
    """Test deep learning classifier."""
    
    def setUp(self):
        self.input_dim = 10
        self.model = DeepLearningClassifier(self.input_dim, [32, 16])
        
        # Sample data
        self.X_train = np.random.randn(100, self.input_dim)
        self.y_train = np.random.randint(0, 2, 100)
        self.X_test = np.random.randn(20, self.input_dim)
        self.y_test = np.random.randint(0, 2, 20)
    
    def test_model_creation(self):
        """Test model creation."""
        self.assertIsNotNone(self.model)
        self.assertEqual(self.model.network[0].in_features, self.input_dim)
    
    def test_forward_pass(self):
        """Test forward pass."""
        x = np.random.randn(5, self.input_dim)
        import torch
        x_tensor = torch.FloatTensor(x)
        output = self.model(x_tensor)
        
        self.assertEqual(output.shape[0], 5)
        self.assertEqual(output.shape[1], 1)

class TestOfflineRLAgent(unittest.TestCase):
    """Test offline RL agent."""
    
    def setUp(self):
        self.state_dim = 10
        self.agent = OfflineRLAgent(self.state_dim)
        
        # Sample data
        self.X = np.random.randn(50, self.state_dim)
        self.df = pd.DataFrame({
            'loan_amnt': np.random.randint(1000, 10000, 50),
            'int_rate': np.random.uniform(5, 20, 50),
            'target': np.random.randint(0, 2, 50)
        })
    
    def test_agent_creation(self):
        """Test agent creation."""
        self.assertIsNotNone(self.agent)
        self.assertEqual(self.agent.state_dim, self.state_dim)
        self.assertEqual(self.agent.action_dim, 2)
    
    def test_mdp_dataset_creation(self):
        """Test MDP dataset creation."""
        dataset = self.agent.create_mdp_dataset(self.X, self.df)
        self.assertIsNotNone(dataset)

class TestLoanApprovalEnvironment(unittest.TestCase):
    """Test loan approval environment."""
    
    def setUp(self):
        self.env = LoanApprovalEnvironment()
    
    def test_reward_calculation(self):
        """Test reward calculation."""
        # Test deny action
        reward_deny = self.env.calculate_reward(0, 1000, 10, 0)
        self.assertEqual(reward_deny, 0)
        
        # Test approve action with payment
        reward_approve_paid = self.env.calculate_reward(1, 1000, 10, 0)
        self.assertEqual(reward_approve_paid, 100)  # 1000 * 0.1
        
        # Test approve action with default
        reward_approve_default = self.env.calculate_reward(1, 1000, 10, 1)
        self.assertEqual(reward_approve_default, -1000)

class TestModelEvaluator(unittest.TestCase):
    """Test model evaluation utilities."""
    
    def setUp(self):
        self.evaluator = ModelEvaluator()
        
        # Sample data
        self.y_true = np.array([0, 1, 1, 0, 1])
        self.y_pred_proba = np.array([0.2, 0.8, 0.7, 0.3, 0.9])
        self.y_pred_binary = np.array([0, 1, 1, 0, 1])
    
    def test_classifier_evaluation(self):
        """Test classifier evaluation."""
        results = self.evaluator.evaluate_classifier(
            self.y_true, self.y_pred_proba, self.y_pred_binary
        )
        
        self.assertIn('auc', results)
        self.assertIn('f1_score', results)
        self.assertIn('precision', results)
        self.assertIn('recall', results)
        
        self.assertTrue(0 <= results['auc'] <= 1)
        self.assertTrue(0 <= results['f1_score'] <= 1)
    
    def test_rl_policy_evaluation(self):
        """Test RL policy evaluation."""
        actions = np.array([0, 1, 1, 0, 1])
        rewards = np.array([0, 100, -1000, 0, 200])
        
        results = self.evaluator.evaluate_rl_policy(actions, rewards)
        
        self.assertIn('average_reward', results)
        self.assertIn('approval_rate', results)
        self.assertIn('total_reward', results)

if __name__ == '__main__':
    unittest.main()

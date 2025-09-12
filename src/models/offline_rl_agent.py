"""
Offline reinforcement learning agent for loan approval decisions.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
import logging

# Handle different d3rlpy versions gracefully
try:
    # Try newer d3rlpy version (2.x)
    from d3rlpy.algos import CQLConfig, CQL
    from d3rlpy.dataset import create_cartpole_dataset, MDPDataset
    from d3rlpy.logging import configure_logger
    D3RLPY_VERSION = "2.x"
    HAVE_D3RLPY = True
except ImportError:
    try:
        # Try older d3rlpy version (1.x)
        from d3rlpy.algos import CQL
        from d3rlpy.dataset import MDPDataset
        from d3rlpy.metrics import TDErrorEvaluator
        D3RLPY_VERSION = "1.x"
        HAVE_D3RLPY = True
    except ImportError:
        print("Warning: d3rlpy not available, using fallback RL implementation")
        HAVE_D3RLPY = False
        D3RLPY_VERSION = None

from config.config import MODEL_CONFIG, REWARD_CONFIG

logger = logging.getLogger(__name__)

class LoanApprovalEnvironment:
    """Environment for loan approval decisions."""
    
    def __init__(self, reward_config=None):
        self.reward_config = reward_config or REWARD_CONFIG
    
    def calculate_reward(self, action: int, loan_amnt: float, 
                        int_rate: float, loan_status: int) -> float:
        """Calculate reward based on action and outcome."""
        if action == 0:  # Deny loan
            return self.reward_config.deny_reward
        else:  # Approve loan
            if loan_status == 0:  # Fully paid
                return loan_amnt * (int_rate / 100) * self.reward_config.approve_paid_multiplier
            else:  # Default
                return loan_amnt * self.reward_config.approve_default_multiplier

class FallbackRLAgent:
    """Simple fallback RL agent when d3rlpy is not available."""
    
    def __init__(self, state_dim: int, action_dim: int = 2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.environment = LoanApprovalEnvironment()
        self.fitted = False
        self.policy_threshold = 0.0
        self.feature_weights = None
        
    def create_mdp_dataset(self, X: np.ndarray, df: pd.DataFrame) -> Dict:
        """Create simple dataset representation."""
        observations = []
        actions = []
        rewards = []
        
        for i, (_, row) in enumerate(df.iterrows()):
            state = X[i]
            
            # Create both actions for each state
            for action in [0, 1]:
                observations.append(state)
                actions.append(action)
                
                reward = self.environment.calculate_reward(
                    action=action,
                    loan_amnt=row['loan_amnt'],
                    int_rate=row['int_rate'],
                    loan_status=row['target']
                )
                rewards.append(reward)
        
        return {
            'observations': np.array(observations),
            'actions': np.array(actions),
            'rewards': np.array(rewards)
        }
    
    def train(self, X_train: np.ndarray, df_train: pd.DataFrame) -> Dict:
        """Train simple reward-based policy."""
        dataset = self.create_mdp_dataset(X_train, df_train)
        
        # Analyze rewards for different actions
        rewards_deny = dataset['rewards'][dataset['actions'] == 0]
        rewards_approve = dataset['rewards'][dataset['actions'] == 1]
        
        avg_reward_deny = np.mean(rewards_deny)
        avg_reward_approve = np.mean(rewards_approve)
        
        logger.info(f"Training Fallback RL Agent:")
        logger.info(f"  Average reward (Deny): ${avg_reward_deny:.2f}")
        logger.info(f"  Average reward (Approve): ${avg_reward_approve:.2f}")
        
        # Learn simple policy: use loan characteristics to make decisions
        # Create feature weights based on correlation with positive rewards
        positive_rewards_mask = dataset['rewards'] > 0
        positive_observations = dataset['observations'][positive_rewards_mask]
        
        if len(positive_observations) > 0:
            # Simple heuristic: features that correlate with positive outcomes
            self.feature_weights = np.mean(positive_observations, axis=0)
            self.policy_threshold = np.percentile(
                positive_observations @ self.feature_weights, 50
            )  # Conservative threshold
        else:
            # Very conservative: mostly deny
            self.feature_weights = np.random.normal(0, 0.01, self.state_dim)
            self.policy_threshold = 10.0  # High threshold = mostly deny
        
        self.fitted = True
        
        logger.info(f"  Policy threshold: {self.policy_threshold:.4f}")
        logger.info(f"  Training completed with fallback agent")
        
        return {'training_completed': True, 'fallback_used': True}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict actions using learned policy."""
        if not self.fitted:
            raise ValueError("Agent must be trained before making predictions")
        
        # Simple linear policy
        scores = X @ self.feature_weights
        actions = (scores > self.policy_threshold).astype(int)
        
        return actions

class OfflineRLAgent:
    """Offline RL agent with version compatibility."""
    
    def __init__(self, state_dim: int, action_dim: int = 2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.environment = LoanApprovalEnvironment()
        self.fitted = False
        
        if HAVE_D3RLPY:
            self._init_d3rlpy_agent()
            logger.info(f"Using d3rlpy version {D3RLPY_VERSION}")
        else:
            self.fallback_agent = FallbackRLAgent(state_dim, action_dim)
            logger.warning("Using fallback RL agent")
        
        self.agent_type = "d3rlpy" if HAVE_D3RLPY else "fallback"
    
    def _init_d3rlpy_agent(self):
        """Initialize d3rlpy agent based on version."""
        if D3RLPY_VERSION == "2.x":
            self.cql = CQL(
                config=CQLConfig(
                    batch_size=MODEL_CONFIG.rl_batch_size,
                    learning_rate=MODEL_CONFIG.rl_learning_rate,
                )
            )
        elif D3RLPY_VERSION == "1.x":
            self.cql = CQL(
                batch_size=MODEL_CONFIG.rl_batch_size,
                learning_rate=MODEL_CONFIG.rl_learning_rate,
            )
    
    def create_mdp_dataset(self, X: np.ndarray, df: pd.DataFrame):
        """Create MDP dataset compatible with d3rlpy version."""
        if self.agent_type == "fallback":
            return self.fallback_agent.create_mdp_dataset(X, df)
        
        observations = []
        actions = []
        rewards = []
        terminals = []
        next_observations = []
        
        for i, (_, row) in enumerate(df.iterrows()):
            state = X[i]
            
            # For each loan, create approve and deny actions
            for action in [0, 1]:
                observations.append(state)
                actions.append(action)
                
                reward = self.environment.calculate_reward(
                    action=action,
                    loan_amnt=row['loan_amnt'],
                    int_rate=row['int_rate'],
                    loan_status=row['target']
                )
                rewards.append(reward)
                terminals.append(True)  # Each loan is terminal
                next_observations.append(state)  # Same state (terminal)
        
        # Convert to numpy arrays
        observations = np.array(observations, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)
        rewards = np.array(rewards, dtype=np.float32)
        terminals = np.array(terminals, dtype=bool)
        next_observations = np.array(next_observations, dtype=np.float32)
        
        logger.info(f"Created MDP dataset with {len(observations)} transitions")
        logger.info(f"Reward statistics - Mean: {np.mean(rewards):.2f}, "
                   f"Std: {np.std(rewards):.2f}")
        
        if D3RLPY_VERSION == "2.x":
            # Create dataset for d3rlpy 2.x
            dataset = MDPDataset(
                observations=observations,
                actions=actions,
                rewards=rewards,
                terminals=terminals,
                next_observations=next_observations
            )
        else:
            # Create dataset for d3rlpy 1.x
            dataset = MDPDataset(
                observations=observations,
                actions=actions,
                rewards=rewards,
                terminals=terminals
            )
        
        return dataset
    
    def train(self, X_train: np.ndarray, df_train: pd.DataFrame) -> Dict:
        """Train the offline RL agent."""
        if self.agent_type == "fallback":
            result = self.fallback_agent.train(X_train, df_train)
            self.fitted = True
            return result
        
        try:
            dataset = self.create_mdp_dataset(X_train, df_train)
            
            logger.info("Training d3rlpy CQL agent...")
            
            if D3RLPY_VERSION == "2.x":
                self.cql.fit(
                    dataset,
                    n_steps=MODEL_CONFIG.rl_n_epochs * 100,  # Reduced for faster training
                )
            else:
                # d3rlpy 1.x
                self.cql.fit(
                    dataset,
                    n_steps=MODEL_CONFIG.rl_n_epochs * 100,
                    evaluators={'td_error': TDErrorEvaluator()}
                )
            
            self.fitted = True
            
            # Try to save model
            try:
                import os
                os.makedirs('results/models', exist_ok=True)
                self.cql.save('results/models/offline_rl_agent')
                logger.info("Model saved successfully")
            except Exception as e:
                logger.warning(f"Could not save model: {e}")
            
            return {'training_completed': True}
            
        except Exception as e:
            logger.error(f"d3rlpy training failed: {e}")
            logger.info("Falling back to simple RL agent")
            
            # Fall back to simple agent
            self.fallback_agent = FallbackRLAgent(self.state_dim, self.action_dim)
            self.agent_type = "fallback"
            result = self.fallback_agent.train(X_train, df_train)
            self.fitted = True
            return result
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict actions for given states."""
        if not self.fitted:
            raise ValueError("Agent must be trained before making predictions")
        
        if self.agent_type == "fallback":
            return self.fallback_agent.predict(X)
        
        try:
            actions = self.cql.predict(X)
            return actions
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            # Emergency fallback: conservative policy
            return np.zeros(len(X), dtype=int)  # Deny all
    
    def evaluate_policy(self, X_test: np.ndarray, df_test: pd.DataFrame) -> Dict:
        """Evaluate the learned policy."""
        actions = self.predict(X_test)
        
        total_reward = 0.0
        action_counts = {0: 0, 1: 0}
        
        for i, (_, row) in enumerate(df_test.iterrows()):
            action = actions[i] if i < len(actions) else 0
            action_counts[action] += 1
            
            reward = self.environment.calculate_reward(
                action=action,
                loan_amnt=row['loan_amnt'],
                int_rate=row['int_rate'],
                loan_status=row['target']
            )
            total_reward += reward
        
        avg_reward = total_reward / len(df_test)
        approval_rate = action_counts.get(1, 0) / len(df_test)
        
        logger.info(f"Policy evaluation - Average reward: ${avg_reward:.2f}, "
                   f"Approval rate: {approval_rate:.2%}")
        
        return {
            'average_reward': avg_reward,
            'total_reward': total_reward,
            'approval_rate': approval_rate,
            'action_distribution': action_counts,
            'actions': actions
        }
    
    def compare_with_historical(self, df_test: pd.DataFrame) -> Dict:
        """Compare with historical approval policy."""
        total_historical_reward = 0.0
        
        for _, row in df_test.iterrows():
            reward = self.environment.calculate_reward(
                action=1,  # Always approve
                loan_amnt=row['loan_amnt'],
                int_rate=row['int_rate'],
                loan_status=row['target']
            )
            total_historical_reward += reward
        
        avg_historical_reward = total_historical_reward / len(df_test)
        
        return {
            'historical_average_reward': avg_historical_reward,
            'historical_total_reward': total_historical_reward
        }

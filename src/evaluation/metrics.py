"""
Evaluation metrics and utilities.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from typing import Dict, Tuple, List, Any
import logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation utilities."""
    
    def __init__(self):
        self.results = {}
    
    def evaluate_classifier(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                          y_pred_binary: np.ndarray, model_name: str = "model") -> Dict[str, Any]:
        """Evaluate binary classifier performance."""
        
        # Basic metrics
        auc = roc_auc_score(y_true, y_pred_proba)
        f1 = f1_score(y_true, y_pred_binary)
        precision = precision_score(y_true, y_pred_binary)
        recall = recall_score(y_true, y_pred_binary)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred_binary)
        tn, fp, fn, tp = cm.ravel()
        
        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative predictive value
        
        # ROC curve data
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_proba)
        
        # Precision-Recall curve data
        pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_true, y_pred_proba)
        
        results = {
            'model_name': model_name,
            'auc': auc,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'npv': npv,
            'confusion_matrix': cm,
            'classification_report': classification_report(y_true, y_pred_binary),
            'roc_curve': {'fpr': fpr, 'tpr': tpr, 'thresholds': roc_thresholds},
            'pr_curve': {'precision': pr_precision, 'recall': pr_recall, 'thresholds': pr_thresholds}
        }
        
        self.results[model_name] = results
        
        logger.info(f"{model_name} - AUC: {auc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        
        return results
    
    def evaluate_rl_policy(self, actions: np.ndarray, rewards: np.ndarray, 
                          model_name: str = "rl_agent") -> Dict[str, Any]:
        """Evaluate RL policy performance."""
        
        total_reward = np.sum(rewards)
        average_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        
        # Action distribution
        unique_actions, action_counts = np.unique(actions, return_counts=True)
        action_distribution = dict(zip(unique_actions, action_counts))
        
        # Approval rate (assuming action 1 is approve)
        approval_rate = action_counts[1] / len(actions) if 1 in unique_actions else 0
        
        results = {
            'model_name': model_name,
            'total_reward': total_reward,
            'average_reward': average_reward,
            'std_reward': std_reward,
            'approval_rate': approval_rate,
            'action_distribution': action_distribution,
            'num_decisions': len(actions)
        }
        
        self.results[model_name] = results
        
        logger.info(f"{model_name} - Avg Reward: {average_reward:.2f}, "
                   f"Approval Rate: {approval_rate:.2%}, Total Decisions: {len(actions)}")
        
        return results
    
    def compare_models(self) -> pd.DataFrame:
        """Compare all evaluated models."""
        if not self.results:
            logger.warning("No models to compare. Run evaluation first.")
            return pd.DataFrame()
        
        comparison_data = []
        
        for model_name, results in self.results.items():
            row = {'model_name': model_name}
            
            # Add relevant metrics based on model type
            if 'auc' in results:
                row.update({
                    'auc': results['auc'],
                    'f1_score': results['f1_score'],
                    'precision': results['precision'],
                    'recall': results['recall']
                })
            
            if 'average_reward' in results:
                row.update({
                    'average_reward': results['average_reward'],
                    'approval_rate': results['approval_rate'],
                    'total_reward': results['total_reward']
                })
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def calculate_financial_impact(self, y_true: np.ndarray, actions: np.ndarray,
                                 loan_amounts: np.ndarray, interest_rates: np.ndarray,
                                 model_name: str = "model") -> Dict[str, float]:
        """Calculate financial impact of decisions."""
        
        total_profit = 0.0
        total_loss = 0.0
        missed_profit = 0.0
        avoided_loss = 0.0
        
        for i in range(len(actions)):
            action = actions[i]
            actual_outcome = y_true[i]  # 0: paid, 1: default
            loan_amt = loan_amounts[i]
            int_rate = interest_rates[i]
            
            if action == 1:  # Approved
                if actual_outcome == 0:  # Paid
                    total_profit += loan_amt * (int_rate / 100)
                else:  # Default
                    total_loss += loan_amt
            else:  # Denied
                if actual_outcome == 0:  # Would have paid
                    missed_profit += loan_amt * (int_rate / 100)
                else:  # Would have defaulted
                    avoided_loss += loan_amt
        
        net_profit = total_profit - total_loss
        opportunity_cost = missed_profit - avoided_loss
        
        results = {
            'total_profit': total_profit,
            'total_loss': total_loss,
            'net_profit': net_profit,
            'missed_profit': missed_profit,
            'avoided_loss': avoided_loss,
            'opportunity_cost': opportunity_cost,
            'roi': (net_profit / (total_profit + total_loss)) * 100 if (total_profit + total_loss) > 0 else 0
        }
        
        logger.info(f"{model_name} Financial Impact - Net Profit: ${net_profit:.2f}, ROI: {results['roi']:.2f}%")
        
        return results
    
    def threshold_analysis(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                          thresholds: List[float] = None) -> pd.DataFrame:
        """Analyze performance across different decision thresholds."""
        
        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.1)
        
        results = []
        
        for threshold in thresholds:
            y_pred_binary = (y_pred_proba >= threshold).astype(int)
            
            auc = roc_auc_score(y_true, y_pred_proba)
            f1 = f1_score(y_true, y_pred_binary)
            precision = precision_score(y_true, y_pred_binary)
            recall = recall_score(y_true, y_pred_binary)
            
            approval_rate = np.mean(1 - y_pred_binary)  # 1 - default prediction = approval
            
            results.append({
                'threshold': threshold,
                'auc': auc,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'approval_rate': approval_rate
            })
        
        return pd.DataFrame(results)

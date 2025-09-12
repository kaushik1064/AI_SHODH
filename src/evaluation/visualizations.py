"""
Visualization utilities for model results.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging

logger = logging.getLogger(__name__)

class ResultVisualizer:
    """Visualization utilities for model results and comparisons."""
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (12, 8)):
        plt.style.use(style)
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    def plot_training_history(self, history: Dict[str, List[float]], 
                            save_path: str = None) -> plt.Figure:
        """Plot training history for deep learning model."""
        
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        
        # Loss plot
        epochs = range(1, len(history['train_losses']) + 1)
        axes[0].plot(epochs, history['train_losses'], 'b-', label='Training Loss', alpha=0.8)
        axes[0].plot(epochs, history['val_losses'], 'r-', label='Validation Loss', alpha=0.8)
        axes[0].set_title('Model Loss During Training')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # AUC plot
        axes[1].plot(epochs, history['val_aucs'], 'g-', label='Validation AUC', alpha=0.8)
        axes[1].set_title('Model AUC During Training')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('AUC')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_roc_curves(self, results_dict: Dict[str, Dict], 
                       save_path: str = None) -> plt.Figure:
        """Plot ROC curves for multiple models."""
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for i, (model_name, results) in enumerate(results_dict.items()):
            if 'roc_curve' in results:
                roc_data = results['roc_curve']
                auc_score = results['auc']
                
                ax.plot(roc_data['fpr'], roc_data['tpr'], 
                       color=self.colors[i % len(self.colors)],
                       label=f'{model_name} (AUC = {auc_score:.3f})',
                       linewidth=2)
        
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_precision_recall_curves(self, results_dict: Dict[str, Dict],
                                    save_path: str = None) -> plt.Figure:
        """Plot Precision-Recall curves for multiple models."""
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for i, (model_name, results) in enumerate(results_dict.items()):
            if 'pr_curve' in results:
                pr_data = results['pr_curve']
                
                ax.plot(pr_data['recall'], pr_data['precision'],
                       color=self.colors[i % len(self.colors)],
                       label=f'{model_name}',
                       linewidth=2)
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curves Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_confusion_matrices(self, results_dict: Dict[str, Dict],
                              save_path: str = None) -> plt.Figure:
        """Plot confusion matrices for multiple models."""
        
        n_models = len([r for r in results_dict.values() if 'confusion_matrix' in r])
        
        if n_models == 0:
            logger.warning("No confusion matrices found in results")
            return None
        
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        if n_models == 1:
            axes = [axes]
        
        model_idx = 0
        for model_name, results in results_dict.items():
            if 'confusion_matrix' in results:
                cm = results['confusion_matrix']
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           ax=axes[model_idx],
                           xticklabels=['Predicted: Paid', 'Predicted: Default'],
                           yticklabels=['Actual: Paid', 'Actual: Default'])
                
                axes[model_idx].set_title(f'{model_name}\nConfusion Matrix')
                model_idx += 1
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_feature_importance(self, feature_names: List[str], 
                              importance_scores: np.ndarray,
                              top_n: int = 20, save_path: str = None) -> plt.Figure:
        """Plot feature importance."""
        
        # Get top features
        indices = np.argsort(importance_scores)[-top_n:]
        top_features = [feature_names[i] for i in indices]
        top_scores = importance_scores[indices]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        bars = ax.barh(range(len(top_features)), top_scores, 
                      color=self.colors[0], alpha=0.7)
        
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features)
        ax.set_xlabel('Importance Score')
        ax.set_title(f'Top {top_n} Feature Importance')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                   f'{width:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_reward_distribution(self, rewards: np.ndarray, actions: np.ndarray,
                               save_path: str = None) -> plt.Figure:
        """Plot reward distribution by action."""
        
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        
        # Reward distribution by action
        for action in np.unique(actions):
            action_rewards = rewards[actions == action]
            action_label = 'Deny' if action == 0 else 'Approve'
            
            axes[0].hist(action_rewards, bins=50, alpha=0.7, 
                        label=f'{action_label} (n={len(action_rewards)})')
        
        axes[0].set_xlabel('Reward')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Reward Distribution by Action')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Box plot
        reward_data = []
        action_labels = []
        
        for action in np.unique(actions):
            action_rewards = rewards[actions == action]
            action_label = 'Deny' if action == 0 else 'Approve'
            reward_data.extend(action_rewards)
            action_labels.extend([action_label] * len(action_rewards))
        
        df_rewards = pd.DataFrame({'Reward': reward_data, 'Action': action_labels})
        sns.boxplot(data=df_rewards, x='Action', y='Reward', ax=axes[1])
        axes[1].set_title('Reward Distribution by Action (Box Plot)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_threshold_analysis(self, threshold_df: pd.DataFrame,
                              save_path: str = None) -> plt.Figure:
        """Plot threshold analysis results."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # F1 Score vs Threshold
        axes[0, 0].plot(threshold_df['threshold'], threshold_df['f1_score'], 
                       'b-', marker='o', linewidth=2)
        axes[0, 0].set_xlabel('Threshold')
        axes[0, 0].set_ylabel('F1 Score')
        axes[0, 0].set_title('F1 Score vs Threshold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Precision vs Recall
        axes[0, 1].plot(threshold_df['recall'], threshold_df['precision'], 
                       'r-', marker='s', linewidth=2)
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Precision vs Recall')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Approval Rate vs Threshold
        axes[1, 0].plot(threshold_df['threshold'], threshold_df['approval_rate'], 
                       'g-', marker='^', linewidth=2)
        axes[1, 0].set_xlabel('Threshold')
        axes[1, 0].set_ylabel('Approval Rate')
        axes[1, 0].set_title('Approval Rate vs Threshold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Combined metrics
        axes[1, 1].plot(threshold_df['threshold'], threshold_df['precision'], 
                       'r-', marker='s', label='Precision', linewidth=2)
        axes[1, 1].plot(threshold_df['threshold'], threshold_df['recall'], 
                       'b-', marker='o', label='Recall', linewidth=2)
        axes[1, 1].plot(threshold_df['threshold'], threshold_df['f1_score'], 
                       'g-', marker='^', label='F1 Score', linewidth=2)
        axes[1, 1].set_xlabel('Threshold')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('All Metrics vs Threshold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame,
                            save_path: str = None) -> plt.Figure:
        """Plot model comparison metrics."""
        
        # Determine which metrics are available
        metric_columns = [col for col in comparison_df.columns 
                         if col not in ['model_name']]
        
        n_metrics = len(metric_columns)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, metric in enumerate(metric_columns):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            bars = ax.bar(comparison_df['model_name'], comparison_df[metric],
                         color=self.colors[i % len(self.colors)], alpha=0.7)
            
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom')
        
        # Hide unused subplots
        for i in range(n_metrics, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if n_rows > 1:
                axes[row, col].set_visible(False)
            elif n_cols > 1:
                axes[col].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_interactive_dashboard(self, results_dict: Dict[str, Dict]) -> go.Figure:
        """Create interactive dashboard with plotly."""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('ROC Curves', 'Precision-Recall Curves', 
                          'Model Comparison', 'Feature Analysis'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # ROC Curves
        for i, (model_name, results) in enumerate(results_dict.items()):
            if 'roc_curve' in results:
                roc_data = results['roc_curve']
                auc_score = results['auc']
                
                fig.add_trace(
                    go.Scatter(x=roc_data['fpr'], y=roc_data['tpr'],
                             mode='lines', name=f'{model_name} (AUC={auc_score:.3f})',
                             line=dict(color=self.colors[i % len(self.colors)])),
                    row=1, col=1
                )
        
        # Diagonal line for ROC
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                     line=dict(dash='dash', color='black'),
                     showlegend=False),
            row=1, col=1
        )
        
        # Precision-Recall Curves
        for i, (model_name, results) in enumerate(results_dict.items()):
            if 'pr_curve' in results:
                pr_data = results['pr_curve']
                
                fig.add_trace(
                    go.Scatter(x=pr_data['recall'], y=pr_data['precision'],
                             mode='lines', name=f'{model_name}',
                             line=dict(color=self.colors[i % len(self.colors)]),
                             showlegend=False),
                    row=1, col=2
                )
        
        # Update layout
        fig.update_layout(
            title="Model Performance Dashboard",
            height=800,
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="False Positive Rate", row=1, col=1)
        fig.update_yaxes(title_text="True Positive Rate", row=1, col=1)
        fig.update_xaxes(title_text="Recall", row=1, col=2)
        fig.update_yaxes(title_text="Precision", row=1, col=2)
        
        return fig

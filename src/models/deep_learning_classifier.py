"""
Deep learning classifier for loan default prediction.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from typing import Tuple, Dict, List
import logging
from config.config import MODEL_CONFIG

logger = logging.getLogger(__name__)

class DeepLearningClassifier(nn.Module):
    """Deep learning classifier for binary classification."""
    
    def __init__(self, input_dim: int, hidden_layers: List[int] = None, 
                 dropout_rate: float = 0.3):
        super(DeepLearningClassifier, self).__init__()
        
        if hidden_layers is None:
            hidden_layers = MODEL_CONFIG.dl_hidden_layers
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class LoanClassifierTrainer:
    """Trainer for the loan default classifier."""
    
    def __init__(self, model: DeepLearningClassifier, device: str = None):
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), 
                                  lr=MODEL_CONFIG.dl_learning_rate)
        self.criterion = nn.BCELoss()
        
        self.train_losses = []
        self.val_losses = []
        self.val_aucs = []
        
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data).squeeze()
            loss = self.criterion(output, target.float())
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data).squeeze()
                loss = self.criterion(output, target.float())
                total_loss += loss.item()
                
                all_predictions.extend(output.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        auc = roc_auc_score(all_targets, all_predictions)
        
        return avg_loss, auc
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Train the model."""
        # Create data loaders
        train_dataset = TensorDataset(torch.FloatTensor(X_train), 
                                    torch.LongTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), 
                                  torch.LongTensor(y_val))
        
        train_loader = DataLoader(train_dataset, 
                                batch_size=MODEL_CONFIG.dl_batch_size, 
                                shuffle=True)
        val_loader = DataLoader(val_dataset, 
                              batch_size=MODEL_CONFIG.dl_batch_size, 
                              shuffle=False)
        
        best_val_auc = 0.0
        patience_counter = 0
        
        for epoch in range(MODEL_CONFIG.dl_epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_auc = self.validate(val_loader)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_aucs.append(val_auc)
            
            # Early stopping
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'results/models/best_dl_model.pth')
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
            
            if patience_counter >= MODEL_CONFIG.dl_early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('results/models/best_dl_model.pth'))
        
        return {
            'best_val_auc': best_val_auc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_aucs': self.val_aucs
        }
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions."""
        self.model.eval()
        dataset = TensorDataset(torch.FloatTensor(X))
        loader = DataLoader(dataset, batch_size=MODEL_CONFIG.dl_batch_size, 
                          shuffle=False)
        
        predictions = []
        with torch.no_grad():
            for (data,) in loader:
                data = data.to(self.device)
                output = self.model(data).squeeze()
                predictions.extend(output.cpu().numpy())
        
        probabilities = np.array(predictions)
        binary_predictions = (probabilities > 0.5).astype(int)
        
        return probabilities, binary_predictions
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate the model on test data."""
        probabilities, predictions = self.predict(X_test)
        
        auc = roc_auc_score(y_test, probabilities)
        f1 = f1_score(y_test, predictions)
        
        logger.info(f"Test AUC: {auc:.4f}, Test F1: {f1:.4f}")
        
        return {
            'auc': auc,
            'f1': f1,
            'predictions': predictions,
            'probabilities': probabilities,
            'classification_report': classification_report(y_test, predictions)
        }

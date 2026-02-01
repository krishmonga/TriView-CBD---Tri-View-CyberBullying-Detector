"""
Attention Optimization for Meaningful View Weights
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict

class AttentionOptimizer:
    """Optimizes attention weights for meaningful view contributions"""
    
    @staticmethod
    def compute_attention_entropy(attention_weights: torch.Tensor) -> torch.Tensor:
        """Compute entropy of attention weights (higher = more diverse)"""
        epsilon = 1e-10
        entropy = -torch.sum(attention_weights * torch.log(attention_weights + epsilon), dim=1)
        return entropy.mean()
    
    @staticmethod
    def compute_attention_diversity(attention_weights: torch.Tensor) -> torch.Tensor:
        """Compute diversity of attention across views"""
        batch_size, num_views = attention_weights.shape
        diversity = 0.0
        
        for i in range(num_views):
            for j in range(i+1, num_views):
                diversity += torch.mean(torch.abs(attention_weights[:, i] - attention_weights[:, j]))
        
        num_pairs = num_views * (num_views - 1) / 2
        return diversity / num_pairs
    
    @staticmethod
    def attention_regularization_loss(attention_weights: torch.Tensor,
                                     entropy_weight: float = 0.1,
                                     diversity_weight: float = 0.1) -> torch.Tensor:
        """Compute regularization loss for attention weights"""
        entropy_loss = -AttentionOptimizer.compute_attention_entropy(attention_weights)
        diversity_loss = -AttentionOptimizer.compute_attention_diversity(attention_weights)
        
        total_loss = (entropy_weight * entropy_loss + 
                     diversity_weight * diversity_loss)
        
        return total_loss
    
    @staticmethod
    def analyze_attention_patterns(attention_weights: torch.Tensor, 
                                  view_names: List[str] = None) -> Dict:
        """Analyze attention patterns for interpretability"""
        if view_names is None:
            view_names = ['Lexical', 'Semantic', 'Structural']
        
        attention_weights_np = attention_weights.detach().cpu().numpy()
        
        analysis = {
            'mean_weights': {},
            'std_weights': {},
            'dominant_view_distribution': {},
            'attention_entropy': float(AttentionOptimizer.compute_attention_entropy(
                torch.tensor(attention_weights_np)
            ).item())
        }
        
        # Compute mean and std for each view
        for i, view_name in enumerate(view_names):
            analysis['mean_weights'][view_name] = float(np.mean(attention_weights_np[:, i]))
            analysis['std_weights'][view_name] = float(np.std(attention_weights_np[:, i]))
        
        # Compute dominant view distribution
        dominant_views = np.argmax(attention_weights_np, axis=1)
        unique, counts = np.unique(dominant_views, return_counts=True)
        
        for view_idx, count in zip(unique, counts):
            view_name = view_names[view_idx]
            percentage = float(count / len(attention_weights_np) * 100)
            analysis['dominant_view_distribution'][view_name] = {
                'count': int(count),
                'percentage': percentage
            }
        
        return analysis

class LearnableTemperature(nn.Module):
    """Learnable temperature parameter for softmax"""
    def __init__(self, initial_temp: float = 1.0):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(initial_temp)))
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature-scaled softmax"""
        temperature = torch.exp(self.log_temperature).clamp(min=0.1, max=10.0)
        return F.softmax(logits / temperature, dim=-1)
    
    def get_temperature(self) -> float:
        """Get current temperature value"""
        return float(torch.exp(self.log_temperature).item())

class GatedAttentionMechanism(nn.Module):
    """Gated attention mechanism"""
    def __init__(self, input_dim: int, num_views: int = 3, 
                 hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.num_views = num_views
        
        # Projections
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_views)
        )
        
        # Temperature
        self.temperature = LearnableTemperature(initial_temp=1.0)
        
    def forward(self, view_features: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            view_features: List of tensors, each shape (batch_size, feature_dim)
        
        Returns:
            attention_weights: (batch_size, num_views)
            fused_features: (batch_size, feature_dim)
        """
        batch_size = view_features[0].shape[0]
        
        # Stack features
        stacked = torch.stack(view_features, dim=1)  # (batch_size, num_views, feature_dim)
        
        # Compute queries and keys
        queries = self.query(stacked)  # (batch_size, num_views, hidden_dim)
        keys = self.key(stacked)
        
        # Compute attention scores
        scores = torch.matmul(queries, keys.transpose(1, 2))  # (batch_size, num_views, num_views)
        scores = scores.mean(dim=-1)  # (batch_size, num_views)
        
        # Apply temperature
        attention_weights = self.temperature(scores)
        
        # Apply gating
        gate_input = torch.cat([queries.mean(dim=1), keys.mean(dim=1)], dim=1)
        gate_weights = torch.sigmoid(self.gate(gate_input))
        
        # Combine attention and gate
        final_weights = attention_weights * gate_weights
        final_weights = F.softmax(final_weights, dim=1)
        
        # Weighted sum
        weighted_features = torch.sum(
            stacked * final_weights.unsqueeze(-1), 
            dim=1
        )
        
        return final_weights, weighted_features

class EnhancedAttentionTrainer:
    """Enhanced trainer with attention optimization"""
    
    def __init__(self, model, optimizer, device, config=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.config = config or {}
        
        # Loss functions
        self.classification_loss = nn.CrossEntropyLoss()
        
        # Attention regularization weights
        self.entropy_weight = self.config.get('attention_entropy_weight', 0.05)
        self.diversity_weight = self.config.get('attention_diversity_weight', 0.1)
    
    def compute_total_loss(self, logits, labels, attention_weights=None):
        """Compute total loss with attention regularization"""
        # Classification loss
        cls_loss = self.classification_loss(logits, labels)
        
        # Attention regularization
        attn_loss = 0
        if attention_weights is not None:
            attn_loss = AttentionOptimizer.attention_regularization_loss(
                attention_weights,
                entropy_weight=self.entropy_weight,
                diversity_weight=self.diversity_weight
            )
        
        total_loss = cls_loss + attn_loss
        
        return total_loss, cls_loss, attn_loss
    
    def train_step(self, sequences, labels, texts):
        """Single training step"""
        self.model.train()
        
        # Forward pass
        if hasattr(self.model, 'forward_full'):
            logits, attention_info = self.model(texts, sequences)
            attention_weights = attention_info.get('attention_weights', None)
        else:
            logits = self.model(sequences)
            attention_weights = None
        
        # Compute loss
        total_loss, cls_loss, attn_loss = self.compute_total_loss(logits, labels, attention_weights)
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        # Optimize
        self.optimizer.step()
        
        return total_loss.item(), cls_loss.item(), attn_loss.item() if attention_weights is not None else 0
    
    def evaluate(self, data_loader, return_predictions=False):
        """Evaluate model"""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for sequences, labels, texts in data_loader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                
                if hasattr(self.model, 'forward_full'):
                    logits, attention_info = self.model(texts, sequences)
                else:
                    logits = self.model(sequences)
                
                # Compute loss
                loss, _, _ = self.compute_total_loss(logits, labels)
                total_loss += loss.item() * sequences.size(0)
                total_samples += sequences.size(0)
                
                # Predictions
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        
        if return_predictions:
            return avg_loss, all_preds, all_labels
        else:
            return avg_loss
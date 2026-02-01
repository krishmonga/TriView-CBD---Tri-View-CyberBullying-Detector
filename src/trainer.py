"""
Optimized Trainer for High Accuracy
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple, List
import numpy as np
from tqdm import tqdm
import os

class FocalLoss(nn.Module):
    """Focal Loss for class imbalance"""
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class OptimizedTrainer:
    """Optimized Trainer for high accuracy"""
    
    def __init__(self, model, config: Dict, device: torch.device):
        self.model = model
        self.config = config
        self.device = device
        self.model.to(device)
        
        # Loss function
        self.criterion = FocalLoss(gamma=2.0, alpha=0.25)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        
        self.best_val_acc = 0
        self.history = {'train_loss': [], 'val_acc': [], 'train_acc': []}
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training', leave=False)
        for sequences, labels, texts in pbar:
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            if hasattr(self.model, 'forward_full'):
                logits, _ = self.model(texts, sequences)
            else:
                logits = self.model(sequences)
            
            # Calculate loss
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': correct / total if total > 0 else 0
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total if total > 0 else 0
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate model"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for sequences, labels, texts in val_loader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                
                if hasattr(self.model, 'forward_full'):
                    logits, _ = self.model(texts, sequences)
                else:
                    logits = self.model(sequences)
                
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        accuracy = correct / total if total > 0 else 0
        return accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int = 50) -> Dict:
        """Main training loop"""
        patience_counter = 0
        patience = self.config['training']['patience']
        
        print(f"🚀 Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_acc = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | "
                  f"Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
                print(f"  💾 Saved best model (Val Acc: {val_acc:.4f})")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"  ⏹️ Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        if os.path.exists('best_model.pth'):
            self.model.load_state_dict(torch.load('best_model.pth'))
        
        print(f"✅ Training completed! Best Val Acc: {self.best_val_acc:.4f}")
        return self.history
    
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """Evaluate model on test set"""
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for sequences, labels, texts in test_loader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                
                if hasattr(self.model, 'forward_full'):
                    logits, _ = self.model(texts, sequences)
                else:
                    logits = self.model(sequences)
                
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'predictions': all_preds,
            'labels': all_labels
        }
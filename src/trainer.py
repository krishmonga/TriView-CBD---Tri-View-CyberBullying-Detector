"""
Standalone trainer module — used by main.py's train_model() internally.
Kept for backward compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple
import os
from tqdm import tqdm


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce)
        return (self.alpha * (1 - pt) ** self.gamma * ce).mean()


class Trainer:
    def __init__(self, model, config: Dict, device: torch.device):
        self.model = model.to(device)
        self.device = device
        tc = config["training"]
        self.criterion = FocalLoss(tc.get("focal_gamma", 2.0),
                                   tc.get("focal_alpha", 0.25))
        self.optimizer = optim.AdamW(model.parameters(),
                                     lr=tc["learning_rate"],
                                     weight_decay=tc["weight_decay"])
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=tc.get("scheduler_T0", 10),
            T_mult=tc.get("scheduler_T_mult", 2),
            eta_min=tc.get("scheduler_eta_min", 1e-6),
        )
        self.patience = tc["patience"]
        self.grad_clip = tc.get("gradient_clip", 1.0)
        self.best_val_acc = 0
        self.history = {"train_loss": [], "val_acc": [], "train_acc": []}

    def train_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        total_loss, correct, total = 0, 0, 0
        for seqs, labels, _ in loader:
            seqs, labels = seqs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(seqs)
            loss = self.criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            total_loss += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
        return total_loss / len(loader), correct / total if total else 0

    def validate(self, loader: DataLoader) -> float:
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for seqs, labels, _ in loader:
                seqs, labels = seqs.to(self.device), labels.to(self.device)
                logits = self.model(seqs)
                correct += (logits.argmax(1) == labels).sum().item()
                total += labels.size(0)
        return correct / total if total else 0

    def train(self, train_loader, val_loader, epochs=100) -> Dict:
        patience_ctr = 0
        for epoch in range(epochs):
            loss, tr_acc = self.train_epoch(train_loader)
            va_acc = self.validate(val_loader)
            self.scheduler.step()
            self.history["train_loss"].append(loss)
            self.history["train_acc"].append(tr_acc)
            self.history["val_acc"].append(va_acc)
            print(f"  Epoch {epoch+1:3d}/{epochs}  loss={loss:.4f}  "
                  f"tr_acc={tr_acc:.4f}  va_acc={va_acc:.4f}")
            if va_acc > self.best_val_acc:
                self.best_val_acc = va_acc
                patience_ctr = 0
                torch.save(self.model.state_dict(), "best_model.pth")
            else:
                patience_ctr += 1
                if patience_ctr >= self.patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break
        if os.path.exists("best_model.pth"):
            self.model.load_state_dict(torch.load("best_model.pth", weights_only=True))
        return self.history

    def evaluate(self, loader: DataLoader) -> Dict:
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        self.model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for seqs, labs, _ in loader:
                seqs = seqs.to(self.device)
                logits = self.model(seqs)
                preds.extend(logits.argmax(1).cpu().numpy())
                labels.extend(labs.numpy())
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_score": f1_score(labels, preds, average="weighted"),
            "precision": precision_score(labels, preds, average="weighted"),
            "recall": recall_score(labels, preds, average="weighted"),
        }

"""
Baseline Models for Comparison
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class BiLSTMBaseline(nn.Module):
    """BiLSTM Baseline Model"""
    def __init__(self, vocab_size: int, embed_dim: int = 300, 
                 hidden_dim: int = 256, num_layers: int = 2,
                 num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        
        lstm_out, _ = self.lstm(embedded)
        
        # Attention
        attention_scores = self.attention(lstm_out)
        padding_mask = (x == 0).unsqueeze(-1)
        attention_scores = attention_scores.masked_fill(padding_mask, float('-inf'))
        attention_weights = F.softmax(attention_scores, dim=1)
        
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        logits = self.classifier(context)
        return logits

class CNNBaseline(nn.Module):
    """CNN Baseline Model"""
    def __init__(self, vocab_size: int, embed_dim: int = 300,
                 num_filters: int = 128, filter_sizes: List[int] = None,
                 num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        
        if filter_sizes is None:
            filter_sizes = [2, 3, 4, 5]
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embed_dim, num_filters, kernel_size=fs, padding=fs//2),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1)
            ) for fs in filter_sizes
        ])
        
        self.classifier = nn.Sequential(
            nn.Linear(num_filters * len(filter_sizes), 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        embedded = embedded.permute(0, 2, 1)
        
        conv_outputs = []
        for conv in self.convs:
            conv_out = conv(embedded)
            conv_out = conv_out.squeeze(-1)
            conv_outputs.append(conv_out)
        
        combined = torch.cat(conv_outputs, dim=1)
        logits = self.classifier(combined)
        return logits
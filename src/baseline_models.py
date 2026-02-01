"""
Baseline Models for Comparison - Including BERT, LightGBM, Tuned LSTM, and Random Forest
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import numpy as np
try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    AutoTokenizer = None
    AutoModel = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    from sklearn.ensemble import RandomForestClassifier
except ImportError:
    RandomForestClassifier = None

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


class BERTBaseline(nn.Module):
    """BERT-based Model for Cyberbullying Detection"""
    def __init__(self, vocab_size: int = None, embed_dim: int = 768,
                 num_classes: int = 2, dropout: float = 0.3, 
                 model_name: str = "distilbert-base-uncased"):
        super().__init__()
        
        try:
            from transformers import AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.bert = AutoModel.from_pretrained(model_name)
            self.bert_dim = self.bert.config.hidden_size
        except ImportError:
            print("⚠️  Transformers not installed. BERT model will be unavailable.")
            self.bert = None
            self.tokenizer = None
            self.bert_dim = embed_dim
        
        # Classifier on top of BERT
        self.classifier = nn.Sequential(
            nn.Linear(self.bert_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and 'bert' not in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name and 'bert' not in name:
                nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Can be token IDs or embeddings depending on input
        """
        if self.bert is not None and x.shape[-1] <= 512:  # Likely token IDs
            # Use BERT
            outputs = self.bert(x, attention_mask=(x != 0).int())
            pooled = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token
        else:
            # Fallback: assume embeddings are provided
            if len(x.shape) == 3:
                pooled = x.mean(dim=1)  # Average pooling
            else:
                pooled = x
        
        logits = self.classifier(pooled)
        return logits


class TunedLSTMBaseline(nn.Module):
    """Highly Optimized LSTM with Bidirectional, Multi-head Attention, and Residual Connections"""
    def __init__(self, vocab_size: int, embed_dim: int = 300,
                 hidden_dim: int = 256, num_layers: int = 3,
                 num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Multi-layer BiLSTM with higher capacity
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        lstm_out_dim = hidden_dim * 2
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_out_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Residual connection projection
        self.residual_proj = nn.Linear(lstm_out_dim, lstm_out_dim) if lstm_out_dim != embed_dim else None
        
        # Classifier with layer normalization
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim * 2, 512),  # 2x for mean+max pooling
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        
        # BiLSTM
        lstm_out, (h_n, c_n) = self.lstm(embedded)
        
        # Multi-head attention
        padding_mask = (x == 0)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out, key_padding_mask=padding_mask)
        
        # Mean and max pooling
        mask_expanded = padding_mask.unsqueeze(-1).expand(attn_out.size())
        attn_out_masked = attn_out.masked_fill(mask_expanded, float('-inf'))
        
        mean_pooled = torch.mean(attn_out, dim=1)
        max_pooled, _ = torch.max(attn_out, dim=1)
        
        combined = torch.cat([mean_pooled, max_pooled], dim=1)
        
        logits = self.classifier(combined)
        return logits


class RandomForestEnsemble(nn.Module):
    """Random Forest wrapper as PyTorch Module for consistency"""
    def __init__(self, vocab_size: int, embed_dim: int = 300,
                 num_classes: int = 2, n_estimators: int = 100):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        # Initialize sklearn RandomForest
        try:
            from sklearn.ensemble import RandomForestClassifier
            self.rf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
            self.is_fitted = False
        except ImportError:
            print("⚠️  scikit-learn not installed. RandomForest model will be unavailable.")
            self.rf = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - extract features from embeddings
        """
        embedded = self.embedding(x)
        
        # Extract features: mean, max, min pooling + statistical features
        features = []
        
        # Mean pooling
        mean_feat = torch.mean(embedded, dim=1)  # (batch, embed_dim)
        features.append(mean_feat)
        
        # Max pooling
        max_feat, _ = torch.max(embedded, dim=1)  # (batch, embed_dim)
        features.append(max_feat)
        
        # Min pooling
        min_feat, _ = torch.min(embedded, dim=1)  # (batch, embed_dim)
        features.append(min_feat)
        
        # Std dev
        std_feat = torch.std(embedded, dim=1)  # (batch, embed_dim)
        features.append(std_feat)
        
        # Concatenate all features
        feature_vec = torch.cat(features, dim=1)  # (batch, embed_dim * 4)
        
        # Convert to numpy for sklearn
        features_np = feature_vec.detach().cpu().numpy()
        
        if self.rf is None:
            # Fallback: simple linear classifier
            linear = nn.Linear(feature_vec.shape[1], self.num_classes)
            logits = linear(feature_vec)
        else:
            if self.is_fitted:
                predictions = self.rf.predict(features_np)
                probabilities = self.rf.predict_proba(features_np)
                logits = torch.tensor(np.log(probabilities + 1e-10), dtype=torch.float32)
            else:
                # Not fitted yet, return dummy logits
                logits = torch.zeros(x.shape[0], self.num_classes)
        
        return logits
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train RandomForest on features"""
        if self.rf is not None:
            self.rf.fit(X_train, y_train)
            self.is_fitted = True


class LightGBMEnsemble(nn.Module):
    """LightGBM wrapper as PyTorch Module for consistency"""
    def __init__(self, vocab_size: int, embed_dim: int = 300,
                 num_classes: int = 2):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        # Initialize LightGBM
        try:
            import lightgbm as lgb
            self.lgb_model = lgb.LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=10,
                num_leaves=31,
                min_child_samples=10,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                n_jobs=-1,
                is_unbalance=True
            )
            self.is_fitted = False
        except ImportError:
            print("⚠️  LightGBM not installed. LightGBM model will be unavailable.")
            self.lgb_model = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - extract features from embeddings
        """
        embedded = self.embedding(x)
        
        # Extract rich features for LightGBM
        features = []
        
        # Positional features (first, last, middle words)
        first_token = embedded[:, 0, :]
        last_token = embedded[:, -1, :]
        mid_token = embedded[:, embedded.shape[1]//2, :]
        
        features.extend([first_token, last_token, mid_token])
        
        # Statistical features
        mean_feat = torch.mean(embedded, dim=1)
        max_feat, _ = torch.max(embedded, dim=1)
        min_feat, _ = torch.min(embedded, dim=1)
        std_feat = torch.std(embedded, dim=1)
        
        features.extend([mean_feat, max_feat, min_feat, std_feat])
        
        # Concatenate all features
        feature_vec = torch.cat(features, dim=1)
        features_np = feature_vec.detach().cpu().numpy()
        
        if self.lgb_model is None:
            # Fallback
            linear = nn.Linear(feature_vec.shape[1], self.num_classes)
            logits = linear(feature_vec)
        else:
            if self.is_fitted:
                probabilities = self.lgb_model.predict_proba(features_np)
                logits = torch.tensor(np.log(probabilities + 1e-10), dtype=torch.float32)
            else:
                # Not fitted, return dummy logits
                logits = torch.zeros(x.shape[0], self.num_classes)
        
        return logits
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train LightGBM on features"""
        if self.lgb_model is not None:
            self.lgb_model.fit(X_train, y_train)
            self.is_fitted = True
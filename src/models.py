"""
OPTIMIZED MODELS - FIXED WEIGHT INITIALIZATION
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Dict

class PositionalEncoding(nn.Module):
    """Positional encoding"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]

class EnhancedLexicalView(nn.Module):
    """Enhanced lexical view with CNN"""
    def __init__(self, vocab_size: int, embed_dim: int = 300, 
                 dropout_rate: float = 0.3, config: Dict = None):
        super().__init__()
        
        # Use config values if provided
        config = config or {}
        
        # FIX: Get embed_dim from config if provided
        if 'embed_dim' in config:
            embed_dim = config['embed_dim']
        
        filter_sizes = config.get('lexical_filters', [2, 3, 4, 5])
        num_filters = config.get('lexical_num_filters', 128)
        input_features = config.get('lexical_features', num_filters * len(filter_sizes))
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Multiple CNN filters
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embed_dim, num_filters, kernel_size=fs, padding=fs//2),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
                nn.Dropout(dropout_rate * 0.5),
                nn.Conv1d(num_filters, num_filters, kernel_size=fs, padding=fs//2),
                nn.BatchNorm1d(num_filters),
                nn.ReLU()
            ) for fs in filter_sizes
        ])
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(num_filters * len(filter_sizes), 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        
        # Output layer - FIXED: Use configurable input_features
        self.output = nn.Sequential(
            nn.Linear(input_features, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Fixed weight initialization"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:  # Only initialize 2D+ weights
                    if 'conv' in name:
                        nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                    else:
                        nn.init.xavier_normal_(param)
                elif len(param.shape) == 1:
                    # For 1D parameters like bias or layer norm
                    if 'bias' in name:
                        nn.init.zeros_(param)
                    elif 'weight' in name and ('norm' in name or 'bn' in name):
                        nn.init.ones_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        embedded = embedded.permute(0, 2, 1)  # (batch_size, embed_dim, seq_len)
        
        # Apply each CNN - FIX: Ensure same output size
        conv_outputs = []
        for conv in self.convs:
            conv_out = conv(embedded)
            # Check if we need to trim/pad
            if conv_out.size(2) != embedded.size(2):
                # Trim or pad to match input seq_len
                target_size = embedded.size(2)
                current_size = conv_out.size(2)
                if current_size > target_size:
                    # Trim from center
                    start = (current_size - target_size) // 2
                    conv_out = conv_out[:, :, start:start+target_size]
                elif current_size < target_size:
                    # Pad with zeros
                    pad_left = (target_size - current_size) // 2
                    pad_right = target_size - current_size - pad_left
                    conv_out = F.pad(conv_out, (pad_left, pad_right))
            
            conv_outputs.append(conv_out)
        
        # Concatenate along feature dimension
        combined = torch.cat(conv_outputs, dim=1)  # (batch_size, num_filters*len(filters), seq_len)
        combined = combined.permute(0, 2, 1)  # (batch_size, seq_len, features)
        
        # Apply attention
        attention_scores = self.attention(combined)  # (batch_size, seq_len, 1)
        
        # Mask padding
        padding_mask = (x == 0).unsqueeze(-1)
        attention_scores = attention_scores.masked_fill(padding_mask, float('-inf'))
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Weighted sum
        context = torch.sum(attention_weights * combined, dim=1)  # (batch_size, features)
        
        # Final projection
        output = self.output(context)
        return output

class EnhancedSemanticView(nn.Module):
    """Enhanced semantic view with Transformer - UPDATED to match config"""
    def __init__(self, vocab_size: int, embed_dim: int = 300,
                 dropout_rate: float = 0.3, config: Dict = None):
        super().__init__()
        
        config = config or {}
        
        # FIX 1: Get embed_dim from config if provided
        if 'embed_dim' in config:
            embed_dim = config['embed_dim']
        
        # FIX 2: Get semantic_features from config
        semantic_features = config.get('semantic_features', embed_dim * 2)
        
        # FIX 3: Ensure embed_dim is divisible by num_heads
        num_heads = config.get('num_heads', 8)
        if embed_dim % num_heads != 0:
            # Adjust embed_dim to be divisible by num_heads
            embed_dim = embed_dim + (num_heads - embed_dim % num_heads)
        
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, self.embed_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(self.embed_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=num_heads,
            dim_feedforward=self.embed_dim * 4,
            dropout=dropout_rate,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Multi-head attention pooling
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=4,  # Use fewer heads for pooling
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Output layer - FIXED: Use configurable semantic_features
        # IMPORTANT: semantic_features should be embed_dim * 2 = 304 * 2 = 608
        self.output = nn.Sequential(
            nn.Linear(semantic_features, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Fixed weight initialization"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_normal_(param)
                elif len(param.shape) == 1:
                    if 'bias' in name:
                        nn.init.zeros_(param)
                    elif 'weight' in name and ('norm' in name or 'ln' in name):
                        nn.init.ones_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        embedded = self.pos_encoder(embedded)
        
        # Create mask
        padding_mask = (x == 0)
        
        # Transformer encoding
        encoded = self.transformer(embedded, src_key_padding_mask=padding_mask)
        
        # Attention pooling
        attn_output, _ = self.attention(encoded, encoded, encoded, 
                                       key_padding_mask=padding_mask)
        
        # Combine mean and max pooling
        mean_pooled = attn_output.mean(dim=1)
        max_pooled, _ = attn_output.max(dim=1)
        combined = torch.cat([mean_pooled, max_pooled], dim=1)
        
        output = self.output(combined)
        return output

class EnhancedStructuralView(nn.Module):
    """Enhanced structural view with BiLSTM - UPDATED to match config"""
    def __init__(self, vocab_size: int, embed_dim: int = 300,
                 dropout_rate: float = 0.3, config: Dict = None):
        super().__init__()
        
        config = config or {}
        
        # FIX: Get embed_dim from config if provided
        if 'embed_dim' in config:
            embed_dim = config['embed_dim']
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # BiLSTM
        hidden_size = embed_dim // 2
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout_rate
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # Highway network
        self.highway = nn.Linear(hidden_size * 2, hidden_size * 2)
        
        # Output layer
        # IMPORTANT: hidden_size * 2 should be embed_dim = 304
        self.output = nn.Sequential(
            nn.Linear(hidden_size * 2, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Fixed weight initialization"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    if 'lstm' in name:
                        # Special initialization for LSTM
                        for i in range(0, param.size(0), self.lstm.hidden_size * 4):
                            nn.init.orthogonal_(param[i:i+self.lstm.hidden_size])
                            nn.init.orthogonal_(param[i+self.lstm.hidden_size:i+self.lstm.hidden_size*2])
                            nn.init.xavier_uniform_(param[i+self.lstm.hidden_size*2:i+self.lstm.hidden_size*3])
                            nn.init.xavier_uniform_(param[i+self.lstm.hidden_size*3:i+self.lstm.hidden_size*4])
                    else:
                        nn.init.xavier_normal_(param)
                elif len(param.shape) == 1:
                    if 'bias' in name:
                        nn.init.zeros_(param)
                    elif 'weight' in name and ('norm' in name or 'ln' in name):
                        nn.init.ones_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        
        # LSTM
        lstm_out, _ = self.lstm(embedded)
        
        # Highway network
        gate = torch.sigmoid(self.highway(lstm_out))
        highway_out = gate * lstm_out + (1 - gate) * embedded
        
        # Attention pooling
        attention_scores = self.attention(highway_out)
        padding_mask = (x == 0).unsqueeze(-1)
        attention_scores = attention_scores.masked_fill(padding_mask, float('-inf'))
        attention_weights = F.softmax(attention_scores, dim=1)
        
        context = torch.sum(attention_weights * highway_out, dim=1)
        
        output = self.output(context)
        return output

class SingleViewClassifier(nn.Module):
    """Classifier for single view models"""
    def __init__(self, view_model, num_classes=2, dropout_rate=0.3):
        super().__init__()
        self.view = view_model
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        # Initialize classifier weights
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        features = self.view(x)
        logits = self.classifier(features)
        return logits

class OptimizedTriFuseModel(nn.Module):
    """Optimized TriFuse Model - FIXED INITIALIZATION"""
    def __init__(self, vocab_size: int, config: Dict = None):
        super().__init__()
        self.config = config or {}
        
        # Initialize views - PASS embed_dim from config
        view_config = config.copy() if config else {}
        
        # Ensure all views use the same embed_dim
        if 'embed_dim' in view_config:
            # Add embed_dim to each view's config
            view_config['embed_dim'] = view_config['embed_dim']
        
        self.lexical_view = EnhancedLexicalView(vocab_size, config=view_config)
        self.semantic_view = EnhancedSemanticView(vocab_size, config=view_config)
        self.structural_view = EnhancedStructuralView(vocab_size, config=view_config)
        
        # Dynamic attention weights
        self.attention_weights = nn.Parameter(torch.ones(3))
        
        # Enhanced fusion
        self.fusion = nn.Sequential(
            nn.Linear(256 * 3, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(config.get('dropout_rate', 0.3)),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(config.get('dropout_rate', 0.3)),
            nn.Linear(512, 256)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(config.get('dropout_rate', 0.3)),
            nn.Linear(128, config.get('num_classes', 2))
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Fixed weight initialization for TriFuse"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_normal_(param)
                elif len(param.shape) == 1:
                    if 'bias' in name:
                        nn.init.zeros_(param)
                    elif 'weight' in name and ('norm' in name or 'ln' in name):
                        nn.init.ones_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        # Initialize attention weights
        nn.init.ones_(self.attention_weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Simplified forward that accepts only tensor"""
        # Get features from each view
        lexical_features = self.lexical_view(x)
        semantic_features = self.semantic_view(x)
        structural_features = self.structural_view(x)
        
        # Dynamic attention
        attention_weights = F.softmax(self.attention_weights, dim=0)
        
        # Apply attention
        weighted_lexical = lexical_features * attention_weights[0]
        weighted_semantic = semantic_features * attention_weights[1]
        weighted_structural = structural_features * attention_weights[2]
        
        # Concatenate
        combined = torch.cat([
            weighted_lexical,
            weighted_semantic,
            weighted_structural
        ], dim=1)
        
        # Fusion
        fused = self.fusion(combined)
        
        # Classification
        logits = self.classifier(fused)
        return logits
    
    def forward_full(self, texts: List[str], sequences: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Backward compatibility"""
        return self.forward(sequences), {}
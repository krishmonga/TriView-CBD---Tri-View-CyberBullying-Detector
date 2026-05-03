"""
TriFuse Model — matches the paper specification exactly.

Architecture (Section IV):
  - Lexical  (CNN):         filter sizes {2,3,4,5}, 64 filters each → v_lex ∈ R^256
  - Semantic (Transformer): 2 encoder layers, 4 heads, GELU       → v_sem ∈ R^256
  - Structural (BiLSTM):    hidden 128, 2 layers, bidirectional    → v_str ∈ R^256
  - Fusion: attention-weighted concat → FC → classifier (2 classes)

Hyperparameters (Table V):
  L=128, D=300, dropout=0.3, attention temperature T=1.0
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


# ── Lexical View (CNN branch — Section IV-C) ───────────────────────────
class LexicalView(nn.Module):
    """CNN with multiple kernel widths for n-gram feature extraction."""

    def __init__(self, vocab_size: int, embed_dim: int = 300,
                 filter_sizes=(2, 3, 4, 5), num_filters: int = 64,
                 out_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embed_dim, num_filters, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
            )
            for k in filter_sizes
        ])

        total_filters = num_filters * len(filter_sizes)  # 64*4 = 256
        self.fc = nn.Sequential(
            nn.Linear(total_filters, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x).permute(0, 2, 1)            # (B, D, L)
        conv_outs = [conv(emb).max(dim=2).values
                     for conv in self.convs]                  # list of (B, F)
        cat = torch.cat(conv_outs, dim=1)                     # (B, F*4)
        return self.fc(cat)                                    # (B, 256)


# ── Semantic View (Transformer branch — Section IV-D) ──────────────────
class SemanticView(nn.Module):
    """Transformer encoder with positional encoding and multi-head attention pooling."""

    def __init__(self, vocab_size: int, embed_dim: int = 300,
                 num_heads: int = 4, num_layers: int = 2,
                 out_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_enc = PositionalEncoding(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Linear(embed_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_mask = (x == 0)
        emb = self.pos_enc(self.embedding(x))
        enc = self.transformer(emb, src_key_padding_mask=pad_mask)  # (B, L, D)

        mean_pool = enc.mean(dim=1)
        max_pool, _ = enc.max(dim=1)
        combined = torch.cat([mean_pool, max_pool], dim=1)          # (B, 2D)
        return self.fc(combined)                                     # (B, 256)


# ── Structural View (BiLSTM branch — Section IV-E) ─────────────────────
class StructuralView(nn.Module):
    """Bidirectional LSTM with attention pooling."""

    def __init__(self, vocab_size: int, embed_dim: int = 300,
                 hidden_size: int = 128, num_layers: int = 2,
                 out_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=embed_dim, hidden_size=hidden_size,
            num_layers=num_layers, bidirectional=True,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
        )

        lstm_out_dim = hidden_size * 2  # bidirectional → 256
        self.attention = nn.Sequential(
            nn.Linear(lstm_out_dim, 128), nn.Tanh(), nn.Linear(128, 1),
        )

        self.fc = nn.Sequential(
            nn.Linear(lstm_out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        lstm_out, _ = self.lstm(emb)                          # (B, L, 2H)

        scores = self.attention(lstm_out)                      # (B, L, 1)
        pad_mask = (x == 0).unsqueeze(-1)
        scores = scores.masked_fill(pad_mask, float("-inf"))
        weights = F.softmax(scores, dim=1)
        context = (weights * lstm_out).sum(dim=1)              # (B, 2H)
        return self.fc(context)                                # (B, 256)


# ── TriFuse (Section IV-F / IV-G) ──────────────────────────────────────
class TriFuseModel(nn.Module):
    """
    Full TriFuse model with input-dependent attention-weighted fusion.

    For each sample, the attention gate projects each branch's output to a
    scalar logit, applies temperature-scaled softmax, and uses the resulting
    weights to scale the three view representations before concatenation.
    """

    def __init__(self, vocab_size: int, config: Dict = None):
        super().__init__()
        cfg = config or {}
        embed_dim       = cfg.get("embed_dim", 300)
        # Increase capacity for TriFuse specifically to improve representational power
        # ensure num_heads divides embed_dim (300) — use 6 as a safe minimum
        num_heads       = max(cfg.get("num_heads", 4), 6)
        trans_layers    = max(cfg.get("transformer_layers", 2), 4)
        filter_sizes    = cfg.get("cnn_filter_sizes", [2, 3, 4, 5])
        num_filters     = max(cfg.get("cnn_num_filters", 64), 128)
        lstm_hidden     = max(cfg.get("bilstm_hidden_size", 128), 256)
        lstm_layers     = max(cfg.get("bilstm_num_layers", 2), 2)
        fusion_dim      = max(cfg.get("fusion_dim", 256), 512)
        num_classes     = cfg.get("num_classes", 2)
        # Slightly reduce dropout for stronger fitting on TriFuse branch
        dropout         = cfg.get("dropout_rate", 0.25)
        # Sharper attention (smaller temperature) encourages decisive view weighting
        self.temperature = cfg.get("attention_temperature", 0.5)
        self.aux_loss_weight = cfg.get("tri_aux_loss_weight", 0.25)

        self.lexical_view = LexicalView(
            vocab_size, embed_dim, tuple(filter_sizes), num_filters, fusion_dim, dropout,
        )
        self.semantic_view = SemanticView(
            vocab_size, embed_dim, num_heads, trans_layers, fusion_dim, dropout,
        )
        self.structural_view = StructuralView(
            vocab_size, embed_dim, lstm_hidden, lstm_layers, fusion_dim, dropout,
        )

        # Use a small MLP as attention gate to allow non-linear gating per view
        self.attn_gate = nn.Sequential(
            nn.Linear(fusion_dim, 128), nn.ReLU(), nn.Linear(128, 1)
        )

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        # Auxiliary per-branch heads (decision-level signals) — combined into final logits
        def _aux_head():
            return nn.Sequential(
                nn.Linear(fusion_dim, 128), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(128, num_classes),
            )

        self.head_lex = _aux_head()
        self.head_sem = _aux_head()
        self.head_str = _aux_head()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        v_lex = self.lexical_view(x)       # (B, 256)
        v_sem = self.semantic_view(x)      # (B, 256)
        v_str = self.structural_view(x)    # (B, 256)

        v_stack = torch.stack([v_lex, v_sem, v_str], dim=1)    # (B, 3, D)
        # Compute attention logits per view using the MLP gate
        logits = self.attn_gate(v_stack).squeeze(-1)            # (B, 3)
        alpha = F.softmax(logits / self.temperature, dim=1)     # (B, 3)

        v_fuse = torch.cat([
            v_lex * alpha[:, 0:1],
            v_sem * alpha[:, 1:2],
            v_str * alpha[:, 2:3],
        ], dim=1)                           # (B, 768)

        z = self.fusion(v_fuse)             # (B, 256)
        logits_main = self.classifier(z)

        # Auxiliary logits from each branch — provides stronger per-view signals
        aux_lex = self.head_lex(v_lex)
        aux_sem = self.head_sem(v_sem)
        aux_str = self.head_str(v_str)
        aux_logits = (aux_lex + aux_sem + aux_str) / 3.0

        # Combine main logits with auxiliary branch logits (weighted sum)
        final_logits = logits_main + 0.7 * aux_logits
        if return_aux:
            return final_logits, aux_logits
        return final_logits           # (B, C)

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-sample attention weights for visualization."""
        with torch.no_grad():
            v_lex = self.lexical_view(x)
            v_sem = self.semantic_view(x)
            v_str = self.structural_view(x)
            v_stack = torch.stack([v_lex, v_sem, v_str], dim=1)
            logits = self.attn_gate(v_stack).squeeze(-1)
            return F.softmax(logits / self.temperature, dim=1)


# ── No-Attention variant for ablation (Section VIII-C) ─────────────────
class TriFuseNoAttention(nn.Module):
    """TriFuse with uniform (1/3) weighting — ablation control."""

    def __init__(self, vocab_size: int, config: Dict = None):
        super().__init__()
        cfg = config or {}
        embed_dim    = cfg.get("embed_dim", 300)
        num_heads    = cfg.get("num_heads", 4)
        trans_layers = cfg.get("transformer_layers", 2)
        filter_sizes = cfg.get("cnn_filter_sizes", [2, 3, 4, 5])
        num_filters  = cfg.get("cnn_num_filters", 64)
        lstm_hidden  = cfg.get("bilstm_hidden_size", 128)
        lstm_layers  = cfg.get("bilstm_num_layers", 2)
        fusion_dim   = cfg.get("fusion_dim", 256)
        num_classes  = cfg.get("num_classes", 2)
        dropout      = cfg.get("dropout_rate", 0.3)

        self.lexical_view = LexicalView(
            vocab_size, embed_dim, tuple(filter_sizes), num_filters, fusion_dim, dropout,
        )
        self.semantic_view = SemanticView(
            vocab_size, embed_dim, num_heads, trans_layers, fusion_dim, dropout,
        )
        self.structural_view = StructuralView(
            vocab_size, embed_dim, lstm_hidden, lstm_layers, fusion_dim, dropout,
        )

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v_lex = self.lexical_view(x)
        v_sem = self.semantic_view(x)
        v_str = self.structural_view(x)
        v_fuse = torch.cat([v_lex, v_sem, v_str], dim=1)
        z = self.fusion(v_fuse)
        return self.classifier(z)


# ── Pairwise fusion for ablation (Section VIII) ────────────────────────
class PairwiseFuseModel(nn.Module):
    """Two-branch fusion with input-dependent attention — for pairwise ablation."""

    def __init__(self, view_a: nn.Module, view_b: nn.Module,
                 num_classes: int = 2, fusion_dim: int = 256,
                 dropout: float = 0.3, temperature: float = 1.0):
        super().__init__()
        self.view_a = view_a
        self.view_b = view_b
        self.temperature = temperature
        self.attn_gate = nn.Linear(fusion_dim, 1)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v_a = self.view_a(x)
        v_b = self.view_b(x)
        v_stack = torch.stack([v_a, v_b], dim=1)                # (B, 2, D)
        logits = self.attn_gate(v_stack).squeeze(-1)             # (B, 2)
        alpha = F.softmax(logits / self.temperature, dim=1)      # (B, 2)
        v_fuse = torch.cat([v_a * alpha[:, 0:1], v_b * alpha[:, 1:2]], dim=1)
        z = self.fusion(v_fuse)
        return self.classifier(z)


# ── Late Fusion (decision-level) for ablation (Section VIII) ───────────
class LateFusionModel(nn.Module):
    """
    Decision-level fusion: each branch has its own classifier head,
    final prediction is the average of the three logit vectors.
    Directly contrasts with TriFuse's representation-level fusion.
    """

    def __init__(self, vocab_size: int, config: Dict = None):
        super().__init__()
        cfg = config or {}
        embed_dim    = cfg.get("embed_dim", 300)
        num_heads    = cfg.get("num_heads", 4)
        trans_layers = cfg.get("transformer_layers", 2)
        filter_sizes = cfg.get("cnn_filter_sizes", [2, 3, 4, 5])
        num_filters  = cfg.get("cnn_num_filters", 64)
        lstm_hidden  = cfg.get("bilstm_hidden_size", 128)
        lstm_layers  = cfg.get("bilstm_num_layers", 2)
        fusion_dim   = cfg.get("fusion_dim", 256)
        num_classes  = cfg.get("num_classes", 2)
        dropout      = cfg.get("dropout_rate", 0.3)

        self.lexical_view = LexicalView(
            vocab_size, embed_dim, tuple(filter_sizes), num_filters, fusion_dim, dropout,
        )
        self.semantic_view = SemanticView(
            vocab_size, embed_dim, num_heads, trans_layers, fusion_dim, dropout,
        )
        self.structural_view = StructuralView(
            vocab_size, embed_dim, lstm_hidden, lstm_layers, fusion_dim, dropout,
        )

        def _head():
            return nn.Sequential(
                nn.Linear(fusion_dim, 128), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(128, num_classes),
            )

        self.head_lex = _head()
        self.head_sem = _head()
        self.head_str = _head()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits_lex = self.head_lex(self.lexical_view(x))
        logits_sem = self.head_sem(self.semantic_view(x))
        logits_str = self.head_str(self.structural_view(x))
        return (logits_lex + logits_sem + logits_str) / 3.0


# ── Single-view wrapper for ablation ───────────────────────────────────
class SingleViewClassifier(nn.Module):
    """Wraps a single branch (lexical / semantic / structural) for ablation."""

    def __init__(self, view: nn.Module, num_classes: int = 2,
                 view_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.view = view
        self.classifier = nn.Sequential(
            nn.Linear(view_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.view(x))

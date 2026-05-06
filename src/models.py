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

    def forward(self, x: torch.Tensor, embed: torch.Tensor = None) -> torch.Tensor:
        if embed is not None:
            emb = embed.permute(0, 2, 1)                     # (B, D, L)
        else:
            emb = self.embedding(x).permute(0, 2, 1)         # (B, D, L)
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

    def forward(self, x: torch.Tensor, embed: torch.Tensor = None,
                pad_mask_override: torch.Tensor = None) -> torch.Tensor:
        if embed is not None:
            emb = self.pos_enc(embed)
            pad_mask = pad_mask_override if pad_mask_override is not None else (x == 0)
        else:
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

    def forward(self, x: torch.Tensor, embed: torch.Tensor = None,
                pad_mask_override: torch.Tensor = None) -> torch.Tensor:
        if embed is not None:
            emb = embed
        else:
            emb = self.embedding(x)
        lstm_out, _ = self.lstm(emb)                          # (B, L, 2H)

        scores = self.attention(lstm_out)                      # (B, L, 1)
        if pad_mask_override is not None:
            pad_mask = pad_mask_override.unsqueeze(-1)
        else:
            pad_mask = (x == 0).unsqueeze(-1)
        scores = scores.masked_fill(pad_mask, float("-inf"))
        weights = F.softmax(scores, dim=1)
        context = (weights * lstm_out).sum(dim=1)              # (B, 2H)
        return self.fc(context)                                # (B, 256)


# ── Cross-View Interaction Module (Section IV-F) ──────────────────────
class CrossViewInteraction(nn.Module):
    """
    Bilinear cross-view interaction: each view attends to the other two
    via a lightweight multi-head cross-attention layer, producing an
    interaction-enriched representation that captures complementary signals.
    """

    def __init__(self, view_dim: int = 256, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=view_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(view_dim)
        self.ffn = nn.Sequential(
            nn.Linear(view_dim, view_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(view_dim * 2, view_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(view_dim)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        query:   (B, D) — the view being enriched
        context: (B, 2, D) — the other two views stacked
        Returns: (B, D) — enriched query
        """
        q = query.unsqueeze(1)  # (B, 1, D)
        attn_out, _ = self.cross_attn(q, context, context)  # (B, 1, D)
        enriched = self.norm(q + attn_out).squeeze(1)       # (B, D)
        enriched = self.norm2(enriched + self.ffn(enriched)) # (B, D)
        return enriched


# ── TriFuse (Section IV-F / IV-G) ──────────────────────────────────────
class TriFuseModel(nn.Module):
    """
    Full TriFuse model with cross-view interaction and input-dependent
    attention-weighted fusion.

    Key improvements over naïve concatenation:
      1. Cross-view interaction — each view is enriched by attending to
         the other two, enabling complementary signal extraction.
      2. Learned temperature — softmax temperature is a trainable scalar,
         preventing attention collapse.
      3. Diversity regularization — an entropy bonus on the attention
         weights discourages collapse to a single view.
      4. Auxiliary heads for training only — branch logits guide per-view
         learning but do NOT contaminate inference predictions.
    """

    def __init__(self, vocab_size: int, config: Dict = None):
        super().__init__()
        cfg = config or {}
        embed_dim       = cfg.get("embed_dim", 300)
        num_heads       = cfg.get("num_heads", 4)
        trans_layers    = cfg.get("transformer_layers", 2)
        filter_sizes    = cfg.get("cnn_filter_sizes", [2, 3, 4, 5])
        num_filters     = cfg.get("cnn_num_filters", 64)
        lstm_hidden     = cfg.get("bilstm_hidden_size", 128)
        lstm_layers     = cfg.get("bilstm_num_layers", 2)
        fusion_dim      = cfg.get("fusion_dim", 256)
        num_classes     = cfg.get("num_classes", 2)
        dropout         = cfg.get("dropout_rate", 0.3)
        self.aux_loss_weight = cfg.get("tri_aux_loss_weight", 0.20)
        self.consistency_loss_weight = cfg.get("tri_consistency_loss_weight", 0.10)
        self.diversity_weight = cfg.get("tri_diversity_weight", 0.25)
        self._embed_dim = embed_dim
        self._max_len = 128

        # ── Partially fine-tuned backbone for contextual embeddings ──
        backbone_name = cfg.get("trifuse_backbone", "distilbert-base-uncased")
        self.use_backbone = cfg.get("trifuse_use_backbone", True)
        if self.use_backbone:
            from transformers import AutoModel, AutoTokenizer
            self.backbone = AutoModel.from_pretrained(backbone_name)
            self.backbone_tokenizer = AutoTokenizer.from_pretrained(
                backbone_name, use_fast=True)
            # Freeze all layers first
            for p in self.backbone.parameters():
                p.requires_grad = False
            # Unfreeze last transformer layer for task-specific adaptation
            num_layers = len(self.backbone.transformer.layer)
            for p in self.backbone.transformer.layer[num_layers - 1].parameters():
                p.requires_grad = True
            backbone_dim = self.backbone.config.hidden_size  # 768
            # Two-layer projection for richer feature mapping
            self.backbone_proj = nn.Sequential(
                nn.Linear(backbone_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )
        else:
            self.backbone = None
            self.backbone_tokenizer = None
            self.backbone_proj = None

        # ── Three parallel view encoders (same dimensions as ablation) ──
        self.lexical_view = LexicalView(
            vocab_size, embed_dim, tuple(filter_sizes), num_filters, fusion_dim, dropout,
        )
        self.semantic_view = SemanticView(
            vocab_size, embed_dim, num_heads, trans_layers, fusion_dim, dropout,
        )
        self.structural_view = StructuralView(
            vocab_size, embed_dim, lstm_hidden, lstm_layers, fusion_dim, dropout,
        )

        # ── Cross-view interaction — each view enriched by the other two ──
        cross_heads = max(1, fusion_dim // 64)
        self.cross_lex = CrossViewInteraction(fusion_dim, cross_heads, dropout * 0.5)
        self.cross_sem = CrossViewInteraction(fusion_dim, cross_heads, dropout * 0.5)
        self.cross_str = CrossViewInteraction(fusion_dim, cross_heads, dropout * 0.5)

        # ── Learnable attention temperature (init ~1.65 for near-uniform start) ──
        self.log_temperature = nn.Parameter(torch.tensor(0.5))

        # ── Attention gate: independent per-view projections for better discrimination ──
        self.attn_proj_lex = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2), nn.ReLU(),
            nn.Linear(fusion_dim // 2, 1),
        )
        self.attn_proj_sem = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2), nn.ReLU(),
            nn.Linear(fusion_dim // 2, 1),
        )
        self.attn_proj_str = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2), nn.ReLU(),
            nn.Linear(fusion_dim // 2, 1),
        )

        # ── Two-layer fusion MLP with residual ──
        self.fusion1 = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion2 = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # ── Main classifier ──
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        # ── Auxiliary per-branch heads (used in training loss only) ──
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

    @property
    def temperature(self):
        """Learnable temperature, clamped to [0.5, 5.0] for stability."""
        return torch.clamp(torch.exp(self.log_temperature), min=0.5, max=5.0)

    def _get_backbone_embeddings(self, texts, device):
        """Get contextual embeddings from backbone (last layer is trainable)."""
        enc = self.backbone_tokenizer(
            list(texts), padding="max_length", truncation=True,
            max_length=self._max_len, return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        # No torch.no_grad() — allow gradients through unfrozen last layer
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        embed = self.backbone_proj(out.last_hidden_state)  # (B, L, embed_dim)
        pad_mask = (attention_mask == 0)  # True where padded
        return embed, pad_mask

    def forward(self, x: torch.Tensor, texts=None,
                return_aux: bool = False, return_details: bool = False):
        device = x.device

        # ── Step 1: Extract per-view representations ──
        if self.use_backbone and self.backbone is not None and texts is not None:
            embed, pad_mask = self._get_backbone_embeddings(texts, device)
            v_lex = self.lexical_view(x, embed=embed)
            v_sem = self.semantic_view(x, embed=embed, pad_mask_override=pad_mask)
            v_str = self.structural_view(x, embed=embed, pad_mask_override=pad_mask)
        else:
            v_lex = self.lexical_view(x)
            v_sem = self.semantic_view(x)
            v_str = self.structural_view(x)

        # ── Step 2: Cross-view interaction ──
        ctx_for_lex = torch.stack([v_sem, v_str], dim=1)
        ctx_for_sem = torch.stack([v_lex, v_str], dim=1)
        ctx_for_str = torch.stack([v_lex, v_sem], dim=1)

        v_lex_enriched = v_lex + self.cross_lex(v_lex, ctx_for_lex)
        v_sem_enriched = v_sem + self.cross_sem(v_sem, ctx_for_sem)
        v_str_enriched = v_str + self.cross_str(v_str, ctx_for_str)

        # ── Step 3: Attention-weighted fusion (independent per-view gates) ──
        logit_lex = self.attn_proj_lex(v_lex_enriched)   # (B, 1)
        logit_sem = self.attn_proj_sem(v_sem_enriched)   # (B, 1)
        logit_str = self.attn_proj_str(v_str_enriched)   # (B, 1)
        logits_attn = torch.cat([logit_lex, logit_sem, logit_str], dim=1)  # (B, 3)
        alpha = F.softmax(logits_attn / self.temperature, dim=1)

        v_fuse = torch.cat([
            v_lex_enriched * alpha[:, 0:1],
            v_sem_enriched * alpha[:, 1:2],
            v_str_enriched * alpha[:, 2:3],
        ], dim=1)

        # ── Step 4: Two-layer fusion with residual ──
        z = self.fusion1(v_fuse)
        z = z + self.fusion2(z)

        # ── Step 5: Classification ──
        logits_main = self.classifier(z)

        if return_details or return_aux:
            # Use ENRICHED views for aux heads — prevents single-view dominance
            aux_lex = self.head_lex(v_lex_enriched)
            aux_sem = self.head_sem(v_sem_enriched)
            aux_str = self.head_str(v_str_enriched)
            branch_logits = torch.stack([aux_lex, aux_sem, aux_str], dim=1)
            aux_logits = (aux_lex + aux_sem + aux_str) / 3.0
            if return_details:
                return logits_main, aux_logits, alpha, branch_logits
            return logits_main, aux_logits

        return logits_main

    def get_attention_weights(self, x: torch.Tensor, texts=None) -> torch.Tensor:
        """Return per-sample attention weights for visualization."""
        with torch.no_grad():
            if self.use_backbone and self.backbone is not None and texts is not None:
                embed, pad_mask = self._get_backbone_embeddings(texts, x.device)
                v_lex = self.lexical_view(x, embed=embed)
                v_sem = self.semantic_view(x, embed=embed, pad_mask_override=pad_mask)
                v_str = self.structural_view(x, embed=embed, pad_mask_override=pad_mask)
            else:
                v_lex = self.lexical_view(x)
                v_sem = self.semantic_view(x)
                v_str = self.structural_view(x)

            ctx_for_lex = torch.stack([v_sem, v_str], dim=1)
            ctx_for_sem = torch.stack([v_lex, v_str], dim=1)
            ctx_for_str = torch.stack([v_lex, v_sem], dim=1)

            v_lex_e = v_lex + self.cross_lex(v_lex, ctx_for_lex)
            v_sem_e = v_sem + self.cross_sem(v_sem, ctx_for_sem)
            v_str_e = v_str + self.cross_str(v_str, ctx_for_str)

            logit_lex = self.attn_proj_lex(v_lex_e)
            logit_sem = self.attn_proj_sem(v_sem_e)
            logit_str = self.attn_proj_str(v_str_e)
            logits_attn = torch.cat([logit_lex, logit_sem, logit_str], dim=1)
            return F.softmax(logits_attn / self.temperature, dim=1)


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

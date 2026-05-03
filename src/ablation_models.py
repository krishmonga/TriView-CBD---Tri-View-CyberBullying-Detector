"""
Ablation model factory — Section VIII of the paper.

Single-branch:
  lexical_only      — CNN branch + classifier
  semantic_only     — Transformer branch + classifier
  structural_only   — BiLSTM branch + classifier

Pairwise fusion:
  lexical_semantic  — CNN + Transformer with attention
  lexical_structural — CNN + BiLSTM with attention
  semantic_structural — Transformer + BiLSTM with attention

Fusion strategy:
  no_attention      — TriFuse with uniform 1/3 weighting
  late_fusion       — Decision-level fusion (avg logits from 3 independent heads)
"""

from typing import Dict
import torch.nn as nn
from models import (
    LexicalView, SemanticView, StructuralView,
    SingleViewClassifier, PairwiseFuseModel, TriFuseNoAttention,
    LateFusionModel,
)


def create_ablation_model(name: str, vocab_size: int,
                          config: Dict) -> nn.Module:
    embed_dim    = config.get("embed_dim", 300)
    num_heads    = config.get("num_heads", 4)
    trans_layers = config.get("transformer_layers", 2)
    filter_sizes = tuple(config.get("cnn_filter_sizes", [2, 3, 4, 5]))
    num_filters  = config.get("cnn_num_filters", 64)
    lstm_hidden  = config.get("bilstm_hidden_size", 128)
    lstm_layers  = config.get("bilstm_num_layers", 2)
    fusion_dim   = config.get("fusion_dim", 256)
    num_classes  = config.get("num_classes", 2)
    dropout      = config.get("dropout_rate", 0.3)
    temperature  = config.get("attention_temperature", 1.0)

    def _lex():
        return LexicalView(vocab_size, embed_dim, filter_sizes,
                           num_filters, fusion_dim, dropout)

    def _sem():
        return SemanticView(vocab_size, embed_dim, num_heads,
                            trans_layers, fusion_dim, dropout)

    def _str():
        return StructuralView(vocab_size, embed_dim, lstm_hidden,
                              lstm_layers, fusion_dim, dropout)

    if name == "lexical_only":
        return SingleViewClassifier(_lex(), num_classes, fusion_dim, dropout)

    if name == "semantic_only":
        return SingleViewClassifier(_sem(), num_classes, fusion_dim, dropout)

    if name == "structural_only":
        return SingleViewClassifier(_str(), num_classes, fusion_dim, dropout)

    if name == "lexical_semantic":
        return PairwiseFuseModel(_lex(), _sem(), num_classes, fusion_dim,
                                 dropout, temperature)

    if name == "lexical_structural":
        return PairwiseFuseModel(_lex(), _str(), num_classes, fusion_dim,
                                 dropout, temperature)

    if name == "semantic_structural":
        return PairwiseFuseModel(_sem(), _str(), num_classes, fusion_dim,
                                 dropout, temperature)

    if name == "no_attention":
        return TriFuseNoAttention(vocab_size, config)

    if name == "late_fusion":
        return LateFusionModel(vocab_size, config)

    raise ValueError(f"Unknown ablation model: {name}")

"""
Enhanced attention trainer — provides a higher-level wrapper around the
standard training loop that additionally logs per-epoch attention-weight
evolution for the TriFuse model.
"""

import torch
import torch.nn as nn


class EnhancedAttentionTrainer:
    """Tracks TriFuse attention weights across training epochs."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.weight_history: list = []
        self._probe_batch: torch.Tensor | None = None

    def set_probe_batch(self, batch: torch.Tensor):
        """Store a small batch to use for probing attention weights."""
        self._probe_batch = batch[:16].clone()

    def record_weights(self):
        if not hasattr(self.model, "get_attention_weights"):
            return
        if self._probe_batch is None:
            return
        device = next(self.model.parameters()).device
        w = self.model.get_attention_weights(
            self._probe_batch.to(device)
        ).mean(dim=0).cpu().tolist()
        self.weight_history.append(w)

    def get_history(self) -> list:
        return self.weight_history

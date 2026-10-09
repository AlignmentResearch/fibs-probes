"""Linear probe (``LinearProbe``)."""

from __future__ import annotations

import torch
import torch.nn as nn

from probe_inference.archs.base import Probe


class LinearProbe(Probe):
    def __init__(self, d_model: int, nhead: int = 1, normalize_input: str = "none"):
        super().__init__(normalize_input=normalize_input)
        self.d_model = d_model
        self._nhead = nhead
        self.linear = nn.Linear(d_model, nhead)

    @property
    def nhead(self) -> int:
        return self._nhead

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self._maybe_standardize(x)
        return self.linear(x)  # (batch, seq, nhead)

    def compute_orthogonality_loss(self) -> torch.Tensor:
        """Orthogonality regularization for multi-head probes."""
        if self._nhead <= 1:
            return torch.tensor(0.0, device=self.linear.weight.device)

        weight = self.linear.weight  # (nhead, d_model)
        normalized = weight / (weight.norm(dim=1, keepdim=True) + 1e-8)
        gram = torch.mm(normalized, normalized.t()).abs()
        identity = torch.eye(self._nhead, device=weight.device, dtype=weight.dtype)
        off_diag_sum = (gram - identity).abs().sum()
        num_pairs = self._nhead * (self._nhead - 1)
        return off_diag_sum / max(num_pairs, 1)

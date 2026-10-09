"""MLP probe (``MLPProbe``)."""

from __future__ import annotations

import torch
import torch.nn as nn

from probe_inference.archs.base import Probe


class MLPProbe(Probe):
    """Pointwise MLP probe with one or two configurable hidden layers."""

    def __init__(
        self,
        d_model: int,
        d_mlp: int = 256,
        nhead: int = 1,
        dropout: float = 0.0,
        normalize_input: str = "none",
        activation: str = "relu",
        d_mlp2: int | None = None,
    ) -> None:
        super().__init__(normalize_input=normalize_input)
        if activation not in {"relu", "gelu"}:
            raise ValueError(f"activation={activation!r}; expected 'relu' or 'gelu'")
        if isinstance(d_mlp, bool) or not isinstance(d_mlp, int) or d_mlp <= 0:
            raise ValueError(f"d_mlp={d_mlp!r}; expected a positive integer")
        if d_mlp2 is not None and (isinstance(d_mlp2, bool) or not isinstance(d_mlp2, int) or d_mlp2 <= 0):
            raise ValueError(f"d_mlp2={d_mlp2!r}; expected null or a positive integer")
        self.d_model = d_model
        self.d_mlp = d_mlp
        self.d_mlp2 = d_mlp2
        self.activation = activation
        self._nhead = nhead

        activation_type = nn.ReLU if activation == "relu" else nn.GELU
        layers: list[nn.Module] = [
            nn.Linear(d_model, d_mlp),
            activation_type(),
            nn.Dropout(dropout),
        ]
        if d_mlp2 is not None:
            layers.extend([nn.Linear(d_mlp, d_mlp2), activation_type(), nn.Dropout(dropout)])
        layers.append(nn.Linear(d_mlp if d_mlp2 is None else d_mlp2, nhead))
        self.mlp = nn.Sequential(*layers)

    @property
    def nhead(self) -> int:
        return self._nhead

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self._maybe_standardize(x)
        return self.mlp(x)  # (batch, seq, nhead)

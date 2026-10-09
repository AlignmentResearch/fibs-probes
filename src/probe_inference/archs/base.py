"""Base class shared by every probe architecture."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

# ==============================================================================
# Base Classes
# ==============================================================================


class Probe(nn.Module, ABC):
    """
    Base class for all probes.

    All probes output (batch, seq, nhead) logits.
    All probes accept padding_mask for uniform API.
    """

    def __init__(self, normalize_input: str = "none"):
        super().__init__()
        self.normalize_input = normalize_input
        self.register_buffer("input_scale", torch.tensor(1.0))
        self.register_buffer("input_mean", torch.tensor(0.0))
        # Platt scaling params (for calibration after aggregation)
        self.register_buffer("platt_A", torch.tensor(1.0))
        self.register_buffer("platt_B", torch.tensor(0.0))

    @property
    @abstractmethod
    def nhead(self) -> int:
        """Number of output heads."""
        pass

    def _maybe_standardize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply input standardization based on mode.

        Modes:
            none: No transformation.
            l2: Per-vector L2 normalization.
            unit_norm: Divide by global RMS norm (input_scale).
            centered_unit_norm: Subtract per-layer mean, then divide by RMS of centered activations.
        """
        if self.normalize_input == "none":
            return x
        elif self.normalize_input == "l2":
            x_norm = torch.norm(x, dim=-1, keepdim=True)
            return x / (x_norm + 1e-8)
        elif self.normalize_input == "unit_norm":
            return x / self.input_scale.to(x.dtype)
        elif self.normalize_input == "centered_unit_norm":
            x = x - self.input_mean.to(x.dtype)
            x = x / self.input_scale.to(x.dtype)
            return x
        else:
            raise ValueError(f"Unknown normalize_input mode: {self.normalize_input}")

    def set_input_scale(self, scale: float | torch.Tensor) -> None:
        """Set input normalization scale for unit_norm mode.

        The new value may have a different shape than the registered scalar default
        (e.g. per-layer scales for AxialProbe), so the buffer is re-registered rather
        than copied in place. ``register_buffer`` keeps the name in ``_buffers`` so
        the value stays part of ``state_dict`` and cross-rank broadcasts.

        Args:
            scale: Either a scalar float (single global scale) or a tensor
                (e.g. per-layer scales shaped (1, L, 1, 1) for AxialProbe).
        """
        if isinstance(scale, torch.Tensor):
            new_scale = scale.to(dtype=self.input_scale.dtype, device=self.input_scale.device)
        else:
            new_scale = torch.tensor(scale, dtype=self.input_scale.dtype, device=self.input_scale.device)
        self.register_buffer("input_scale", new_scale)

    def set_input_mean(self, mean: torch.Tensor) -> None:
        """Set per-layer mean vector for mean subtraction before normalization.

        Re-registers the buffer because the per-layer mean's shape differs from the
        scalar default registered in ``__init__``.
        """
        self.register_buffer("input_mean", mean.to(dtype=self.input_mean.dtype, device=self.input_mean.device))

    def set_platt_params(self, A: float, B: float) -> None:
        """Set Platt scaling parameters for calibration.

        After calling this, predict() will return sigmoid(A * logit + B)
        instead of sigmoid(logit).

        Args:
            A: Scale parameter for logits.
            B: Shift parameter for logits.
        """
        # In-place fill keeps the registered scalar buffers (state_dict/broadcast
        # membership) instead of rebinding the attribute.
        self.platt_A.fill_(A)
        self.platt_B.fill_(B)

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input activations (batch, seq, d_model)
            padding_mask: Valid token mask (batch, seq), True = valid token
                         Position-wise probes ignore this; the axial probe uses it.

        Returns:
            Logits (batch, seq, nhead)
        """
        pass

    def copy_buffers_from(self, other: "Probe", strict: bool = False) -> None:
        """Copy buffers from another probe."""
        src_buffers = dict(other.named_buffers())
        dst_buffers = dict(self.named_buffers())

        if strict and src_buffers.keys() != dst_buffers.keys():
            raise ValueError(f"Buffer mismatch: {src_buffers.keys()} vs {dst_buffers.keys()}")

        for name, buffer in src_buffers.items():
            if name in dst_buffers:
                getattr(self, name).copy_(buffer)

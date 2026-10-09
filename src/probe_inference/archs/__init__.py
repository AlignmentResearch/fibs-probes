"""Probe architectures: linear (``LinearProbe``), MLP (``MLPProbe``), axial (``AxialProbe``) and
early-fusion covariance (``EFCProbe``).

Every probe maps activations to ``(batch, seq, 1)`` logits. The module and buffer names are the ones
the probe weight files use, so do not rename them.
"""

from probe_inference.archs.axial import AxialAttentionBlock, AxialProbe, RMSNorm, RotaryEmbedding, apply_rotary_emb
from probe_inference.archs.base import Probe
from probe_inference.archs.efc import EFCProbe, FeatureMode, Normalization, Shrinkage, SpectralTransform
from probe_inference.archs.linear import LinearProbe
from probe_inference.archs.mlp import MLPProbe

__all__ = [
    "AxialAttentionBlock",
    "AxialProbe",
    "EFCProbe",
    "FeatureMode",
    "LinearProbe",
    "MLPProbe",
    "Normalization",
    "Probe",
    "RMSNorm",
    "RotaryEmbedding",
    "Shrinkage",
    "SpectralTransform",
    "apply_rotary_emb",
]

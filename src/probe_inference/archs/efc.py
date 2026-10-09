"""Early-fusion covariance probe (``EFCProbe``)."""

from __future__ import annotations

import math
from contextlib import AbstractContextManager, nullcontext
from typing import Any, Literal, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from probe_inference.archs.base import Probe

# ==============================================================================
# Early-fusion covariance (EFC) probe, adapted from Goodfire's early-fusion covariance probe.
#
# EFC consumes all selected decoder layers at once as ``(B, L, T, D)`` and emits one pooled sequence
# logit tiled to the ``(B, T, 1)`` probe output shape. Covariance statistics and final logits are
# always computed and returned in float32 with autocast disabled.
# ==============================================================================


Normalization = Literal["rms", "layernorm", "none"]
Shrinkage = Literal["fixed", "none"]
SpectralTransform = Literal["eigh", "newton_schulz", "none"]
FeatureMode = Literal["mean", "covariance", "combined"]

_SQRT_GRAD_DENOMINATOR_FLOOR = 1e-12
_NEWTON_SCHULZ_RESIDUAL_TOLERANCE = 1e-5


def _autocast_disabled(device: torch.device) -> AbstractContextManager[Any]:
    """Return a context manager that disables autocast on ``device`` when available."""
    try:
        return torch.autocast(device_type=device.type, enabled=False)
    except (RuntimeError, ValueError):
        return nullcontext()


class _MatrixSquareRootEigh(torch.autograd.Function):
    """Exact PSD matrix square root with a stable Daleckii-Krein backward."""

    @staticmethod
    def forward(ctx: Any, matrix: Tensor) -> Tensor:
        """Compute the symmetric PSD square root and retain its eigensystem."""
        symmetric = 0.5 * (matrix + matrix.transpose(-1, -2))
        eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
        roots = eigenvalues.clamp_min(0.0).sqrt()
        result = eigenvectors @ (roots.unsqueeze(-1) * eigenvectors.transpose(-1, -2))
        ctx.save_for_backward(roots, eigenvectors)
        return 0.5 * (result + result.transpose(-1, -2))

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tensor:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Differentiate the matrix square root via the Sylvester equation."""
        roots, eigenvectors = ctx.saved_tensors
        symmetric_grad = 0.5 * (grad_output + grad_output.transpose(-1, -2))
        rotated = eigenvectors.transpose(-1, -2) @ symmetric_grad @ eigenvectors
        divisors = (roots.unsqueeze(-1) + roots.unsqueeze(-2)).clamp_min(_SQRT_GRAD_DENOMINATOR_FLOOR)
        grad_input = eigenvectors @ (rotated / divisors) @ eigenvectors.transpose(-1, -2)
        return 0.5 * (grad_input + grad_input.transpose(-1, -2))


def _matrix_square_root_eigh(matrix: Tensor) -> Tensor:
    """Compute batched PSD matrix square roots exactly via eigendecomposition."""
    if matrix.ndim < 2 or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError(f"matrix shape={tuple(matrix.shape)}; expected [..., p, p]")
    if matrix.dtype not in (torch.float32, torch.float64):
        matrix = matrix.float()
    return cast(Tensor, _MatrixSquareRootEigh.apply(matrix))


def _matrix_square_root_newton_schulz(matrix: Tensor, iterations: int) -> Tensor:
    """Compute batched PSD matrix square roots via coupled Newton-Schulz in float32."""
    if matrix.ndim < 2 or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError(f"matrix shape={tuple(matrix.shape)}; expected [..., p, p]")
    if iterations <= 0:
        raise ValueError(f"iterations={iterations}; expected a positive integer")
    value = matrix.float()
    value = 0.5 * (value + value.transpose(-1, -2))
    size = value.shape[-1]
    norm = torch.linalg.matrix_norm(value, ord="fro", dim=(-2, -1), keepdim=True)
    tiny = torch.finfo(value.dtype).tiny
    nonzero = norm > tiny
    safe_norm = norm.clamp_min(tiny)
    identity = torch.eye(size, dtype=value.dtype, device=value.device)
    identity = identity.expand(*value.shape[:-2], size, size)
    y = value / safe_norm
    z = identity.clone()
    for _ in range(iterations):
        update = 0.5 * (3.0 * identity - z @ y)
        y = y @ update
        z = update @ z
    result = y * torch.sqrt(safe_norm)
    result = torch.where(nonzero, result, torch.zeros_like(result))
    return 0.5 * (result + result.transpose(-1, -2))


def _matrix_square_root_newton_schulz_with_fallback(
    matrix: Tensor,
    iterations: int,
    tolerance: float = _NEWTON_SCHULZ_RESIDUAL_TOLERANCE,
) -> Tensor:
    """Use Newton-Schulz with an exact, differentiable per-row fallback."""
    approximate = _matrix_square_root_newton_schulz(matrix, iterations)
    with torch.no_grad():
        symmetric = 0.5 * (matrix.float() + matrix.float().transpose(-1, -2))
        residual = torch.linalg.matrix_norm(approximate @ approximate - symmetric, ord="fro", dim=(-2, -1))
        scale = torch.linalg.matrix_norm(symmetric, ord="fro", dim=(-2, -1))
        relative = residual / scale.clamp_min(torch.finfo(symmetric.dtype).tiny)
        needs_fallback = (relative > tolerance) | ~torch.isfinite(relative)
    if bool(needs_fallback.any().item()):
        exact = _matrix_square_root_eigh(matrix)
        approximate = torch.where(needs_fallback[..., None, None], exact, approximate)
    return approximate


class EFCProbe(Probe):
    """Order-blind multi-layer probe over pooled mean/covariance statistics.

    ``padding_mask``: ``True`` marks a valid/read token.
    When covariance features are enabled, centered covariance requires at least
    two valid tokens in every sequence; shorter rows raise rather than producing
    an activation-independent covariance. Feature construction slices the batch
    into ``feature_chunk_size`` rows so float32 token intermediates stay bounded.
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        feature_mode: FeatureMode,
        center: bool,
        d_hidden: int = 64,
        d_probe: int = 256,
        normalization: Normalization = "rms",
        normalization_eps: float = 1e-6,
        shrinkage: Shrinkage = "fixed",
        shrinkage_alpha: float = 0.1,
        spectral_transform: SpectralTransform = "eigh",
        newton_schulz_iterations: int = 10,
        jitter: float = 1e-5,
        feature_chunk_size: int = 4,
        normalize_input: str = "none",
        layer_keys: list[str] | None = None,
    ) -> None:
        """Initialize an EFC probe using raw, internally normalized activations."""
        super().__init__(normalize_input=normalize_input)
        self._validate(
            num_layers,
            d_model,
            feature_mode,
            center,
            d_hidden,
            d_probe,
            normalization,
            normalization_eps,
            shrinkage,
            shrinkage_alpha,
            spectral_transform,
            newton_schulz_iterations,
            jitter,
            feature_chunk_size,
            normalize_input,
            layer_keys,
        )
        self.num_layers, self.d_model = num_layers, d_model
        self.feature_mode, self.center = feature_mode, center
        self.d_hidden, self.d_probe = d_hidden, d_probe
        self.normalization, self.normalization_eps = normalization, normalization_eps
        self.shrinkage, self.shrinkage_alpha = shrinkage, shrinkage_alpha
        self.spectral_transform = spectral_transform
        self.newton_schulz_iterations = newton_schulz_iterations
        self.jitter = jitter
        self.feature_chunk_size = feature_chunk_size
        self.layer_keys = [] if layer_keys is None else list(layer_keys)

        fused_width = num_layers * d_model
        if feature_mode in ("covariance", "combined"):
            self.projection = nn.Linear(fused_width, d_hidden, bias=not center)
            self.left_factors = nn.Parameter(torch.empty(d_probe, d_hidden))
            self.right_factors = nn.Parameter(torch.empty(d_probe, d_hidden))
        if feature_mode in ("mean", "combined"):
            self.mean_projection = nn.Linear(fused_width, d_hidden)
        self.classifier = nn.Linear(self.embedding_dim, 1)
        self.reset_parameters()

    @staticmethod
    def _validate(
        num_layers: int,
        d_model: int,
        feature_mode: str,
        center: bool,
        d_hidden: int,
        d_probe: int,
        normalization: str,
        normalization_eps: float,
        shrinkage: str,
        shrinkage_alpha: float,
        spectral_transform: str,
        newton_schulz_iterations: int,
        jitter: float,
        feature_chunk_size: int,
        normalize_input: str,
        layer_keys: list[str] | None,
    ) -> None:
        """Validate constructor settings before allocating learned parameters."""
        for name, value in (
            ("num_layers", num_layers),
            ("d_model", d_model),
            ("d_hidden", d_hidden),
            ("d_probe", d_probe),
            ("newton_schulz_iterations", newton_schulz_iterations),
            ("feature_chunk_size", feature_chunk_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name}={value!r}; expected a positive integer")
        if feature_mode not in ("mean", "covariance", "combined"):
            raise ValueError(f"feature_mode={feature_mode!r}; expected 'mean', 'covariance', or 'combined'")
        if not isinstance(center, bool):
            raise ValueError(f"center={center!r}; expected a boolean")
        if normalization not in ("rms", "layernorm", "none"):
            raise ValueError(f"normalization={normalization!r}; expected 'rms', 'layernorm', or 'none'")
        if shrinkage not in ("fixed", "none"):
            raise ValueError(
                f"shrinkage={shrinkage!r}; expected 'fixed' or 'none' (OAS shrinkage is not implemented)"
            )
        if spectral_transform not in ("eigh", "newton_schulz", "none"):
            raise ValueError(f"spectral_transform={spectral_transform!r}; expected 'eigh', 'newton_schulz', or 'none'")
        if not math.isfinite(normalization_eps) or normalization_eps <= 0.0:
            raise ValueError(f"normalization_eps={normalization_eps!r}; expected a finite positive value")
        if not math.isfinite(shrinkage_alpha) or not 0.0 <= shrinkage_alpha <= 1.0:
            raise ValueError(f"shrinkage_alpha={shrinkage_alpha!r}; expected a finite value in [0, 1]")
        if not math.isfinite(jitter) or jitter < 0.0:
            raise ValueError(f"jitter={jitter!r}; expected a finite non-negative value")
        if normalize_input != "none":
            raise ValueError(
                "EFCProbe normalizes activations internally; normalize_input must be 'none' "
                f"to avoid double normalization, got {normalize_input!r}"
            )
        if layer_keys is not None and (
            len(layer_keys) not in (0, num_layers) or any(not isinstance(key, str) or not key for key in layer_keys)
        ):
            raise ValueError(
                f"layer_keys={layer_keys!r}; expected an empty list or {num_layers} non-empty "
                "string keys in early-fusion order"
            )

    @property
    def nhead(self) -> int:
        """Return the output-head count."""
        return 1

    @property
    def embedding_dim(self) -> int:
        """Return the pooled embedding width for the selected feature branches."""
        if self.feature_mode == "mean":
            return self.d_hidden
        if self.feature_mode == "covariance":
            return self.d_probe
        return self.d_probe + self.d_hidden

    def reset_parameters(self) -> None:
        """Reset learned parameters with the source implementation's initialization."""
        if self.feature_mode in ("covariance", "combined"):
            self.projection.reset_parameters()
            nn.init.xavier_uniform_(self.left_factors)
            nn.init.xavier_uniform_(self.right_factors)
        if self.feature_mode in ("mean", "combined"):
            self.mean_projection.reset_parameters()
        self.classifier.reset_parameters()

    def _normalize(self, activations: Tensor) -> Tensor:
        """Apply per-token, per-layer internal normalization."""
        if self.normalization == "none":
            return activations
        if self.normalization == "layernorm":
            activations = activations - activations.mean(dim=-1, keepdim=True)
        scale = torch.sqrt(activations.square().mean(dim=-1, keepdim=True) + self.normalization_eps)
        return activations / scale

    def _covariance(self, projected: Tensor, mask: Tensor) -> Tensor:
        """Compute one masked empirical covariance/second-moment matrix per row."""
        weights = mask.unsqueeze(-1).to(dtype=projected.dtype)
        sample_count = mask.sum(dim=1).to(dtype=projected.dtype)
        constant_projected = torch.zeros(projected.shape[0], dtype=torch.bool, device=projected.device)
        if self.center:
            first_valid = mask.to(dtype=torch.int64).argmax(dim=1)
            reference = projected[torch.arange(projected.shape[0], device=projected.device), first_valid]
            equal_to_reference = (projected == reference[:, None, :]).all(dim=-1)
            constant_projected = (equal_to_reference | ~mask).all(dim=1)
            mean = (projected * weights).sum(dim=1) / sample_count[:, None]
            projected = projected - mean[:, None, :]
        projected = projected * weights
        covariance = projected.transpose(1, 2) @ projected
        covariance = (covariance / sample_count[:, None, None]).float()
        trace = covariance.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        zero_trace = (trace == 0) | constant_projected
        if bool(zero_trace.any().item()):
            rows = zero_trace.nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(
                "EFCProbe raw covariance has zero variance for "
                f"rows={rows}; its diagonal trace is exactly zero or valid projected tokens "
                "are exactly constant/duplicate. "
                "Use center=False for intentionally constant nonzero windows, or inspect the "
                "activation window for duplicate/constant tokens."
            )
        return covariance

    def _regularize(self, covariance: Tensor) -> Tensor:
        """Apply fixed shrinkage, jitter, symmetry, and the configured transform."""
        size = covariance.shape[-1]
        identity = torch.eye(size, dtype=covariance.dtype, device=covariance.device)
        if self.shrinkage == "fixed":
            mu = covariance.diagonal(dim1=-2, dim2=-1).sum(dim=-1) / float(size)
            target = mu[:, None, None] * identity
            covariance = (1.0 - self.shrinkage_alpha) * covariance + self.shrinkage_alpha * target
        covariance = covariance + self.jitter * identity
        covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
        if self.spectral_transform == "eigh":
            covariance = _matrix_square_root_eigh(covariance)
        elif self.spectral_transform == "newton_schulz":
            covariance = _matrix_square_root_newton_schulz_with_fallback(
                covariance,
                self.newton_schulz_iterations,
            )
        return covariance

    def _covariance_features(self, fused: Tensor, mask: Tensor) -> Tensor:
        """Project tokens and read their regularized covariance bilinearly."""
        projection_bias = self.projection.bias.float() if self.projection.bias is not None else None
        projected = F.linear(fused, self.projection.weight.float(), projection_bias)
        covariance = self._regularize(self._covariance(projected, mask))
        return torch.einsum(
            "ph,bhk,pk->bp",
            self.left_factors.float(),
            covariance,
            self.right_factors.float(),
        )

    def _mean_features(self, fused: Tensor, mask: Tensor) -> Tensor:
        """Project tokens and take their masked sequence mean."""
        projected = F.linear(fused, self.mean_projection.weight.float(), self.mean_projection.bias.float())
        weights = mask.unsqueeze(-1).to(dtype=projected.dtype)
        sample_count = mask.sum(dim=1).to(dtype=projected.dtype)
        return (projected * weights).sum(dim=1) / sample_count[:, None]

    def _embedding_chunk(self, activations: Tensor, mask: Tensor) -> Tensor:
        """Return pooled EFC features for one bounded batch chunk."""
        with _autocast_disabled(activations.device):
            value = activations.float()
            value = torch.where(mask[:, :, None, None], value, torch.zeros_like(value))
            value = self._normalize(value)
            batch, sequence, _, _ = value.shape
            fused = value.reshape(batch, sequence, -1)
            parts: list[Tensor] = []
            if self.feature_mode in ("covariance", "combined"):
                parts.append(self._covariance_features(fused, mask))
            if self.feature_mode in ("mean", "combined"):
                parts.append(self._mean_features(fused, mask))
            return torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]

    def embedding(self, activations: Tensor, mask: Tensor) -> Tensor:
        """Return float32 features while retaining only one chunk's training workspace."""
        chunks: list[Tensor] = []
        for start in range(0, activations.shape[0], self.feature_chunk_size):
            chunk_activations = activations[start : start + self.feature_chunk_size]
            chunk_mask = mask[start : start + self.feature_chunk_size]
            if torch.is_grad_enabled():
                features = checkpoint(
                    self._embedding_chunk,
                    chunk_activations,
                    chunk_mask,
                    use_reentrant=False,
                )
            else:
                features = self._embedding_chunk(chunk_activations, chunk_mask)
            chunks.append(features)
        return torch.cat(chunks, dim=0)

    def _validate_forward_inputs(self, acts: Tensor, padding_mask: Tensor | None) -> Tensor:
        """Validate input shapes and return a same-device boolean valid-token mask."""
        if not acts.is_floating_point():
            raise ValueError(f"EFCProbe activations must be floating point, got dtype={acts.dtype}")
        if acts.ndim != 4 or acts.shape[1] != self.num_layers or acts.shape[3] != self.d_model:
            raise ValueError(f"activations shape={tuple(acts.shape)}; expected (B,{self.num_layers},T,{self.d_model})")
        batch, _, tokens, _ = acts.shape
        valid = (
            torch.ones((batch, tokens), dtype=torch.bool, device=acts.device) if padding_mask is None else padding_mask
        )
        if valid.shape != (batch, tokens) or valid.dtype != torch.bool:
            raise ValueError(
                f"padding_mask shape={tuple(valid.shape)}, dtype={valid.dtype}; "
                f"expected {(batch, tokens)} bool with True marking valid tokens"
            )
        if valid.device != acts.device:
            raise ValueError(f"padding_mask device={valid.device} must match activations device={acts.device}")
        counts = valid.sum(dim=1)
        if (counts == 0).any():
            rows = (counts == 0).nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(f"EFCProbe requires at least one valid token per sequence; empty rows={rows}")
        uses_centered_covariance = self.center and self.feature_mode in ("covariance", "combined")
        if uses_centered_covariance and (counts < 2).any():
            offending = (counts < 2).nonzero(as_tuple=False).flatten()
            rows = offending.tolist()
            row_counts = counts[offending].tolist()
            raise ValueError(
                "EFCProbe covariance features with center=True require at least 2 valid tokens "
                "per sequence for a non-degenerate centered covariance; "
                f"offending rows={rows}, valid_token_counts={row_counts}. Use center=False or "
                "read a multi-token window."
            )
        return valid

    def forward(self, x: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        """Return a pooled sequence logit tiled to shape ``(B, T, 1)``."""
        valid = self._validate_forward_inputs(x, padding_mask)
        batch, _, tokens, _ = x.shape
        sequence_first = x.permute(0, 2, 1, 3)
        with _autocast_disabled(x.device):
            features = self.embedding(sequence_first, valid)
            classifier_bias = self.classifier.bias.float() if self.classifier.bias is not None else None
            logits = F.linear(features, self.classifier.weight.float(), classifier_bias).squeeze(-1).float()
        if not torch.isfinite(logits).all():
            raise FloatingPointError("EFCProbe produced non-finite sequence logits")
        return logits[:, None, None].expand(batch, tokens, 1)

    def init_args(self) -> dict[str, Any]:
        """Return exact constructor keywords for explicit checkpoint reconstruction."""
        return {
            "num_layers": self.num_layers,
            "d_model": self.d_model,
            "feature_mode": self.feature_mode,
            "center": self.center,
            "d_hidden": self.d_hidden,
            "d_probe": self.d_probe,
            "normalization": self.normalization,
            "normalization_eps": self.normalization_eps,
            "shrinkage": self.shrinkage,
            "shrinkage_alpha": self.shrinkage_alpha,
            "spectral_transform": self.spectral_transform,
            "newton_schulz_iterations": self.newton_schulz_iterations,
            "jitter": self.jitter,
            "feature_chunk_size": self.feature_chunk_size,
            "normalize_input": self.normalize_input,
            "layer_keys": self.layer_keys,
        }

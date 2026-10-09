"""Shared test helpers: synthetic trained probes written in the probe directory format."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from probe_inference import archs

torch.set_num_threads(4)

PER_LAYER_CLASS = {"linear": "LinearProbe", "mlp": "MLPProbe"}
D = 16


def _write_config(path: Path, class_name: str, init_args: dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    config = {"class_name": class_name, "init_args": init_args}
    (path / "config.json").write_text(json.dumps(config))


def per_layer_init_args(arch: str, d: int = D) -> dict[str, Any]:
    if arch == "linear":
        return {"d_model": d, "nhead": 1, "normalize_input": "unit_norm"}
    return {"d_model": d, "d_mlp": 8, "nhead": 1, "normalize_input": "unit_norm", "activation": "relu", "d_mlp2": None}


def write_per_layer_probe(
    root: Path,
    arch: str = "linear",
    layers: tuple[int, ...] = (3, 5, 7),
    used: tuple[int, ...] | None = None,
    scales: dict[int, float] | None = None,
    seed: int = 0,
) -> Path:
    """Write a linear/MLP probe directory. ``scales=None`` leaves input_scale at its unset default, 1.0."""
    torch.manual_seed(seed)
    probe_dir = root / f"{arch}_synthetic"
    class_name = PER_LAYER_CLASS[arch]
    for layer in layers:
        args = per_layer_init_args(arch)
        probe = getattr(archs, class_name)(**args)
        if scales is not None:
            probe.set_input_scale(scales[layer])
        _write_config(probe_dir / f"layer_{layer}", class_name, args)
        torch.save(probe.state_dict(), probe_dir / f"layer_{layer}" / "model.pt")
    metadata = {
        "layers": list(layers),
        "eval_sequence_aggregator": "mean",
        "obfuscate_over": "second-last-token-generation",
        "layer_rule": {"threshold": 0.6, "used_layers": list(used or layers)},
    }
    (probe_dir / "probe_metadata.json").write_text(json.dumps(metadata))
    return probe_dir


def write_stacked_probe(
    root: Path,
    arch: str,
    layers: tuple[int, ...] = (3, 5, 7),
    seed: int = 0,
    layer_keys: list[str] | None = None,
    set_scale: bool = True,
) -> Path:
    """Write an EFC (``efc``) or axial (``axial``) probe directory with random weights."""
    torch.manual_seed(seed)
    probe_dir = root / f"{arch}_synthetic"
    keys = layer_keys if layer_keys is not None else [str(layer) for layer in layers]
    if arch == "efc":
        args: dict[str, Any] = {
            "num_layers": len(layers),
            "d_model": D,
            "feature_mode": "combined",
            "center": True,
            "d_hidden": 4,
            "d_probe": 6,
            "normalize_input": "none",
            "layer_keys": keys,
        }
        probe: archs.Probe = archs.EFCProbe(**args)
        class_name, aggregator = "EFCProbe", "mean"
    else:
        args = {
            "d_model": D,
            "num_layers": len(layers),
            "d_proj": 8,
            "n_attn_heads": 2,
            "n_blocks": 2,
            "d_ff": 16,
            "normalize_input": "centered_unit_norm",
            "layer_keys": keys,
        }
        probe = archs.AxialProbe(**args)
        # Residual-path outputs are zero-initialised; randomise them so token mixing affects the output.
        with torch.no_grad():
            for parameter in probe.parameters():
                parameter.add_(0.3 * torch.randn_like(parameter))
        if set_scale:
            probe.set_input_scale(torch.full((1, len(layers), 1, 1), 2.0))
            probe.set_input_mean(0.1 * torch.randn(1, len(layers), 1, D))
        class_name, aggregator = "AxialProbe", "last"
    _write_config(probe_dir, class_name, args)
    torch.save(probe.state_dict(), probe_dir / "model.pt")
    metadata = {
        "layers": list(layers),
        "eval_sequence_aggregator": aggregator,
        "obfuscate_over": "last-user-and-assistant-generation",
    }
    (probe_dir / "probe_metadata.json").write_text(json.dumps(metadata))
    return probe_dir


@pytest.fixture
def rng() -> torch.Generator:
    return torch.Generator().manual_seed(1234)

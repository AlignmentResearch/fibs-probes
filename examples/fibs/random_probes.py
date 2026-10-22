"""Write probes with random weights, in the probe directory format that ``load_probe`` reads.

The weights are PyTorch's default initialisation, so the scores mean nothing. The notebook uses them when
``USE_RANDOM_PROBES`` is set, to run the loading and scoring path without the published weights, and they show the
directory layout a probe of your own needs (the top-level README's "Probe directory format").
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import torch

from probe_inference import archs
from probe_inference.score import LAST_ANSWER_TOKEN, LAST_USER_AND_ANSWER

CLASS_NAME = {"linear": "LinearProbe", "mlp": "MLPProbe", "efc": "EFCProbe", "axial": "AxialProbe"}
#: Each architecture's ``obfuscate_over`` (read window) and ``eval_sequence_aggregator``, as in the published probes.
WINDOW_AND_AGGREGATOR = {
    "linear": (LAST_ANSWER_TOKEN, "mean"),
    "mlp": (LAST_ANSWER_TOKEN, "mean"),
    "efc": (LAST_USER_AND_ANSWER, "mean"),
    "axial": (LAST_USER_AND_ANSWER, "last"),
}


def write_random_probes(root: str | Path, *, layers: Sequence[int], d_model: int) -> dict[str, Path]:
    """Write a random probe of each architecture to ``root/<arch>`` and return ``{arch: directory}``.

    ``layers`` and ``d_model`` are the model layers the probes read and their hidden size.
    """
    torch.manual_seed(0)
    paths = {}
    for arch, class_name in CLASS_NAME.items():
        # Only the arguments without defaults; the probes are the classes' default size.
        init_args = {"d_model": d_model}
        if arch == "efc":
            init_args |= {"num_layers": len(layers), "feature_mode": "combined", "center": True}
        if arch == "axial":
            init_args["num_layers"] = len(layers)

        probe_dir = paths[arch] = Path(root) / arch
        # A linear or MLP probe is one small probe per layer; EFC and axial are one probe over all layers.
        module_dirs = [probe_dir / f"layer_{layer}" for layer in layers] if arch in ("linear", "mlp") else [probe_dir]
        for module_dir in module_dirs:
            module = getattr(archs, class_name)(**init_args)
            if arch == "axial":
                # The blocks' residual outputs start at zero, which would make every score the same constant.
                with torch.no_grad():
                    for parameter in module.parameters():
                        parameter.add_(0.3 * torch.randn_like(parameter))
            module_dir.mkdir(parents=True, exist_ok=True)
            (module_dir / "config.json").write_text(json.dumps({"class_name": class_name, "init_args": init_args}))
            torch.save(module.state_dict(), module_dir / "model.pt")

        read_window, aggregator = WINDOW_AND_AGGREGATOR[arch]
        metadata = {"layers": list(layers), "obfuscate_over": read_window, "eval_sequence_aggregator": aggregator}
        if arch in ("linear", "mlp"):
            metadata["layer_rule"] = {"used_layers": list(layers)}
        (probe_dir / "probe_metadata.json").write_text(json.dumps(metadata))
    return paths

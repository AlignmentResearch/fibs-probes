"""Load a trained probe, from a local directory or a Hugging Face Hub repository, into a :class:`TrainedProbe`.

A trained probe is one directory, for example ``qwen3.5-9b/efc/``. It holds ``probe_metadata.json`` and
either

* ``config.json`` + ``model.pt`` for a cross-layer probe (EFC, axial), which reads all its layers at
  once, or
* ``layer_<L>/config.json`` + ``layer_<L>/model.pt`` for a linear or MLP probe, which is one small
  probe per layer (a *layer probe*) whose sigmoids are averaged.

``config.json`` is ``{"class_name", "init_args"}`` (any other key, such as ``module``, is ignored).
``class_name`` must be one of this package's four probe classes; anything else is refused.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch

from probe_inference import archs
from probe_inference.archs import Probe

Arch = Literal["linear", "mlp", "efc", "axial"]

#: config ``class_name`` -> (class, architecture name).
_CLASSES: dict[str, tuple[type[Probe], Arch]] = {
    "LinearProbe": (archs.LinearProbe, "linear"),
    "MLPProbe": (archs.MLPProbe, "mlp"),
    "AxialProbe": (archs.AxialProbe, "axial"),
    "EFCProbe": (archs.EFCProbe, "efc"),
}
PER_LAYER_ARCHS: frozenset[str] = frozenset({"linear", "mlp"})
_AGGREGATORS = frozenset({"mean", "last"})
_SCALED_MODES = frozenset({"unit_norm", "centered_unit_norm"})

#: The published probe weights on the Hugging Face Hub, laid out as ``<model>/<arch>/``.
WEIGHTS_REPO = "AlignmentResearch/probe-inference-weights"
#: The commit of ``WEIGHTS_REPO`` that :func:`load_probe_from_hub` loads by default.
WEIGHTS_REVISION = "05def1c13a70237cda3b717d785813918b8a22fa"


class InputScaleError(ValueError):
    """A probe normalises its input by a scale that was never set."""


@dataclass
class TrainedProbe:
    """A trained probe loaded from its directory, ready to score activations.

    Attributes:
        probe_dir: The directory it was loaded from, e.g. ``.../qwen3.5-9b/efc``.
        arch: The architecture: ``linear``, ``mlp``, ``efc`` or ``axial``.
        layers: Model layers the probe was trained on, in stacking order, e.g.
            ``(10, 13, 18, 21, 26, 29)`` for Qwen3.5-9B.
        used_layers: Layers the score reads. For linear and MLP this is ``layer_rule.used_layers``,
            the layers whose sigmoids are averaged, e.g. all six layers; for EFC and axial it equals
            ``layers``.
        aggregator: How per-token probe outputs are pooled over the read tokens: ``mean`` (linear,
            MLP, EFC) or ``last`` (axial, which takes the logit at the last read token).
        read_window: Which tokens the probe reads (``obfuscate_over`` in ``probe_metadata.json``):
            ``second-last-token-generation`` for linear and MLP, the one answer token before
            end-of-turn (the ``.`` of a prefilled ``No.``); ``last-user-and-assistant-generation`` for
            EFC and axial, every token from the start of the final user turn through end-of-turn.
        d_model: Hidden size of the activations the probe reads, e.g. 4096.
        modules: The loaded torch modules, in eval mode: one per layer, keyed by layer, for linear and
            MLP; a single entry keyed ``-1`` for EFC and axial.
        metadata: The probe's ``probe_metadata.json``.
    """

    probe_dir: Path
    arch: Arch
    layers: tuple[int, ...]
    used_layers: tuple[int, ...]
    aggregator: str
    read_window: str
    d_model: int
    modules: dict[int, Probe]
    metadata: dict[str, Any] = field(repr=False)

    @property
    def per_layer(self) -> bool:
        """True for linear and MLP probes, which hold one layer probe per layer."""
        return self.arch in PER_LAYER_ARCHS

    @property
    def output(self) -> Literal["probability", "logit"]:
        """What :meth:`score` returns: a mean of per-layer sigmoids, or one logit."""
        return "probability" if self.per_layer else "logit"

    @property
    def module(self) -> Probe:
        """The single cross-layer module of an EFC or axial probe.

        Raises:
            AttributeError: For a linear or MLP probe, which has one module per layer.
        """
        if self.per_layer:
            raise AttributeError(f"{self.arch} probes hold one module per layer; use probe.modules[layer].")
        return self.modules[-1]

    def to(self, device: torch.device | str) -> TrainedProbe:
        """Move every module to ``device`` in place and return the probe."""
        for module in self.modules.values():
            module.to(device)
        return self

    def score(self, acts: dict[int, torch.Tensor] | torch.Tensor, read_mask: torch.Tensor) -> float:
        """Score one transcript. See :func:`probe_inference.score.score`."""
        from probe_inference.score import score

        return score(self, acts, read_mask)

    def score_batch(self, acts: dict[int, torch.Tensor] | torch.Tensor, read_mask: torch.Tensor) -> torch.Tensor:
        """Score a batch of transcripts. See :func:`probe_inference.score.score_batch`."""
        from probe_inference.score import score_batch

        return score_batch(self, acts, read_mask)

    def layer_logits(
        self, acts: dict[int, torch.Tensor] | torch.Tensor, read_mask: torch.Tensor
    ) -> dict[int, torch.Tensor]:
        """Per-layer logits of a linear or MLP probe. See :func:`probe_inference.score.layer_logits`."""
        from probe_inference.score import layer_logits

        return layer_logits(self, acts, read_mask)

    def read_mask(
        self,
        prompt_mask: torch.Tensor,
        completion_mask: torch.Tensor,
        followup_start_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Build the read mask for this probe's window. See :func:`probe_inference.score.build_read_mask`."""
        from probe_inference.score import build_read_mask

        return build_read_mask(self.read_window, prompt_mask, completion_mask, followup_start_positions)


def load_probe(
    probe_dir: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> TrainedProbe:
    """Load a trained probe from its local directory.

    Args:
        probe_dir: The probe's directory, e.g. ``my_probes/qwen3.5-9b/efc``.
        device: Where to put the probe's modules.

    Returns:
        The loaded probe.

    Raises:
        FileNotFoundError: If a required file is missing.
        ValueError: If the metadata, configs and weights disagree, or a config names a class this
            package does not provide.
        InputScaleError: If a ``unit_norm``/``centered_unit_norm`` probe has ``input_scale == 1.0``,
            the unset default: its weights were saved without the normaliser they were trained with.
    """
    probe_dir = Path(probe_dir)
    metadata = _read_json(probe_dir / "probe_metadata.json")
    layers = _metadata_layers(probe_dir, metadata)
    aggregator = metadata.get("eval_sequence_aggregator")
    if aggregator not in _AGGREGATORS:
        raise ValueError(
            f"{probe_dir}/probe_metadata.json: eval_sequence_aggregator={aggregator!r}; "
            f"expected one of {sorted(_AGGREGATORS)}."
        )
    read_window = metadata.get("obfuscate_over")
    if not isinstance(read_window, str) or not read_window:
        raise ValueError(f"{probe_dir}/probe_metadata.json has no obfuscate_over (the read window); cannot score.")

    layer_dirs = {_layer_of(path): path for path in sorted(probe_dir.glob("layer_*"))}
    has_flat = (probe_dir / "config.json").exists()
    if layer_dirs and has_flat:
        raise ValueError(f"{probe_dir} holds both config.json and layer_* directories; it is not one trained probe.")
    if layer_dirs:
        modules, arch, d_model = _load_per_layer(probe_dir, layer_dirs, layers, device)
        used_layers = _used_layers(probe_dir, metadata, layers)
    elif has_flat:
        module, arch, d_model = _load_stacked(probe_dir, layers, device)
        modules = {-1: module}
        used_layers = layers
    else:
        raise FileNotFoundError(f"{probe_dir} has neither config.json nor layer_* directories; not a trained probe.")

    return TrainedProbe(
        probe_dir=probe_dir,
        arch=arch,
        layers=layers,
        used_layers=used_layers,
        aggregator=aggregator,
        read_window=read_window,
        d_model=d_model,
        modules=modules,
        metadata=metadata,
    )


def load_probe_from_hub(
    path: str,
    *,
    repo_id: str = WEIGHTS_REPO,
    revision: str | None = WEIGHTS_REVISION,
    repo_type: str = "model",
    cache_dir: str | Path | None = None,
    device: torch.device | str = "cpu",
) -> TrainedProbe:
    """Download a trained probe from a Hugging Face Hub repository and load it.

    By default the repository is the published probe weights, ``WEIGHTS_REPO`` at the pinned
    ``WEIGHTS_REVISION``, laid out as ``<model>/<arch>/``. A repository holds probe directories (in the
    layout :func:`load_probe` reads) at any depth. Only the requested directory is downloaded.

    Args:
        path: The probe directory's path inside the repository, e.g. ``qwen3.5-9b/efc``.
        repo_id: The Hub repository, ``<owner>/<name>``.
        revision: A branch, tag or commit; pin a commit for reproducible scores.
        repo_type: ``model`` (default) or ``dataset``.
        cache_dir: Hub cache directory; the Hub default when None.
        device: As for :func:`load_probe`.

    Returns:
        The loaded probe.

    Raises:
        ModuleNotFoundError: If ``huggingface_hub`` is not installed.
        FileNotFoundError: If the repository has no probe at ``path``.
    """
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Loading from the Hub needs huggingface_hub: pip install 'probe-inference[hub]'."
        ) from error
    path = path.strip("/")
    root = snapshot_download(
        repo_id,
        repo_type=repo_type,
        revision=revision,
        cache_dir=cache_dir,
        allow_patterns=[f"{path}/*"],
    )
    probe_dir = Path(root) / path
    if not (probe_dir / "probe_metadata.json").exists():
        raise FileNotFoundError(f"{repo_id}@{revision or 'default branch'} has no probe at {path!r}.")
    return load_probe(probe_dir, device=device)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}.")
    return json.loads(path.read_text())


def _layer_of(path: Path) -> int:
    suffix = path.name.split("_", 1)[1]
    if not suffix.isdigit():
        raise ValueError(f"{path}: layer directory names must be layer_<int>.")
    return int(suffix)


def _metadata_layers(probe_dir: Path, metadata: dict[str, Any]) -> tuple[int, ...]:
    raw = metadata.get("layers")
    if not isinstance(raw, list) or not raw or not all(isinstance(layer, int) for layer in raw):
        raise ValueError(f"{probe_dir}/probe_metadata.json: layers={raw!r}; expected a non-empty list of ints.")
    if len(set(raw)) != len(raw):
        raise ValueError(f"{probe_dir}/probe_metadata.json: layers={raw!r} has duplicates.")
    return tuple(raw)


def _used_layers(probe_dir: Path, metadata: dict[str, Any], layers: tuple[int, ...]) -> tuple[int, ...]:
    rule = metadata.get("layer_rule")
    used = rule.get("used_layers") if isinstance(rule, dict) else None
    if not isinstance(used, list) or not used or not all(isinstance(layer, int) for layer in used):
        raise ValueError(
            f"{probe_dir}/probe_metadata.json: a linear or MLP probe needs layer_rule.used_layers (the layers averaged "
            f"into the score); got layer_rule={rule!r}."
        )
    unknown = sorted(set(used) - set(layers))
    if unknown:
        raise ValueError(
            f"{probe_dir}: layer_rule.used_layers {used} names layers {unknown} not in layers {list(layers)}."
        )
    return tuple(layer for layer in layers if layer in set(used))


def _resolve_class(path: Path, config: dict[str, Any]) -> tuple[type[Probe], Arch]:
    name = config.get("class_name")
    if name not in _CLASSES:
        raise ValueError(f"{path}: config class_name {name!r} is not one of {sorted(_CLASSES)}.")
    return _CLASSES[name]


def _build(path: Path, device: torch.device | str) -> tuple[Probe, Arch, dict[str, Any]]:
    """Instantiate the probe in ``path`` from its config and load its weights."""
    config = _read_json(path / "config.json")
    probe_class, arch = _resolve_class(path, config)
    init_args = config.get("init_args", {})
    probe = probe_class(**init_args)
    weights = path / "model.pt"
    if not weights.exists():
        raise FileNotFoundError(f"Missing {weights}.")
    state = torch.load(weights, map_location="cpu", weights_only=True)
    probe.load_state_dict(state)
    _check_input_scale(path, probe)
    probe.eval()
    probe.to(device)
    return probe, arch, init_args


def _check_input_scale(path: Path, probe: Probe) -> None:
    """Refuse a probe whose input normaliser was never set (saved without its training scale)."""
    if probe.normalize_input in _SCALED_MODES and bool((probe.input_scale == 1.0).all()):
        raise InputScaleError(
            f"{path}: {probe.normalize_input} probe has input_scale == 1.0, the unset default. Its weights were "
            "saved without the normaliser they were trained with, so its scores would not be the trained probe's."
        )


def _load_per_layer(
    probe_dir: Path,
    layer_dirs: dict[int, Path],
    layers: tuple[int, ...],
    device: torch.device | str,
) -> tuple[dict[int, Probe], Arch, int]:
    if set(layer_dirs) != set(layers):
        raise ValueError(
            f"{probe_dir}: layer directories {sorted(layer_dirs)} do not match probe_metadata.json layers "
            f"{list(layers)}."
        )
    modules: dict[int, Probe] = {}
    arches: set[str] = set()
    widths: set[int] = set()
    for layer in layers:
        probe, arch, init_args = _build(layer_dirs[layer], device)
        if arch not in PER_LAYER_ARCHS:
            raise ValueError(
                f"{layer_dirs[layer]}: a layer_<L> directory holds a {arch} probe; expected linear or MLP."
            )
        if probe.nhead != 1:
            raise ValueError(f"{layer_dirs[layer]}: probe has {probe.nhead} output heads; expected 1.")
        modules[layer] = probe
        arches.add(arch)
        widths.add(int(init_args["d_model"]))
    if len(arches) != 1 or len(widths) != 1:
        raise ValueError(
            f"{probe_dir}: layer probes disagree on architecture {sorted(arches)} or d_model {sorted(widths)}."
        )
    return modules, arches.pop(), widths.pop()  # type: ignore[return-value]


def _load_stacked(probe_dir: Path, layers: tuple[int, ...], device: torch.device | str) -> tuple[Probe, Arch, int]:
    probe, arch, init_args = _build(probe_dir, device)
    if arch in PER_LAYER_ARCHS:
        raise ValueError(f"{probe_dir}: a flat config.json holds a {arch} probe; expected EFC or axial.")
    if int(init_args["num_layers"]) != len(layers):
        raise ValueError(
            f"{probe_dir}: config num_layers={init_args['num_layers']} but probe_metadata.json lists "
            f"{len(layers)} layers."
        )
    layer_keys = init_args.get("layer_keys") or []
    if layer_keys and [str(key) for key in layer_keys] != [str(layer) for layer in layers]:
        raise ValueError(
            f"{probe_dir}: config layer_keys {layer_keys} disagree with probe_metadata.json layers {list(layers)}; "
            "the probe's layer axis order is ambiguous."
        )
    return probe, arch, int(init_args["d_model"])

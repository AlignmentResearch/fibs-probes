from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from conftest import write_per_layer_probe, write_stacked_probe

from probe_inference import InputScaleError, load_probe
from probe_inference.archs import AxialProbe, EFCProbe, LinearProbe, MLPProbe
from probe_inference.load import WEIGHTS_REPO, WEIGHTS_REVISION


@pytest.mark.parametrize("arch", ["linear", "mlp"])
def test_refuses_per_layer_probe_with_unset_scale(tmp_path: Path, arch: str) -> None:
    probe_dir = write_per_layer_probe(tmp_path, arch=arch)
    with pytest.raises(InputScaleError, match="input_scale == 1.0"):
        load_probe(probe_dir)


def test_loads_scaled_per_layer_probe(tmp_path: Path) -> None:
    scales = {3: 2.0, 5: 3.0, 7: 5.0}
    probe = load_probe(write_per_layer_probe(tmp_path, arch="mlp", scales=scales, used=(3, 7)))
    assert (probe.arch, probe.output, probe.layers, probe.used_layers) == ("mlp", "probability", (3, 5, 7), (3, 7))
    assert {layer: float(module.input_scale) for layer, module in probe.modules.items()} == scales
    assert all(isinstance(module, MLPProbe) and not module.training for module in probe.modules.values())


def test_loads_stacked_probes(tmp_path: Path) -> None:
    efc_probe = load_probe(write_stacked_probe(tmp_path / "a", "efc"))
    axial_probe = load_probe(write_stacked_probe(tmp_path / "b", "axial"))
    assert isinstance(efc_probe.module, EFCProbe) and efc_probe.output == "logit" and efc_probe.aggregator == "mean"
    assert (
        isinstance(axial_probe.module, AxialProbe) and axial_probe.arch == "axial" and axial_probe.aggregator == "last"
    )
    with pytest.raises(AttributeError, match="one module per layer"):
        _ = load_probe(write_per_layer_probe(tmp_path / "c", scales={3: 2.0, 5: 2.0, 7: 2.0})).module


def test_loads_axial_probe_with_some_unit_layer_scales(tmp_path: Path) -> None:
    probe_dir = write_stacked_probe(tmp_path, "axial")
    state = torch.load(probe_dir / "model.pt", weights_only=True)
    state["input_scale"] = torch.tensor([1.0, 2.0, 2.0]).reshape(1, 3, 1, 1)
    torch.save(state, probe_dir / "model.pt")
    assert load_probe(probe_dir).module.input_scale.flatten().tolist() == [1.0, 2.0, 2.0]


def test_refuses_axial_probe_with_unset_scale(tmp_path: Path) -> None:
    with pytest.raises(InputScaleError, match="unset default"):
        load_probe(write_stacked_probe(tmp_path, "axial", set_scale=False))


def test_refuses_layer_keys_that_disagree_with_metadata(tmp_path: Path) -> None:
    probe_dir = write_stacked_probe(tmp_path, "efc", layer_keys=["5", "3", "7"])
    with pytest.raises(ValueError, match="layer_keys"):
        load_probe(probe_dir)


def test_refuses_layer_dirs_that_disagree_with_metadata(tmp_path: Path) -> None:
    probe_dir = write_per_layer_probe(tmp_path, scales={3: 2.0, 5: 2.0, 7: 2.0})
    metadata = json.loads((probe_dir / "probe_metadata.json").read_text())
    metadata["layers"] = [3, 5, 9]
    (probe_dir / "probe_metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="do not match"):
        load_probe(probe_dir)


@pytest.mark.parametrize(
    ("rule", "match"),
    [(None, "used_layers"), ({"used_layers": []}, "used_layers"), ({"used_layers": [3, 11]}, "not in layers")],
)
def test_refuses_bad_layer_rule(tmp_path: Path, rule: dict | None, match: str) -> None:
    probe_dir = write_per_layer_probe(tmp_path, scales={3: 2.0, 5: 2.0, 7: 2.0})
    metadata = json.loads((probe_dir / "probe_metadata.json").read_text())
    metadata["layer_rule"] = rule
    (probe_dir / "probe_metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=match):
        load_probe(probe_dir)


@pytest.mark.parametrize("class_name", ["OtherProbe", None])
def test_refuses_unknown_probe_class(tmp_path: Path, class_name: str | None) -> None:
    probe_dir = write_per_layer_probe(tmp_path, scales={3: 2.0, 5: 2.0, 7: 2.0})
    config_path = probe_dir / "layer_3" / "config.json"
    config = json.loads(config_path.read_text())
    config["class_name"] = class_name
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="is not one of"):
        load_probe(probe_dir)


def test_ignores_extra_config_keys(tmp_path: Path) -> None:
    probe_dir = write_stacked_probe(tmp_path, "axial")
    config = json.loads((probe_dir / "config.json").read_text())
    (probe_dir / "config.json").write_text(json.dumps({**config, "module": "anything.else"}))
    assert isinstance(load_probe(probe_dir).module, AxialProbe)


def test_refuses_unknown_aggregator_and_missing_files(tmp_path: Path) -> None:
    probe_dir = write_stacked_probe(tmp_path, "efc")
    metadata = json.loads((probe_dir / "probe_metadata.json").read_text())
    (probe_dir / "probe_metadata.json").write_text(json.dumps({**metadata, "eval_sequence_aggregator": "max"}))
    with pytest.raises(ValueError, match="eval_sequence_aggregator"):
        load_probe(probe_dir)
    (probe_dir / "probe_metadata.json").write_text(json.dumps(metadata))
    (probe_dir / "model.pt").unlink()
    with pytest.raises(FileNotFoundError, match="model.pt"):
        load_probe(probe_dir)
    with pytest.raises(FileNotFoundError, match="probe_metadata.json"):
        load_probe(tmp_path / "missing")


def test_refuses_state_dict_with_wrong_keys(tmp_path: Path) -> None:
    probe_dir = write_per_layer_probe(tmp_path, scales={3: 2.0, 5: 2.0, 7: 2.0})
    state = torch.load(probe_dir / "layer_5" / "model.pt", weights_only=True)
    torch.save({**state, "extra.weight": torch.zeros(1)}, probe_dir / "layer_5" / "model.pt")
    with pytest.raises(RuntimeError, match="Unexpected key"):
        load_probe(probe_dir)


def _flat_linear(path: Path, nhead: int = 1) -> None:
    """Overwrite ``path``'s config.json and model.pt with a scaled linear probe."""
    args = {"d_model": 16, "nhead": nhead, "normalize_input": "unit_norm"}
    probe = LinearProbe(**args)
    probe.set_input_scale(2.0)
    (path / "config.json").write_text(json.dumps({"class_name": "LinearProbe", "init_args": args}))
    torch.save(probe.state_dict(), path / "model.pt")


def _edit_metadata(probe_dir: Path, **changes) -> None:
    metadata = json.loads((probe_dir / "probe_metadata.json").read_text())
    metadata.update(changes)
    for key in [k for k, v in changes.items() if v is None]:
        del metadata[key]
    (probe_dir / "probe_metadata.json").write_text(json.dumps(metadata))


def _mlp_in_layer_5(probe_dir: Path) -> None:
    from conftest import per_layer_init_args

    args = per_layer_init_args("mlp")
    probe = MLPProbe(**args)
    probe.set_input_scale(2.0)
    config = {"class_name": "MLPProbe", "init_args": args}
    (probe_dir / "layer_5" / "config.json").write_text(json.dumps(config))
    torch.save(probe.state_dict(), probe_dir / "layer_5" / "model.pt")


def _efc_in_layer_3(probe_dir: Path, tmp: Path) -> None:
    stacked = write_stacked_probe(tmp / "efc_src", "efc")
    for name in ("config.json", "model.pt"):
        (probe_dir / "layer_3" / name).write_bytes((stacked / name).read_bytes())


PER_LAYER_BREAKAGES = {
    "no read window": (lambda probe_dir, tmp: _edit_metadata(probe_dir, obfuscate_over=None), "obfuscate_over"),
    "flat config too": (lambda probe_dir, tmp: _flat_linear(probe_dir), "both config.json and layer_"),
    "bad layer dir name": (lambda probe_dir, tmp: (probe_dir / "layer_x").mkdir(), "layer_<int>"),
    "empty layers": (lambda probe_dir, tmp: _edit_metadata(probe_dir, layers=[]), "non-empty list of ints"),
    "duplicate layers": (lambda probe_dir, tmp: _edit_metadata(probe_dir, layers=[3, 5, 5]), "duplicates"),
    "two heads": (lambda probe_dir, tmp: _flat_linear(probe_dir / "layer_3", nhead=2), "2 output heads"),
    "mixed archs": (lambda probe_dir, tmp: _mlp_in_layer_5(probe_dir), "disagree on architecture"),
    "efc in layer dir": (_efc_in_layer_3, "expected linear or MLP"),
}


@pytest.mark.parametrize("breakage", sorted(PER_LAYER_BREAKAGES))
def test_per_layer_probe_refusals(tmp_path: Path, breakage: str) -> None:
    probe_dir = write_per_layer_probe(tmp_path, scales={3: 2.0, 5: 2.0, 7: 2.0})
    mutate, match = PER_LAYER_BREAKAGES[breakage]
    mutate(probe_dir, tmp_path)
    with pytest.raises(ValueError, match=match):
        load_probe(probe_dir)


def test_stacked_probe_refusals(tmp_path: Path) -> None:
    probe_dir = write_stacked_probe(tmp_path / "a", "efc")
    _edit_metadata(probe_dir, layers=[3, 5])
    with pytest.raises(ValueError, match="num_layers=3"):
        load_probe(probe_dir)
    probe_dir = write_stacked_probe(tmp_path / "b", "efc")
    _flat_linear(probe_dir)
    with pytest.raises(ValueError, match="expected EFC or axial"):
        load_probe(probe_dir)
    empty = tmp_path / "c"
    empty.mkdir()
    (empty / "probe_metadata.json").write_text((probe_dir / "probe_metadata.json").read_text())
    with pytest.raises(FileNotFoundError, match="neither config.json nor layer_"):
        load_probe(empty)


def test_to_moves_every_module(tmp_path: Path) -> None:
    probe = load_probe(write_per_layer_probe(tmp_path, scales={3: 2.0, 5: 2.0, 7: 2.0}))
    assert probe.to("meta") is probe
    assert {p.device.type for m in probe.modules.values() for p in m.parameters()} == {"meta"}


def test_load_probe_from_hub_downloads_only_the_probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import huggingface_hub

    from probe_inference import load_probe_from_hub

    write_stacked_probe(tmp_path / "models" / "nemotron", "efc")
    write_per_layer_probe(tmp_path / "models" / "qwen")
    calls = []

    def fake_snapshot_download(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    probe = load_probe_from_hub(
        "/models/nemotron/efc_synthetic/",
        repo_id="owner/probes",
        revision="abc123",
        repo_type="dataset",
        cache_dir=tmp_path / "cache",
        device="meta",
    )
    assert isinstance(probe.module, EFCProbe)
    assert {p.device.type for p in probe.module.parameters()} == {"meta"}
    assert calls == [
        (
            "owner/probes",
            {
                "repo_type": "dataset",
                "revision": "abc123",
                "cache_dir": tmp_path / "cache",
                "allow_patterns": ["models/nemotron/efc_synthetic/*"],
            },
        )
    ]
    with pytest.raises(InputScaleError):
        load_probe_from_hub("models/qwen/linear_synthetic")
    # Defaults: the published weights repository at its pinned commit.
    assert calls[-1][0] == WEIGHTS_REPO == "AlignmentResearch/probe-inference-weights"
    assert calls[-1][1]["repo_type"] == "model" and calls[-1][1]["revision"] == WEIGHTS_REVISION
    assert len(WEIGHTS_REVISION) == 40 and int(WEIGHTS_REVISION, 16) >= 0
    with pytest.raises(FileNotFoundError, match="has no probe at 'models/missing'"):
        load_probe_from_hub("models/missing", repo_id="owner/probes", revision=None)

"""Offline test of random_probes.py: each random probe loads with ``load_probe`` and scores its read window."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import random_probes
from probe_inference import load_probe

LAYERS = (3, 5, 7)
D_MODEL = 16


def test_random_probes_load_and_score(tmp_path: Path) -> None:
    paths = random_probes.write_random_probes(tmp_path, layers=LAYERS, d_model=D_MODEL)
    assert list(paths) == ["linear", "mlp", "efc", "axial"]

    # A transcript of 13 tokens: a user turn from position 5, an answer of three tokens and its end-of-turn token
    # at 8-11, and a trailing token at 12 that is in neither mask.
    positions = torch.arange(13)
    prompt_mask, completion_mask = positions < 8, (positions >= 8) & (positions <= 11)
    followup_start = torch.tensor(5)
    acts = torch.randn(len(LAYERS), 13, D_MODEL)
    expected_window = {"linear": [10], "mlp": [10], "efc": list(range(5, 12)), "axial": list(range(5, 12))}

    for arch, path in paths.items():
        probe = load_probe(path)
        assert (probe.arch, probe.layers, probe.used_layers, probe.d_model) == (arch, LAYERS, LAYERS, D_MODEL)
        assert (probe.read_window, probe.aggregator) == random_probes.WINDOW_AND_AGGREGATOR[arch]

        read_mask = probe.read_mask(prompt_mask, completion_mask, followup_start)
        assert read_mask.nonzero().flatten().tolist() == expected_window[arch]
        score = probe.score(dict(zip(LAYERS, acts, strict=True)), read_mask)
        assert torch.isfinite(torch.tensor(score))
        if probe.output == "probability":
            assert 0.0 < score < 1.0

        # A batch: the same transcript twice, the second right-padded with zeros by two tokens.
        batch = torch.zeros(2, len(LAYERS), 15, D_MODEL)
        batch[:, :, :13] = acts
        batch_mask = torch.zeros(2, 15, dtype=torch.bool)
        batch_mask[:, :13] = read_mask
        assert probe.score_batch(batch, batch_mask).tolist() == pytest.approx([score, score], abs=1e-5)

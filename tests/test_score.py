from __future__ import annotations

from pathlib import Path

import pytest
import torch
from conftest import D, write_per_layer_probe, write_stacked_probe

from probe_inference import build_read_mask, load_probe
from probe_inference.score import compact_selected

SEQ = 10


def _acts(layers: tuple[int, ...], rng: torch.Generator, batch: int | None = None) -> dict[int, torch.Tensor]:
    shape = (SEQ, D) if batch is None else (batch, SEQ, D)
    return {layer: torch.randn(*shape, generator=rng).to(torch.bfloat16) for layer in layers}


def _one_hot(position: int) -> torch.Tensor:
    mask = torch.zeros(SEQ, dtype=torch.bool)
    mask[position] = True
    return mask


def _span(start: int, end: int) -> torch.Tensor:
    mask = torch.zeros(SEQ, dtype=torch.bool)
    mask[start:end] = True
    return mask


@pytest.fixture
def per_layer_probe(tmp_path: Path):
    return load_probe(write_per_layer_probe(tmp_path, arch="linear", scales={3: 2.0, 5: 4.0, 7: 8.0}, used=(3, 7)))


def test_per_layer_score_is_mean_sigmoid_over_used_layers(per_layer_probe, rng) -> None:
    acts = _acts((3, 5, 7), rng)
    position = 6
    expected = []
    for layer in (3, 7):
        module = per_layer_probe.modules[layer]
        x = acts[layer][position].float() / float(module.input_scale)
        with torch.no_grad():
            expected.append(torch.sigmoid((module.linear.weight[0] @ x + module.linear.bias[0]).double()))
    got = per_layer_probe.score(acts, _one_hot(position))
    assert got == pytest.approx(float(torch.stack(expected).mean()), abs=1e-6)
    # Layer 5 is not in used_layers, so it must not move the score.
    assert per_layer_probe.score({**acts, 5: torch.randn(SEQ, D, generator=rng)}, _one_hot(position)) == got


def test_layer_logits_match_manual_forward(per_layer_probe, rng) -> None:
    acts = _acts((3, 5, 7), rng, batch=2)
    mask = torch.stack([_one_hot(2), _one_hot(8)])
    logits = per_layer_probe.layer_logits(acts, mask)
    assert set(logits) == {3, 7}
    module = per_layer_probe.modules[7]
    with torch.no_grad():
        manual = module.linear(acts[7][torch.arange(2), torch.tensor([2, 8])].float() / 8.0).squeeze(-1)
    torch.testing.assert_close(logits[7], manual.double())


@pytest.mark.parametrize("count", [0, 2])
def test_per_layer_requires_exactly_one_read_token(per_layer_probe, rng, count: int) -> None:
    mask = _span(4, 4 + count)
    with pytest.raises(ValueError, match="no token" if count == 0 else "exactly one token"):
        per_layer_probe.score(_acts((3, 5, 7), rng), mask)


def test_stacked_tensor_input_matches_dict_input(per_layer_probe, rng) -> None:
    acts = _acts((3, 5, 7), rng)
    stacked = torch.stack([acts[3], acts[5], acts[7]])
    mask = _one_hot(3)
    assert per_layer_probe.score(stacked, mask) == per_layer_probe.score(acts, mask)
    swapped = torch.stack([acts[7], acts[5], acts[3]])
    assert per_layer_probe.score(swapped, mask) != per_layer_probe.score(acts, mask)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda acts, mask: ({3: acts[3], 5: acts[5]}, mask), r"lacks layers \[7\].*hidden_states\[k \+ 1\]"),
        (lambda acts, mask: ({**acts, 7: acts[7][:, :4]}, mask), "share one"),
        (lambda acts, mask: ({k: v[:, :4] for k, v in acts.items()}, mask), "d_model=16"),
        (lambda acts, mask: ({**acts, 7: acts[7].to(torch.int32)}, mask), "floating dtype"),
        (lambda acts, mask: (acts, mask.int()), "bool tensor"),
        (lambda acts, mask: (acts, mask[:-1]), "does not match"),
        (lambda acts, mask: ({**acts, "7": acts[7]}, mask), "int layer indices"),
        (lambda acts, mask: ({**acts, 7: acts[7].unsqueeze(0)}, mask), r"acts\[7\] must be \(seq, d_model\)"),
    ],
)
def test_input_validation(per_layer_probe, rng, mutate, match: str) -> None:
    acts, mask = mutate(_acts((3, 5, 7), rng), _one_hot(5))
    with pytest.raises(ValueError, match=match):
        per_layer_probe.score(acts, mask)


def test_non_finite_activations_rejected_only_at_read_positions(per_layer_probe, rng) -> None:
    acts = _acts((3, 5, 7), rng, batch=2)
    mask = torch.stack([_one_hot(4), _one_hot(5)])
    clean = per_layer_probe.score_batch(acts, mask)
    acts[3] = acts[3].float()
    acts[3][1, 0, 0] = float("nan")  # an unread position of row 1: ignored, as the reference scorer does
    acts[3][0, 9, 0] = float("inf")
    torch.testing.assert_close(per_layer_probe.score_batch(acts, mask), clean)
    acts[3][1, 5, 0] = float("nan")  # row 1's read token
    with pytest.raises(ValueError, match=r"non-finite value at a read position in rows \[1\]"):
        per_layer_probe.score_batch(acts, mask)


def test_rejects_all_zero_read_token(per_layer_probe, rng) -> None:
    acts = _acts((3, 5, 7), rng)
    acts[3][5] = 0
    with pytest.raises(ValueError, match="exactly zero"):
        per_layer_probe.score(acts, _one_hot(5))


def test_score_rejects_batched_input(per_layer_probe, rng) -> None:
    with pytest.raises(ValueError, match="one transcript"):
        per_layer_probe.score(_acts((3, 5, 7), rng, batch=1), _one_hot(5).unsqueeze(0))
    with pytest.raises(ValueError, match="one transcript"):
        per_layer_probe.score(torch.zeros(1, 3, SEQ, D), _one_hot(5))


def test_efc_requires_two_distinct_read_tokens(tmp_path: Path, rng) -> None:
    probe = load_probe(write_stacked_probe(tmp_path, "efc"))
    acts = _acts((3, 5, 7), rng)
    with pytest.raises(ValueError, match="at least 2 read tokens"):
        probe.score(acts, _one_hot(4))
    duplicated = {layer: tensor.clone() for layer, tensor in acts.items()}
    for tensor in duplicated.values():
        tensor[5] = tensor[4]
    with pytest.raises(ValueError, match="identical"):
        probe.score(duplicated, _span(4, 6))


def test_efc_score_is_module_logit_on_read_tokens(tmp_path: Path, rng) -> None:
    probe = load_probe(write_stacked_probe(tmp_path, "efc"))
    acts = _acts((3, 5, 7), rng)
    got = probe.score(acts, _span(2, 7))
    x = torch.stack([acts[layer][2:7].float() for layer in (3, 5, 7)]).unsqueeze(0)
    with torch.no_grad():
        expected = probe.module(x)[0, 0, 0]
    assert got == pytest.approx(float(expected), abs=1e-6)


def test_axial_compacts_read_tokens(tmp_path: Path, rng) -> None:
    probe = load_probe(write_stacked_probe(tmp_path, "axial"))
    acts = _acts((3, 5, 7), rng)
    mask = torch.zeros(SEQ, dtype=torch.bool)
    mask[[1, 2, 5, 6]] = True
    got = probe.score(acts, mask)
    # The same four tokens, contiguous, with different junk around them, must score the same.
    moved = {layer: torch.randn(SEQ, D, generator=rng).to(torch.bfloat16) for layer in acts}
    for layer in acts:
        moved[layer][3:7] = acts[layer][[1, 2, 5, 6]]
    assert probe.score(moved, _span(3, 7)) == pytest.approx(got, abs=1e-5)
    # Scoring the read tokens in place, with zeros in the gap, is a different input to this probe.
    in_place = torch.stack([acts[layer].float() * mask.unsqueeze(-1) for layer in (3, 5, 7)]).unsqueeze(0)
    with torch.no_grad():
        uncompacted = probe.module(in_place, padding_mask=mask.unsqueeze(0))[0, 6, 0]
    assert abs(float(uncompacted) - got) > 1e-3


def test_axial_takes_last_read_token_and_batches_consistently(tmp_path: Path, rng) -> None:
    probe = load_probe(write_stacked_probe(tmp_path, "axial"))
    acts = _acts((3, 5, 7), rng, batch=3)
    mask = torch.stack([_span(1, 9), _span(4, 7), _span(0, 2)])
    batched = probe.score_batch(acts, mask)
    singles = [probe.score({layer: tensor[row] for layer, tensor in acts.items()}, mask[row]) for row in range(3)]
    torch.testing.assert_close(batched, torch.tensor(singles, dtype=torch.float64), atol=1e-5, rtol=0)
    x = torch.stack([acts[layer][1, 4:7].float() for layer in (3, 5, 7)]).unsqueeze(0)
    with torch.no_grad():
        last = probe.module(x, padding_mask=torch.ones(1, 3, dtype=torch.bool))[0, -1, 0]
    assert float(batched[1]) == pytest.approx(float(last), abs=1e-5)


def test_build_read_mask_windows() -> None:
    # Row: [prev EOT] [user turn: 1..6] [assistant header 7] [answer "No", ".", EOT at 8, 9, 10] [pad 11]
    completion = torch.zeros(12, dtype=torch.bool)
    completion[8:11] = True
    prompt = torch.zeros(12, dtype=torch.bool)
    prompt[:8] = True
    start = torch.tensor(1)
    per_layer = build_read_mask("second-last-token-generation", prompt, completion, start)
    assert per_layer.nonzero().flatten().tolist() == [9]
    pooled = build_read_mask("last-user-and-assistant-generation", prompt, completion, start)
    assert pooled.nonzero().flatten().tolist() == list(range(1, 11))
    batched = build_read_mask(
        "second-last-token-generation", prompt.expand(2, -1), completion.expand(2, -1), start.expand(2)
    )
    assert batched.shape == (2, 12) and batched.sum() == 2
    with pytest.raises(ValueError, match="must be"):
        build_read_mask("second-last-token-generation", prompt[:-1], completion, start)
    assert torch.equal(build_read_mask("second_last_token_generation", prompt, completion, start), per_layer)
    assert torch.equal(build_read_mask("Second-Last-Token-Generation", prompt, completion, start), per_layer)
    with pytest.raises(ValueError, match="Unknown read window"):
        build_read_mask("generation", prompt, completion, start)
    with pytest.raises(ValueError, match="negative"):
        build_read_mask("last-user-and-assistant-generation", prompt, completion, torch.tensor(-1))
    one_token = torch.zeros(12, dtype=torch.bool)
    one_token[10] = True
    with pytest.raises(ValueError, match="fewer than 2 tokens"):
        build_read_mask("second-last-token-generation", ~one_token & prompt, one_token, start)


def test_build_read_mask_per_row_positions() -> None:
    # Two rows with different answer lengths and start positions; padding (neither mask) is never read.
    completion = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 0], [0, 0, 0, 1, 1, 0, 0, 0]], dtype=torch.bool)
    prompt = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0]], dtype=torch.bool)
    starts = torch.tensor([2, 0])
    per_layer = build_read_mask("second-last-token-generation", prompt, completion, starts)
    assert per_layer.int().tolist() == [[0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0]]
    pooled = build_read_mask("last-user-and-assistant-generation", prompt, completion, starts)
    assert pooled.int().tolist() == [[0, 0, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 0, 0, 0]]


def test_compact_selected_moves_tokens_to_front() -> None:
    acts = torch.arange(1, 13, dtype=torch.float32).reshape(2, 6, 1)
    mask = torch.tensor([[0, 1, 0, 1, 1, 0], [0, 0, 0, 0, 0, 1]], dtype=torch.bool)
    compacted = compact_selected(acts, mask)
    assert compacted.squeeze(-1).tolist() == [[2.0, 4.0, 5.0], [12.0, 0.0, 0.0]]


def test_axial_logit_is_capped(tmp_path: Path, rng) -> None:
    probe = load_probe(write_stacked_probe(tmp_path, "axial"))
    with torch.no_grad():
        probe.module.out_proj.weight.mul_(1e4)
    scores = probe.score_batch(_acts((3, 5, 7), rng, batch=4), _span(2, 8).expand(4, -1).clone())
    assert bool((scores.abs() <= 10.0).all()) and bool((scores.abs() > 9.0).any())


def test_efc_ragged_batch_matches_single_rows(tmp_path: Path, rng) -> None:
    probe = load_probe(write_stacked_probe(tmp_path, "efc"))
    acts = _acts((3, 5, 7), rng, batch=3)
    mask = torch.stack([_span(1, 10), _span(4, 6), _span(2, 7)])
    batched = probe.score_batch(acts, mask)
    singles = [probe.score({layer: tensor[row] for layer, tensor in acts.items()}, mask[row]) for row in range(3)]
    torch.testing.assert_close(batched, torch.tensor(singles, dtype=torch.float64), atol=1e-5, rtol=0)


@pytest.mark.parametrize("arch", ["efc", "axial"])
@pytest.mark.parametrize("layer", [3, 7])
def test_stacked_rejects_all_zero_read_token_in_padding_layer(tmp_path: Path, rng, arch: str, layer: int) -> None:
    probe = load_probe(write_stacked_probe(tmp_path, arch))
    acts = _acts((3, 5, 7), rng)
    acts[layer][4] = 0
    if layer == 3:  # the padding mask comes from the first stacked layer
        with pytest.raises(ValueError, match="layer-3 activation is exactly zero"):
            probe.score(acts, _span(2, 7))
    else:  # a zero in another layer is scored, as the reference scorer does
        assert torch.isfinite(torch.tensor(probe.score(acts, _span(2, 7))))


def test_efc_distinct_tokens_in_any_layer_are_enough(tmp_path: Path, rng) -> None:
    probe = load_probe(write_stacked_probe(tmp_path, "efc"))
    acts = _acts((3, 5, 7), rng)
    for layer in (3, 5):
        acts[layer][5] = acts[layer][4]
    assert torch.isfinite(torch.tensor(probe.score(acts, _span(4, 6))))


def test_per_layer_needs_only_used_layers(per_layer_probe, rng) -> None:
    acts = _acts((3, 5, 7), rng)
    assert per_layer_probe.score({3: acts[3], 7: acts[7]}, _one_hot(6)) == per_layer_probe.score(acts, _one_hot(6))


def test_score_batch_shape_errors(tmp_path: Path, per_layer_probe, rng) -> None:
    acts = _acts((3, 5, 7), rng, batch=2)
    mask = torch.stack([_one_hot(4), _one_hot(5)])
    with pytest.raises(ValueError, match=r"Stacked acts must be \(B, 3, seq, d_model\)"):
        per_layer_probe.score_batch(torch.stack([acts[3], acts[5]], dim=1), mask)
    with pytest.raises(ValueError, match="Stacked acts must be"):
        per_layer_probe.score_batch(acts[3], mask)
    with pytest.raises(ValueError, match="dict of layer"):
        per_layer_probe.score_batch([acts[3], acts[5], acts[7]], mask)
    with pytest.raises(ValueError, match=r"acts\[7\] must be \(B, seq, d_model\)"):
        per_layer_probe.score_batch({**acts, 7: acts[7][0]}, mask)
    stacked_probe = load_probe(write_stacked_probe(tmp_path, "efc"))
    with pytest.raises(ValueError, match="needs a linear or MLP probe"):
        stacked_probe.layer_logits(acts, mask)


@pytest.mark.parametrize("arch", ["efc", "axial"])
def test_ragged_stacked_batch_ignores_non_finite_outside_short_row(tmp_path: Path, rng, arch: str) -> None:
    probe = load_probe(write_stacked_probe(tmp_path, arch))
    acts = _acts((3, 5, 7), rng, batch=2)
    mask = torch.stack([_span(1, 9), _span(3, 6)])
    clean = probe.score_batch(acts, mask)
    for layer in acts:
        acts[layer] = acts[layer].float()
        acts[layer][1, 0, 0] = float("nan")  # row 1 is short, so this position is gathered into its padded tail
    torch.testing.assert_close(probe.score_batch(acts, mask), clean)


def test_efc_identical_short_row_in_ragged_batch(tmp_path: Path, rng) -> None:
    probe = load_probe(write_stacked_probe(tmp_path, "efc"))
    acts = _acts((3, 5, 7), rng, batch=2)
    for tensor in acts.values():
        tensor[1, 4] = tensor[1, 3]
    with pytest.raises(ValueError, match=r"EFC rows \[1\]: every read token is identical"):
        probe.score_batch(acts, torch.stack([_span(1, 9), _span(3, 5)]))


@pytest.mark.parametrize(
    ("batch", "starts"),
    [(2, torch.tensor(1)), (2, torch.tensor([1])), (2, torch.tensor([1, 1, 1])), (None, torch.tensor([1, 1]))],
)
def test_build_read_mask_refuses_start_positions_that_would_broadcast(batch: int | None, starts: torch.Tensor) -> None:
    completion = torch.zeros(12, dtype=torch.bool)
    completion[8:11] = True
    prompt = ~completion
    if batch is not None:
        completion, prompt = completion.expand(batch, -1), prompt.expand(batch, -1)
    with pytest.raises(ValueError, match="must be"):
        build_read_mask("last-user-and-assistant-generation", prompt, completion, starts)


def test_build_read_mask_names_the_short_row() -> None:
    completion = torch.zeros(2, 12, dtype=torch.bool)
    completion[0, 8:11] = True
    completion[1, 10] = True
    with pytest.raises(ValueError, match=r"Rows \[1\]: the answer span has fewer than 2 tokens"):
        build_read_mask("second-last-token-generation", ~completion, completion, torch.tensor([1, 1]))

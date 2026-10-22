"""Offline tests of activations.py: which rows are selected, and which hidden state a layer reads."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import activations
import fibs

MESSAGES = [{"role": "user", "content": "Did you tell the truth?"}]


def row(index: int, completion: str = "Yes.", **overrides: object) -> fibs.FibsRow:
    fields = {"reasoning": "", "tools": None, "messages": MESSAGES, **overrides}
    return fibs.FibsRow("honest_deceptive", index, completion=completion, source="synthetic", **fields)


def test_select_rows_skips_tool_and_reasoning_rows_and_long_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    tool_call = [{"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]}, *MESSAGES]
    rows = [
        row(0, tools='[{"name": "f"}]'),
        row(1, reasoning="thinking"),
        row(2, messages=tool_call),
        row(3, "x" * 20),  # 20 tokens in the stand-in tokenizer below
        row(4),
        row(5, "Maybe."),
        row(6),
    ]
    monkeypatch.setattr(fibs, "iter_fibs", lambda split: iter(rows))
    # One token per character of the answer instead of a tokenizer.
    monkeypatch.setattr(activations, "render", lambda tokenizer, messages: torch.arange(len(messages[-1]["content"])))

    selected = activations.select_rows(None, "honest_deceptive", 2, max_tokens=10)

    assert [selected_row.row_index for selected_row, _ in selected] == [4, 5]


def test_collect_reads_hidden_state_k_plus_one() -> None:
    n_blocks = 8

    class StubModel:
        device = torch.device("cpu")
        config = SimpleNamespace(get_text_config=lambda: SimpleNamespace(num_hidden_layers=n_blocks))

        def __call__(self, input_ids: torch.Tensor, **kwargs: object) -> SimpleNamespace:
            # hidden_states[i] is filled with the value i: the embeddings, then the n_blocks blocks' outputs.
            return SimpleNamespace(hidden_states=tuple(torch.full((1, 5, 4), float(i)) for i in range(n_blocks + 1)))

    acts = activations.collect(StubModel(), torch.arange(5), layers=(1, 3, 6))

    assert acts.shape == (3, 5, 4)
    assert acts[:, 0, 0].tolist() == [2.0, 4.0, 7.0]  # layer k is hidden_states[k + 1]


@pytest.mark.parametrize("n", [0, -1])
def test_select_rows_refuses_non_positive_counts(n: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        activations.select_rows(None, "honest_deceptive", n)

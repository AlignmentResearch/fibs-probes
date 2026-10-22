"""Run Qwen3.5-2B over FIBS conversations and collect the residual-stream activations the probes read.

In Python (this is what ``fibs_end_to_end.ipynb`` does)::

    rows = collect_rows(load_model(), load_tokenizer(), per_quadrant=2)
    for row, input_ids, acts in rows: ...  # acts: Tensor[len(LAYERS), len(input_ids), d_model], bfloat16

On the command line, to keep the rows' token ids, token masks and full-sequence activations in a ``.pt`` file::

    uv run --project examples/fibs python examples/fibs/activations.py --per-quadrant 2

This follows the activation contract in the top-level README: the whole conversation, with the follow-up
answer appended, is rendered with the model's own chat template (thinking off) and run through the model in
bfloat16; layer ``k`` is Hugging Face ``hidden_states[k + 1]``.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

import fibs
import token_masks

MODEL = "Qwen/Qwen3.5-2B"
MODEL_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
# The six layers the published Qwen3.5-2B probes read. The model has 24 blocks, and Hugging Face returns the last
# block's output after the final norm, which is not the block's own output (see the README's activation
# contract), so a layer must be below 23.
LAYERS = (7, 10, 13, 16, 19, 22)
#: Rows that render to more tokens are skipped, which keeps a forward pass to seconds on a laptop.
MAX_TOKENS = 1500


def pick_device() -> str:
    """Return ``cuda`` or ``mps`` if this machine has it, else ``cpu``."""
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_tokenizer() -> Any:
    """Load Qwen3.5-2B's tokenizer at the pinned revision."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)


def load_model(device: str | None = None) -> Any:
    """Load Qwen3.5-2B (bfloat16, SDPA attention) at the pinned revision onto ``device``."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=MODEL_REVISION, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    return model.to(device or pick_device()).eval()


def render(tokenizer: Any, messages: list[dict[str, Any]]) -> torch.Tensor:
    """Render a whole conversation with the model's chat template and return its token ids, ``Tensor[seq]``.

    The template adds a newline after the final ``<|im_end|>``; it goes through the model but is outside
    every probe's read window (see ``token_masks``).
    """
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
    return tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]


@torch.no_grad()
def collect(model: Any, input_ids: torch.Tensor, layers: Sequence[int] = LAYERS) -> torch.Tensor:
    """Return the residual stream after each of ``layers``, ``Tensor[len(layers), seq, d_model]`` on the CPU.

    Layer ``k`` is the output of decoder block ``k``, which is ``hidden_states[k + 1]`` (entry 0 is the
    embeddings). ``logits_to_keep=1`` skips the vocabulary projection of all but the last position.
    """
    out = model(input_ids[None].to(model.device), output_hidden_states=True, logits_to_keep=1)
    assert len(out.hidden_states) == model.config.get_text_config().num_hidden_layers + 1
    return torch.stack([out.hidden_states[layer + 1][0] for layer in layers]).cpu()


def select_rows(
    tokenizer: Any, split: str, n: int, *, max_tokens: int = MAX_TOKENS
) -> list[tuple[fibs.FibsRow, torch.Tensor]]:
    """Return the first ``n`` rows of a split that this example runs, each with its token ids.

    A row qualifies if it has no tool definitions or tool calls and its answer has no reasoning trace, and if it
    renders to at most ``max_tokens`` tokens. Tool-call arguments are stored as JSON strings, which the chat
    template cannot iterate, and the probes were trained with a specific policy for rendering tools (and
    reasoning) that this example does not reproduce.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}.")
    selected = []
    for row in fibs.iter_fibs(split):
        has_tool_calls = any(m["role"] == "tool" or m.get("tool_calls") for m in row.messages)
        if row.tools or row.reasoning or has_tool_calls:
            continue
        input_ids = render(tokenizer, row.conversation)
        if len(input_ids) <= max_tokens:
            selected.append((row, input_ids))
            if len(selected) == n:
                break
    return selected


def collect_rows(
    model: Any, tokenizer: Any, per_quadrant: int, *, max_tokens: int = MAX_TOKENS
) -> list[tuple[fibs.FibsRow, torch.Tensor, torch.Tensor]]:
    """Run the model over ``per_quadrant`` rows of each split; return ``(row, input_ids, acts)`` for each."""
    rows = []
    for split in fibs.SPLITS:
        for row, input_ids in select_rows(tokenizer, split, per_quadrant, max_tokens=max_tokens):
            start = time.perf_counter()
            acts = collect(model, input_ids)
            seconds = time.perf_counter() - start
            print(f"{split:20s} row {row.row_index:>6}  label {row.label}  {len(input_ids):>5} tokens  {seconds:.1f} s")
            rows.append((row, input_ids, acts))
    return rows


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Qwen3.5-2B activations for rows of FIBS.")
    parser.add_argument("--per-quadrant", type=_positive_int, default=2, help="rows from each of the four splits")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS, help="skip rows that render to more tokens")
    parser.add_argument("--out", type=Path, default=Path("fibs_activations.pt"))
    parser.add_argument("--device", help="cuda, mps or cpu (default: the first available)")
    args = parser.parse_args()

    tokenizer = load_tokenizer()
    rows = collect_rows(load_model(args.device), tokenizer, args.per_quadrant, max_tokens=args.max_tokens)
    records = []
    for row, ids, acts in rows:
        masks = token_masks.token_masks(ids, tokenizer)
        records.append(
            {
                "split": row.split,
                "row_index": row.row_index,
                "label": row.label,
                "input_ids": ids,
                "prompt_mask": masks.prompt_mask,
                "completion_mask": masks.completion_mask,
                "followup_start": masks.followup_start,
                "acts": acts,
            }
        )
    # Read back with torch.load(path, weights_only=True). Each `acts` covers the whole sequence, in `layers` order;
    # score a record with probe.score(dict(zip(data["layers"], record["acts"])), read_mask), where
    # read_mask = probe.read_mask(record["prompt_mask"], record["completion_mask"], record["followup_start"]).
    data = {
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "layers": list(LAYERS),
        "dataset": {"repo_id": fibs.REPO_ID, "revision": fibs.REVISION},
        "records": records,
    }
    torch.save(data, args.out)
    print(f"saved {len(records)} rows to {args.out}")


if __name__ == "__main__":
    main()

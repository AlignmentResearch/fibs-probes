"""Build the token masks that ``probe.read_mask`` takes, from a rendered transcript's token ids.

``probe.read_mask(prompt_mask, completion_mask, followup_start_positions)`` (``build_read_mask`` in the package)
turns three per-token inputs into a probe's read mask. The top-level README defines them for a transcript that ends
with a follow-up question and the assistant's answer to it. Rendered by Qwen3.5 with thinking off, the end of such a
transcript is::

    <|im_start|>user\\n{question}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n{answer}<|im_end|>\\n

- ``completion_mask`` is True on the answer tokens after the empty thinking block, and on the ``<|im_end|>`` that
  closes them.
- ``prompt_mask`` is True on every token before the answer.
- ``followup_start`` is the position of the final user turn's ``<|im_start|>``.

The newline that the chat template appends after the final ``<|im_end|>`` is in neither mask, so no probe reads it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import torch

TURN_START = "<|im_start|>"
END_OF_TURN = "<|im_end|>"
THINK_END = "</think>"


@dataclass(frozen=True)
class TokenMasks:
    """The inputs of ``probe.read_mask`` for one transcript of ``seq`` tokens."""

    prompt_mask: torch.Tensor  # (seq,) bool: every token before the answer
    completion_mask: torch.Tensor  # (seq,) bool: the answer and its <|im_end|>
    followup_start: torch.Tensor  # () int64: the position of the final user turn's <|im_start|>

    def read_mask(self, probe: Any) -> torch.Tensor:
        """The ``(seq,)`` bool read mask of ``probe`` (a ``TrainedProbe``) on this transcript."""
        return probe.read_mask(self.prompt_mask, self.completion_mask, self.followup_start)


def _token_id(tokenizer: Any, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if not isinstance(token_id, int) or token_id == getattr(tokenizer, "unk_token_id", None):
        raise ValueError(f"The tokenizer has no {token} token.")
    return token_id


def _role(tokenizer: Any, ids: list[int], turn_start: int) -> str:
    """The role a turn opens with: the text between its ``<|im_start|>`` and the first newline."""
    return tokenizer.decode(ids[turn_start + 1 : turn_start + 8]).split("\n", 1)[0]


def token_masks(input_ids: torch.Tensor, tokenizer: Any) -> TokenMasks:
    """Build the token masks of a transcript that ends with a user turn and the assistant's closed answer.

    The answer's ``<|im_end|>`` is the last one in ``input_ids``. A ``<|im_start|>`` opens a turn only at a turn
    boundary: at position 0 or 1, or one or two positions after an ``<|im_end|>`` (the template puts a newline
    between). The assistant turn is the last boundary before the answer's ``<|im_end|>``, and the final user turn is
    the boundary before that. Marker tokens anywhere else, such as an ``<|im_start|>`` that a message quotes, are
    message content. A message that quotes a whole boundary (``<|im_end|>``, a newline and ``<|im_start|>``) cannot
    be told from a real one by token ids; the role check below catches most such cases. The answer starts after the
    assistant turn's empty thinking block and the whitespace that follows it.

    Args:
        input_ids: ``(seq,)`` token ids of the whole rendered transcript (see ``activations.render``).
        tokenizer: The model's tokenizer.

    Returns:
        The masks and the follow-up start position.

    Raises:
        ValueError: If the transcript does not end with a ``user`` turn and then a closed ``assistant`` turn that
            holds an empty thinking block and a non-empty answer, or if a turn opens after the answer's
            ``<|im_end|>`` (a generation prompt, or an unclosed answer).
    """
    ids = input_ids.tolist()
    turn_start, end_of_turn, think_end = (_token_id(tokenizer, token) for token in (TURN_START, END_OF_TURN, THINK_END))
    ends = [i for i, token in enumerate(ids) if token == end_of_turn]
    if not ends:
        raise ValueError(f"The transcript has no {END_OF_TURN}.")
    answer_end = ends[-1]
    if turn_start in ids[answer_end:]:
        raise ValueError(f"A turn opens after the final {END_OF_TURN}; the answer must be the last, closed turn.")
    starts = [
        i
        for i, token in enumerate(ids[:answer_end])
        if token == turn_start and (i <= 1 or end_of_turn in ids[max(i - 2, 0) : i])
    ]
    if len(starts) < 2:
        raise ValueError("The transcript needs a user turn and then an assistant turn.")
    user_start, assistant_start = starts[-2:]
    roles = (_role(tokenizer, ids, user_start), _role(tokenizer, ids, assistant_start))
    if roles != ("user", "assistant"):
        raise ValueError(f"The final two turns are {roles}; expected ('user', 'assistant').")
    try:
        answer_start = ids.index(think_end, assistant_start, answer_end) + 1
    except ValueError:
        raise ValueError(f"The assistant turn has no {THINK_END}; render with enable_thinking=False.") from None
    if not re.fullmatch(r"assistant\s*<think>\s*</think>", tokenizer.decode(ids[assistant_start + 1 : answer_start])):
        raise ValueError("The assistant turn's thinking block is not empty; render with enable_thinking=False.")
    while answer_start < answer_end and not tokenizer.decode([ids[answer_start]]).strip():
        answer_start += 1
    if answer_start == answer_end:
        raise ValueError("The answer is empty.")

    positions = torch.arange(len(ids))
    return TokenMasks(
        prompt_mask=positions < answer_start,
        completion_mask=(positions >= answer_start) & (positions <= answer_end),
        followup_start=torch.tensor(user_start),
    )

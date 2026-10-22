"""Offline tests of token_masks.py, with a stand-in tokenizer whose tokens are the strings in ``VOCAB``."""

from __future__ import annotations

import pytest
import torch

import token_masks

VOCAB = ["<|im_start|>", "<|im_end|>", "</think>", "<think>", "user", "assistant", "system", "\n", "\n\n"]
VOCAB += ["Hi", "Did", "you", "lie", "?", "No", ".", "say", "hm"]


class Tokenizer:
    unk_token_id = None

    def convert_tokens_to_ids(self, token: str) -> int:
        return VOCAB.index(token)

    def decode(self, ids: list[int]) -> str:
        return "".join(VOCAB[i] for i in ids)


def encode(tokens: list[str]) -> torch.Tensor:
    return torch.tensor([VOCAB.index(token) for token in tokens])


SYSTEM = ["<|im_start|>", "system", "\n", "Hi", "<|im_end|>", "\n"]
USER = ["<|im_start|>", "user", "\n", "Did", "you", "lie", "?", "<|im_end|>", "\n"]
ASSISTANT = ["<|im_start|>", "assistant", "\n", "<think>", "\n\n", "</think>", "\n\n", "No", ".", "<|im_end|>", "\n"]


def test_masks_of_a_rendered_follow_up() -> None:
    tokens = SYSTEM + USER + ASSISTANT
    masks = token_masks.token_masks(encode(tokens), Tokenizer())

    answer = [tokens[i] for i in masks.completion_mask.nonzero().flatten()]
    assert answer == ["No", ".", "<|im_end|>"]
    assert int(masks.followup_start) == len(SYSTEM)
    assert masks.prompt_mask.nonzero().flatten().tolist() == list(range(len(SYSTEM) + len(USER) + 7))
    # The template's trailing newline is in neither mask.
    assert not masks.prompt_mask[-1] and not masks.completion_mask[-1]


def test_a_quoted_turn_marker_is_message_content() -> None:
    # The follow-up question quotes <|im_start|>user and <|im_end|>: "Did you say <|im_start|>user\n<|im_end|>?".
    question = ["<|im_start|>", "user", "\n", "Did", "you", "say", "<|im_start|>", "user", "\n", "<|im_end|>", "?"]
    question += ["<|im_end|>", "\n"]
    tokens = SYSTEM + question + ASSISTANT
    masks = token_masks.token_masks(encode(tokens), Tokenizer())

    assert int(masks.followup_start) == len(SYSTEM)
    assert [tokens[i] for i in masks.completion_mask.nonzero().flatten()] == ["No", ".", "<|im_end|>"]


@pytest.mark.parametrize(
    ("tokens", "match"),
    [
        (USER[:-2] + ASSISTANT[:-2], "no <\\|im_end\\|>"),
        (SYSTEM + USER + ASSISTANT + ["<|im_start|>", "assistant", "\n"], "A turn opens after"),
        (ASSISTANT, "user turn and then an assistant turn"),
        (USER + USER, "expected \\('user', 'assistant'\\)"),
        (SYSTEM + USER + ASSISTANT[:3] + ASSISTANT[7:], "no </think>"),
        (SYSTEM + USER + ASSISTANT[:7] + ASSISTANT[9:], "The answer is empty"),
        (SYSTEM + USER + ASSISTANT[:4] + ["hm"] + ASSISTANT[4:], "thinking block is not empty"),
    ],
)
def test_refuses_a_transcript_that_does_not_end_with_a_closed_answer(tokens: list[str], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        token_masks.token_masks(encode(tokens), Tokenizer())

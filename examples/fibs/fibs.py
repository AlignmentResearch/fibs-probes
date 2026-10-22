"""Read rows of FIBS, the follow-up deception dataset, from the Hugging Face Hub.

A FIBS row is a conversation that ends with a follow-up question (a user turn), plus ``completion``, the
assistant's answer to it. The four splits are named ``<first answer>_<follow-up answer>``, as in
``honest_deceptive``: a conversation whose first answer was honest and whose follow-up answer is deceptive. The
probes score the follow-up answer, so a row's label is set by the second term.

The dataset is public, so no token is needed. ``REVISION`` pins the commit the published probes were trained on.
A split is a ~200 MB parquet file, read here over the Hub's file system one row group at a time, so taking a few
rows downloads only the first row groups (about 10-15 MB).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

REPO_ID = "AlignmentResearch/fibs-v1"
REVISION = "65dccf12934620bfe01db665ea3d7d0e3c9f1355"

#: The label of each split's rows: 1 if the follow-up answer (the second term) is deceptive, else 0.
LABEL_OF_QUADRANT = {"honest_honest": 0, "honest_deceptive": 1, "deceptive_honest": 0, "deceptive_deceptive": 1}
SPLITS = tuple(LABEL_OF_QUADRANT)

COLUMNS = ["messages_json", "completion", "completion_reasoning", "tools", "source"]


@dataclass(frozen=True)
class FibsRow:
    """One FIBS row."""

    split: str
    row_index: int  # position in the split (across its parquet files, in name order)
    messages: list[dict[str, Any]]  # the conversation; the last message is the follow-up question (a user turn)
    completion: str  # the assistant's answer to the follow-up question
    reasoning: str  # the answer's reasoning trace, if it has one
    tools: str | None  # the tool definitions, if the row has any
    source: str

    @property
    def label(self) -> int:
        """1 if the follow-up answer is deceptive, else 0."""
        return LABEL_OF_QUADRANT[self.split]

    @property
    def conversation(self) -> list[dict[str, Any]]:
        """The whole conversation as chat messages: ``messages`` and the answer to the follow-up question."""
        return [*self.messages, {"role": "assistant", "content": self.completion}]


def read_rows(source: Any, split: str, *, start: int = 0) -> Iterator[FibsRow]:
    """Yield the rows of a split's parquet file (a path or an open file), in file order, numbered from ``start``."""
    batches = pq.ParquetFile(source).iter_batches(batch_size=64, columns=COLUMNS)
    for index, record in enumerate((record for batch in batches for record in batch.to_pylist()), start=start):
        yield FibsRow(
            split=split,
            row_index=index,
            messages=json.loads(record["messages_json"]),
            completion=record["completion"],
            reasoning=record["completion_reasoning"] or "",
            tools=record["tools"],
            source=record["source"] or "",
        )


def iter_fibs(split: str, *, repo_id: str = REPO_ID, revision: str = REVISION) -> Iterator[FibsRow]:
    """Yield the rows of one split of FIBS lazily, reading the parquet file over the Hub's file system."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}.")
    fs = HfFileSystem()
    paths = sorted(fs.glob(f"datasets/{repo_id}@{revision}/data/{split}-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"{repo_id}@{revision} has no parquet files for split {split!r}.")
    index = 0  # fibs-v1 has one file per split; earlier versions spread some splits over two
    for path in paths:
        with fs.open(path) as file:
            for row in read_rows(file, split, start=index):
                yield row
                index = row.row_index + 1

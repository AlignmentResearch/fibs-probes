"""Offline tests of fibs.py: reading rows from a parquet file, and the label rule."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import fibs

MESSAGES = [
    {"role": "user", "content": "Is the sky green?"},
    {"role": "assistant", "content": "No, it is blue."},
    {"role": "user", "content": "Did you tell the truth?"},
]


def record(completion: str, **overrides: object) -> dict:
    """One parquet record with the columns ``fibs`` reads."""
    base = {
        "messages_json": json.dumps(MESSAGES),
        "completion": completion,
        "completion_reasoning": "",
        "tools": None,
        "source": "synthetic",
    }
    return {**base, **overrides}


def test_read_rows_numbers_rows_by_file_position_across_row_groups(tmp_path: Path) -> None:
    records = [record("A"), record("B", tools='[{"name": "f"}]'), record("C", completion_reasoning="hm"), record("D")]
    path = tmp_path / "honest_deceptive-00000-of-00001.parquet"
    pq.write_table(pa.Table.from_pylist(records), path, row_group_size=3)  # two row groups

    rows = list(fibs.read_rows(path, "honest_deceptive"))

    assert [(row.row_index, row.completion) for row in rows] == [(0, "A"), (1, "B"), (2, "C"), (3, "D")]
    assert rows[0].messages == MESSAGES  # the follow-up question is the last message
    assert rows[0].conversation == [*MESSAGES, {"role": "assistant", "content": "A"}]
    assert (rows[0].tools, rows[0].reasoning) == (None, "")
    assert rows[1].tools == '[{"name": "f"}]'
    assert rows[2].reasoning == "hm"


def test_iter_fibs_numbers_rows_across_a_split_s_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name, completions in (("00000-of-00002", "AB"), ("00001-of-00002", "CD")):
        table = pa.Table.from_pylist([record(completion) for completion in completions])
        pq.write_table(table, tmp_path / f"honest_deceptive-{name}.parquet")

    class LocalFileSystem:
        def glob(self, pattern: str) -> list[str]:
            return [str(path) for path in tmp_path.glob("honest_deceptive-*.parquet")][::-1]  # any order

        def open(self, path: str):
            return open(path, "rb")

    monkeypatch.setattr(fibs, "HfFileSystem", LocalFileSystem)

    rows = list(fibs.iter_fibs("honest_deceptive"))

    assert [(row.row_index, row.completion) for row in rows] == [(0, "A"), (1, "B"), (2, "C"), (3, "D")]


@pytest.mark.parametrize(
    ("split", "label"),
    [("honest_honest", 0), ("honest_deceptive", 1), ("deceptive_honest", 0), ("deceptive_deceptive", 1)],
)
def test_label_is_set_by_the_follow_up_answer(split: str, label: int) -> None:
    # The second term of the split labels the follow-up answer, whatever the first answer was.
    row = fibs.FibsRow(split, 0, MESSAGES, "Yes.", reasoning="", tools=None, source="synthetic")
    assert row.label == label
    assert split in fibs.SPLITS

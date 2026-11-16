import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eval.run_eval import is_correct, numbers_in

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("text,expected", [
    ("Total revenue is Rp 1,234,567.", 1234567),
    ("Total pendapatan Rp 1.234.567", 1234567),
    ("About 1.24 billion rupiah", 1_243_000_000),
    ("Rp 1,24 miliar", 1_243_000_000),
    ("Average is 312.5.", 312.4),
    ("18 employees", 18),
    ("QRIS share is 20.4%", 20.43),
])
def test_numbers_match(text, expected):
    assert is_correct(text, expected)


def test_wrong_number_and_text():
    assert not is_correct("There are 19 employees.", 18)
    assert is_correct("Jakarta has the highest revenue", "Jakarta")
    assert is_correct("It was June 2026.", ["2026-06", "June 2026"])
    assert not is_correct("Bandung", "Jakarta")


def test_numbers_in_handles_sentence_end():
    assert 300.0 in numbers_in("The answer is 300.")


def test_every_task_expression_evaluates():
    """Guards against typos in tasks.json: each expected expression must run on its dataset."""
    tasks = json.loads((ROOT / "eval" / "tasks.json").read_text(encoding="utf-8"))
    assert len(tasks) >= 40
    for t in tasks:
        df = pd.read_csv(ROOT / "samples" / t["dataset"])
        eval(t["expected"], {"df": df, "pd": pd, "np": np})

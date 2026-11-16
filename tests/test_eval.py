import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agent.numbers import numbers_in
from eval.run_eval import is_correct

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


def test_ungrounded_numbers():
    from agent.numbers import ungrounded_numbers
    ev = "sum of 'revenue' over 3,025 rows: 3,474,602,910.00. share_pct 38.2"
    assert ungrounded_numbers("Total is Rp 3.47 billion (38.2%), top 3 cities.", ev) == []
    assert ungrounded_numbers("Total is $11,486,290.00 over 3,025 rows.", ev) == ["11,486,290.00"]


def test_plain_decimal_is_not_read_as_indonesian_thousands():
    from agent.numbers import ungrounded_numbers
    assert 4733435400.0 not in numbers_in("Bandung,473343540.0,100.0")
    assert ungrounded_numbers("Revenue is Rp 4,733,435,400.00.", "Bandung,473343540.0,100.0") == ["4,733,435,400.00"]


def test_csv_row_splits_into_numbers():
    assert {20.0, 14.142135623730951, 10.0} <= set(numbers_in("revenue,2.0,,,,20.0,14.142135623730951,10.0"))


def test_numbers_in_handles_sentence_end():
    assert 300.0 in numbers_in("The answer is 300.")


def test_every_task_expression_evaluates():
    """Guards against typos in tasks.json: each expected expression must run on its dataset."""
    tasks = json.loads((ROOT / "eval" / "tasks.json").read_text(encoding="utf-8"))
    assert len(tasks) >= 40
    for t in tasks:
        df = pd.read_csv(ROOT / "samples" / t["dataset"])
        eval(t["expected"], {"df": df, "pd": pd, "np": np})

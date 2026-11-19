from pathlib import Path

import pandas as pd
import pytest

from agent.actions import available, run
from agent.state import DatasetState

SAMPLES = Path(__file__).parent.parent / "samples"


def labels(df):
    return [a.label.split(" ", 1)[1] for a in available(df)]


@pytest.mark.parametrize("name", ["toko_sembako.csv", "sales_id.csv", "employees.csv"])
def test_every_action_runs_without_the_model(name):
    state = DatasetState(pd.read_csv(SAMPLES / name), name)
    for action in available(state.current):
        ans = run(action, state)
        assert ans.text and ans.steps and all(s.error is None for s in ans.steps)
        assert all("no model call" in s.fixes[0] for s in ans.steps)


def test_buttons_match_the_dataset():
    shop = labels(pd.read_csv(SAMPLES / "toko_sembako.csv"))
    assert "Restock list" in shop and "Best sellers" in shop and "Monthly revenue chart" in shop
    assert "Restock list" not in labels(pd.read_csv(SAMPLES / "sales_id.csv"))  # no stock column
    assert labels(pd.read_csv(SAMPLES / "employees.csv")) == ["Overview", "Clean data",
                                                              "Average salary by department"]


def test_clean_data_cleans_and_can_be_undone():
    df = pd.read_csv(SAMPLES / "sales_id.csv")
    state = DatasetState(df, "s")
    ans = run(next(a for a in available(df) if "Clean" in a.label), state)
    assert not state.current.duplicated().any() and state.current.isna().sum().sum() == 0
    assert "removed **25** duplicate rows" in ans.text
    while state.undo():
        pass
    assert state.current.equals(df)


def test_clean_data_on_clean_data_adds_no_versions():
    state = DatasetState(pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}), "t")
    ans = run(available(state.current)[1], state)
    assert ans.text.startswith("Nothing to clean") and len(state.versions) == 1


def test_monthly_chart_and_restock():
    df = pd.read_csv(SAMPLES / "toko_sembako.csv")
    state = DatasetState(df, "t")
    acts = {a.label.split(" ", 1)[1]: a for a in available(df)}
    chart = run(acts["Monthly revenue chart"], state)
    assert chart.steps[0].result.chart.startswith(b"\x89PNG") and "**2026-05**" in chart.text
    assert "Telur 1kg" in run(acts["Restock list"], state).text

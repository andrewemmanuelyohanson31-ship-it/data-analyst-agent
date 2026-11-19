import pandas as pd
import pytest

from agent.question import mentioned_period, mentioned_values
from agent.tools import run_tool


@pytest.fixture
def df():
    return pd.DataFrame({
        "order_date": ["2025-01-05", "2025-03-20", "2026-03-03", "2026-05-15"],
        "city": ["Jakarta", "Bandung", "Jakarta", "Surabaya"],
        "channel": ["Online", "Store", "Online", "Store"],
        "department": ["Data", "Sales", "Data", "HR"],
        "left_company": ["Yes", "No", "No", "Yes"],
        "produk": ["Teh", "Teh Botol", "Teh Botol", "Kopi Sachet"],
        "revenue": [100.0, 50.0, 200.0, 30.0],
    })


def test_period_parsing():
    assert mentioned_period("Berapa total laba bulan Maret 2026?") == "2026-03"
    assert mentioned_period("What was the total revenue in 2025?") == "2025"
    assert mentioned_period("How many orders were placed in March 2026?") == "2026-03"
    assert mentioned_period("revenue for 2026-05") == "2026-05"
    assert mentioned_period("Compare 2025 and 2026") is None  # a comparison: don't filter to one
    assert mentioned_period("Which month had the highest revenue?") is None
    assert mentioned_period("revenue in March") is None  # month without year is ambiguous


def test_values_named_in_question(df):
    assert mentioned_values("Which product generated the most revenue in Jakarta?", df) == {"city": "Jakarta"}
    assert mentioned_values("total revenue from Online orders", df) == {"channel": "Online"}
    assert mentioned_values("Berapa pendapatan dari Teh Botol?", df) == {"produk": "Teh Botol"}  # longest match wins


def test_values_not_matched(df):
    assert mentioned_values("Online or Store: which channel is higher?", df) == {}  # two values = comparison
    assert mentioned_values("Is there no data for this?", df) == {}  # "No" too short, lowercase "data" is a word
    assert mentioned_values("Give me an overview of this data", df) == {}
    assert mentioned_values("How many Data team members?", df) == {"department": "Data"}  # exact case = the value


def test_group_by_virtual_month_and_year(df):
    r = run_tool("group_aggregate", df, {"by": ["month"], "column": "revenue", "agg": "sum"})
    assert r.table.iloc[0]["month"] == "2026-03" and "Highest: 2026-03 = 200.00" in r.text
    r = run_tool("group_aggregate", df, {"by": "tahun", "column": "revenue", "agg": "sum"})
    assert set(r.table["tahun"]) == {"2025", "2026"}
    r = run_tool("group_aggregate", df, {"by": ["hari"], "agg": "count"})
    assert any("(" in str(h) for h in r.table["hari"])  # weekday names carry the Indonesian name

from pathlib import Path

import pandas as pd
import pytest

from agent.router import answer, indonesian, plan

SAMPLES = Path(__file__).parent.parent / "samples"


@pytest.fixture(scope="module")
def sales():
    return pd.read_csv(SAMPLES / "sales_id.csv")


@pytest.fixture(scope="module")
def toko():
    return pd.read_csv(SAMPLES / "toko_sembako.csv")


def test_group_top_with_filter(sales):
    text, tool, args, _ = answer("Which product generated the most revenue in Jakarta?", sales)
    want = sales[sales.city == "Jakarta"].groupby("product").revenue.sum()
    assert tool == "group_aggregate" and args["agg"] == "sum"
    assert f"**{want.idxmax()}**" in text and f"{want.max():,.0f}" in text


def test_group_bottom_indonesian(toko):
    text, _, args, _ = answer("Kategori mana yang paling sedikit terjual?", toko)
    want = toko.groupby("kategori").terjual.sum()
    assert args == {"by": ["kategori"], "agg": "sum", "column": "terjual"}
    assert f"**{want.idxmin()}**" in text and text.startswith("Kategori dengan")


def test_compound_group_noun(sales):
    assert plan("Which product category sold the most units?", sales).args["by"] == ["category"]


def test_count_with_descriptor_group_word():
    df = pd.DataFrame({"department": ["Marketing", "Sales", "Marketing"], "salary": [1, 2, 3]})
    text, tool, args, _ = answer("Berapa jumlah karyawan di departemen Marketing?", df)
    assert tool == "aggregate" and args["where"] == [{"column": "department", "op": "==", "value": "Marketing"}]
    assert "**2**" in text


def test_period_total_and_month_group(sales):
    text, _, args, _ = answer("What was the total revenue in 2025?", sales)
    assert args["where"] == [{"column": "order_date", "op": "==", "value": "2025"}]
    assert f"{sales[sales.order_date.str.startswith('2025')].revenue.sum():,.2f}" in text
    text, _, args, _ = answer("Bulan apa dengan pendapatan tertinggi di tahun 2025?", sales)
    assert args["by"] == ["month"] and "**2025-" in text
    _, _, args, _ = answer("Which month had the highest total revenue?", sales)
    assert args["by"] == ["month"]


def test_weekday_and_stock(toko):
    daily = toko.groupby("tanggal").pendapatan.sum()
    quietest = daily.groupby(pd.to_datetime(daily.index).day_name()).mean().idxmin()
    text, tool, _, _ = answer("Hari apa penjualan paling sepi?", toko)
    assert tool == "weekday_pattern" and f"**{quietest} (" in text
    text, tool, _, _ = answer("Which product will run out of stock first?", toko)
    assert tool == "restock" and "**Telur 1kg**" in text
    text, _, _, _ = answer("Berapa sisa stok Minyak Goreng 2L sekarang?", toko)
    last = toko[(toko.produk == "Minyak Goreng 2L") & (toko.tanggal == toko.tanggal.max())].stok_akhir.iloc[0]
    assert f"**{last}**" in text


@pytest.mark.parametrize("q", [
    "Give me an overview of this data",
    "Clean the missing values and remove duplicates",
    "Show the revenue trend or the forecast",
    "Forecast revenue for the next 3 months",
    "How many employees left the company?",
    "Which channel has the higher average revenue per order, Online or Store?",
    "How many orders had a discount?",
    "What is the revenue of the cheapest product?",
    "Which city grew the fastest?",
    "Kenapa penjualan turun?",
    "How many orders have more than 5 items?",
    "Which product is not selling in Jakarta?",
    "What is the median unit price?",
    "Which city has the highest revenue in Jakarta?",  # contradictory
])
def test_unclear_questions_go_to_the_model(q, sales):
    assert plan(q, sales) is None


def test_ambiguous_measure_goes_to_the_model(toko):
    assert plan("Produk mana yang harganya paling tinggi?", toko) is None  # harga_modal or harga_jual?


def test_language_detection():
    assert indonesian("Produk apa yang paling laris?")
    assert not indonesian("Which product sells the most?")


def test_trend_chart_is_a_line_chart(sales, toko):
    text, tool, args, result = answer("Show the monthly revenue trend as a line chart", sales)
    assert tool == "chart" and args["kind"] == "line" and args["x"] == "month" and args["y"] == "revenue"
    assert result.chart.startswith(b"\x89PNG") and len(result.table) == sales.order_date.str[:7].nunique()
    _, _, args, _ = answer("Buat grafik penjualan mingguan", toko)
    assert args["x"] == "week" and args["y"] == "pendapatan"
    _, _, args, _ = answer("Show the daily revenue trend in March 2026", sales)
    assert args["x"] == "order_date" and args["where"] == [{"column": "order_date", "op": "==", "value": "2026-03"}]


def test_chart_by_group_is_a_bar_chart(sales):
    text, _, args, result = answer("Chart revenue by city", sales)
    assert args["kind"] == "bar" and args["x"] == "city"
    assert f"**{sales.groupby('city').revenue.sum().idxmax()}**" in text and result.chart

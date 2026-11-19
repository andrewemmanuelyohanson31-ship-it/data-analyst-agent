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


@pytest.fixture(scope="module")
def sizes():
    import numpy as np
    r = np.random.default_rng(0)
    n = 400
    return pd.DataFrame({"weight": r.integers(45, 90, n), "age": r.integers(18, 70, n).astype(float),
                         "height": r.normal(165, 7, n).round(1), "size": r.choice(["S", "M", "L", "XL", "XXXL"], n)})


@pytest.mark.parametrize("q,expected", [
    ("What is the average age for size M?", lambda d: d[d["size"] == "M"].age.mean()),
    ("Berapa rata-rata berat badan untuk ukuran L?", lambda d: d[d["size"] == "L"].weight.mean()),
    ("How many people wear size XL?", lambda d: (d["size"] == "XL").sum()),
    ("What is the maximum weight for size S?", lambda d: d[d["size"] == "S"].weight.max()),
    ("What is the average weight of people older than 50?", lambda d: d[d.age > 50].weight.mean()),
    ("How many people are taller than 170 cm?", lambda d: (d.height > 170).sum()),
    ("What is the average weight for size M and age under 30?",
     lambda d: d[(d["size"] == "M") & (d.age < 30)].weight.mean()),
    ("What percentage of people wear size L?", lambda d: (d["size"] == "L").mean() * 100),
    ("What is the oldest age?", lambda d: d.age.max()),
])
def test_short_values_conditions_and_percentages(sizes, q, expected):
    from agent.numbers import numbers_in
    from eval.run_eval import is_correct
    text = answer(q, sizes)[0]
    assert is_correct(text, float(expected(sizes))), text
    assert numbers_in(text)


def test_short_value_needs_its_column_name(sizes):
    from agent.question import mentioned_values
    assert mentioned_values("What is the average weight for size M?", sizes) == {"size": "M"}
    assert mentioned_values("I am a student", sizes) == {}  # "a"/"M" alone are just words


def test_indonesian_without_question_word():
    assert indonesian("Rata-rata berat badan untuk ukuran M?")
    assert not indonesian("What is the average weight for size M?")


@pytest.fixture(scope="module")
def cars():
    import numpy as np
    r = np.random.default_rng(1)
    n = 300
    return pd.DataFrame({
        "Date": pd.date_range("2023-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
        "Company": r.choice(["Ford", "Honda", "Toyota"], n), "Body Style": r.choice(["SUV", "Sedan", "Hatchback"], n),
        "Dealer_Name": r.choice(["Alpha Motors", "Beta Cars"], n), "Dealer_No": r.choice(["111-1", "222-2"], n),
        "Dealer_Region": r.choice(["Austin", "Pasco"], n), "Price ($)": r.integers(10, 60, n) * 1000})


@pytest.mark.parametrize("q,expected", [
    ("How many cars were sold?", lambda d: len(d)),  # one row = one car
    ("How many cars did Toyota sell?", lambda d: (d.Company == "Toyota").sum()),
    ("How many SUVs were sold in Austin?", lambda d: ((d["Body Style"] == "SUV") & (d.Dealer_Region == "Austin")).sum()),
    ("What is the total revenue?", lambda d: d["Price ($)"].sum()),  # no revenue column: sum of prices
    ("What is the average price of cars sold in Pasco?", lambda d: d[d.Dealer_Region == "Pasco"]["Price ($)"].mean()),
])
def test_one_row_per_sale(cars, q, expected):
    from eval.run_eval import is_correct
    assert is_correct(answer(q, cars)[0], float(expected(cars)))


def test_group_column_ties(cars):
    assert plan("Which company sold the most cars?", cars).args == {"by": ["Company"], "agg": "count", "where": None}
    assert plan("Which dealer region has the highest total sales?", cars).args["by"] == ["Dealer_Region"]
    assert plan("Which dealer sold the most cars?", cars).args["by"] == ["Dealer_Name"]  # "dealer": the name column

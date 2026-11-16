import numpy as np
import pandas as pd
import pytest

from agent.tools import ToolError, apply_where, col, load_file, run_tool


@pytest.fixture
def df():
    return pd.DataFrame({
        "City": ["Jakarta", "Bandung", "Jakarta", "Surabaya", "Bandung", "Jakarta"],
        "revenue": [100.0, 50.0, np.nan, 30.0, 50.0, 200.0],
        "qty": [1, 2, 3, 4, 2, 6],
        "date": pd.to_datetime(["2025-01-05", "2025-01-20", "2025-02-03", "2025-02-15", "2025-01-20", "2025-03-01"]),
    })


def test_profile_reports_missing_and_duplicates(df):
    df = pd.concat([df, df.iloc[[0]]])
    r = run_tool("profile", df, {})
    assert "1 missing" in r.text and "2 duplicate" in r.text  # fixture already has one Bandung duplicate


def test_clean_missing_median_does_not_mutate(df):
    r = run_tool("clean_missing", df, {"strategy": "median", "column": "revenue"})
    assert r.new_df["revenue"].isna().sum() == 0
    assert r.new_df["revenue"].iloc[2] == 50.0
    assert df["revenue"].isna().sum() == 1


def test_clean_missing_whole_table_skips_text(df):
    df.loc[0, "City"] = None
    r = run_tool("clean_missing", df, {"strategy": "mean"})
    assert r.new_df["revenue"].isna().sum() == 0


def test_col_case_insensitive_and_error_lists_columns(df):
    assert col(df, "CITY") == "City"
    with pytest.raises(ToolError, match="Columns are: City, revenue"):
        col(df, "region")


def test_apply_where_ops(df):
    assert len(apply_where(df, [{"column": "revenue", "op": ">", "value": "60"}])) == 2
    assert len(apply_where(df, [{"column": "city", "op": "contains", "value": "jak"}])) == 3
    assert len(apply_where(df, {"column": "city", "op": "==", "value": "jakarta"})) == 3
    assert len(apply_where(df, [{"column": "date", "op": ">=", "value": "2025-02-01"}])) == 3


def test_load_file_semicolon_and_latin1():
    data = "kota;nilai\nMédan;5\nJakarta;7\n".encode("latin-1")
    out = load_file("x.csv", data)
    assert list(out.columns) == ["kota", "nilai"] and out["nilai"].sum() == 12


def test_unknown_tool_and_bad_args(df):
    with pytest.raises(ToolError, match="Unknown tool"):
        run_tool("magic", df, {})
    with pytest.raises(ToolError, match="Bad arguments"):
        run_tool("profile", df, {"nonsense": 1})


def test_drop_duplicates(df):
    assert len(run_tool("drop_duplicates", df, {}).new_df) == 5


def test_convert_types_number():
    d = pd.DataFrame({"price": ["Rp 1.000", "2000", "abc"]})
    r = run_tool("convert_types", d, {"column": "price", "to": "number"})
    assert r.new_df["price"].tolist()[1] == 2000 and "1 values could not" in r.text


def test_group_aggregate_top_and_share(df):
    r = run_tool("group_aggregate", df, {"by": ["city"], "column": "revenue", "agg": "sum"})
    assert r.table.iloc[0]["City"] == "Jakarta"
    assert r.table["share_pct"].sum() == pytest.approx(100, abs=0.05)
    assert "Jakarta = 300.00" in r.text


def test_aggregate_total_count_and_filter(df):
    assert "300.00" in run_tool("aggregate", df, {"agg": "sum", "column": "revenue", "where": [{"column": "city", "op": "==", "value": "Jakarta"}]}).text
    assert "Count of rows: 6" in run_tool("aggregate", df, {}).text
    assert "Count of rows: 2" in run_tool("aggregate", df, {"where": [{"column": "city", "op": "==", "value": "bandung"}]}).text
    with pytest.raises(ToolError, match="needs a column"):
        run_tool("aggregate", df, {"agg": "mean"})


def test_small_model_slips_are_tolerated(df):
    assert "Count of rows: 6" in run_tool("group_aggregate", df, {"agg": "count"}).text  # no `by` -> overall number
    assert "filled" in run_tool("aggregate", df, {"agg": "count", "column": ["city"]}).text  # list-wrapped column
    assert len(run_tool("describe", df, {"column": ["qty"]}).table) == 1  # column vs columns


def test_count_column_reports_missing(df):
    assert "5 filled, 1 missing" in run_tool("aggregate", df, {"agg": "count", "column": "revenue"}).text


def test_group_reports_lowest_and_filter_reports_pct(df):
    assert "Lowest: Surabaya = 30.00" in run_tool("group_aggregate", df, {"by": ["city"], "column": "revenue"}).text
    assert "(50.00%)" in run_tool("filter_rows", df, {"where": [{"column": "city", "op": "==", "value": "Jakarta"}]}).text


def test_year_and_month_filters_on_date_columns(df):
    text_dates = df.assign(date=df["date"].dt.strftime("%Y-%m-%d"))
    for d in (df, text_dates):
        assert len(apply_where(d, [{"column": "date", "op": "==", "value": "2025"}])) == 6
        assert len(apply_where(d, [{"column": "date", "op": "==", "value": "2025-02"}])) == 2


def test_group_count_without_column(df):
    r = run_tool("group_aggregate", df, {"by": "city", "agg": "count"})
    assert r.table.iloc[0]["count"] == 3


def test_numeric_tool_on_text_column(df):
    with pytest.raises(ToolError, match="not numeric"):
        run_tool("correlation", df, {"columns": ["city", "qty"]})
    with pytest.raises(ToolError, match="not numeric"):
        run_tool("group_aggregate", df, {"by": ["qty"], "column": "city", "agg": "mean"})


def test_outliers_iqr_finds_planted():
    d = pd.DataFrame({"v": [10, 11, 12, 10, 11, 12, 10, 500]})
    r = run_tool("outliers", d, {"column": "v"})
    assert r.table["v"].tolist() == [500]


def test_time_trend_monthly(df):
    r = run_tool("time_trend", df, {"date_column": "date", "value_column": "qty", "freq": "M"})
    assert r.table.iloc[:, 1].tolist() == [5, 7, 6]


def test_forecast_linear_perfect_line():
    d = pd.DataFrame({"d": pd.date_range("2025-01-01", periods=6, freq="MS"), "v": [10, 20, 30, 40, 50, 60]})
    r = run_tool("forecast", d, {"date_column": "d", "value_column": "v", "periods": 2})
    assert r.table.iloc[:, 1].tolist() == pytest.approx([70, 80])


def test_chart_png(df):
    for args in ({"kind": "bar", "x": "city", "y": "revenue"}, {"kind": "hist", "x": "qty"}, {"kind": "heatmap"},
                 {"kind": "scatter", "x": "qty", "y": "revenue"}, {"kind": "box", "x": "city", "y": "qty"},
                 {"kind": "line", "x": "date", "y": "qty"}):
        assert run_tool("chart", df, args).chart.startswith(b"\x89PNG")


def test_pivot_value_counts_describe_sort(df):
    assert "Jakarta" in run_tool("pivot", df, {"index": "city", "columns": "qty", "values": "revenue"}).table["City"].tolist()
    assert run_tool("value_counts", df, {"column": "city"}).table.iloc[0]["count"] == 3
    assert len(run_tool("describe", df, {"columns": ["qty"]}).table) == 1
    assert run_tool("sort_top_n", df, {"column": "qty", "n": 2}).table["qty"].tolist() == [6, 4]

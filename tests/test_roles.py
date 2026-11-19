import sys
from pathlib import Path

import pandas as pd
import pytest

from agent.roles import detect_roles, split_name

sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))
from score_roles import detector, score  # noqa: E402


def test_split_name():
    assert split_name("OrderDate") == "order_date"
    assert split_name("QUANTITYORDERED") == "quantity_ordered"
    assert split_name("PRODUCTCODE") == "product_code"
    assert split_name("Harga Jual (Rp)") == "harga_jual"
    assert split_name("Supercalifragilistic") == "supercalifragilistic"  # unknown words: left alone


def test_arithmetic_finds_quantity_price_revenue_whatever_the_names():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5, 2], "b": [10.0, 20.0, 15.0, 10.0, 12.5, 30.0],
                       "c": [10.0, 40.0, 45.0, 40.0, 62.5, 60.0], "note": list("xyzxyz")})
    assert detect_roles(df) == {"a": "quantity", "b": "price", "c": "revenue", "note": None}


def test_qty_times_cost_is_spending_not_revenue():
    df = pd.DataFrame({"Qty": [2, 3, 4, 5, 6], "Harga Beli": [100, 200, 150, 100, 120],
                       "Total": [200, 600, 600, 500, 720]})
    assert detect_roles(df) == {"Qty": "quantity", "Harga Beli": "cost", "Total": None}


def test_values_override_a_misleading_name():
    df = pd.DataFrame({"Location": ["Takeaway", "In-store", "UNKNOWN", "Takeaway", "In-store"]})
    assert detect_roles(df)["Location"] == "channel"
    stores = pd.DataFrame({"Location": ["Cabang Kemang", "Cabang Depok", "Cabang Kemang"]})
    assert detect_roles(stores)["Location"] == "store"


def test_stock_report_in_and_out():
    df = pd.DataFrame({"Stok Awal": [10, 20], "Masuk": [5, 0], "Keluar": [3, 4], "Stok Akhir": [12, 16]})
    assert detect_roles(df) == {"Stok Awal": "opening_stock", "Masuk": "stock_in", "Keluar": "quantity",
                                "Stok Akhir": "stock"}


def test_ambiguous_jumlah_decided_by_values():
    small = pd.DataFrame({"Jumlah": [1, 2, 3, 1, 2]})
    assert detect_roles(small)["Jumlah"] == "quantity"
    money = pd.DataFrame({"Jumlah": [150000.0, 230000.0, 99000.5]})
    assert detect_roles(money)["Jumlah"] is None  # large decimals: not a count; left for the user to confirm


def test_type_must_fit_and_one_column_per_role():
    df = pd.DataFrame({"Tanggal": ["Senin", "Selasa"], "Tgl Beli": ["2026-05-01", "2026-05-02"],
                       "Jatuh Tempo": ["2026-05-15", "2026-05-16"]})
    roles = detect_roles(df)
    assert roles["Tanggal"] is None  # named like a date, but the values aren't dates
    assert roles["Tgl Beli"] == "date" and roles["Jatuh Tempo"] is None  # only one date column


@pytest.mark.parametrize("groups", [{"samples"}, {"messy"}, {"heldout"}])
def test_answer_key(groups, capsys):
    right, total, misses = score(detector(names_only=False), None, verbose=False, groups=groups)
    assert total > 0 and misses == [], misses

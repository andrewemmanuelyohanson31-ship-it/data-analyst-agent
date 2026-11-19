from pathlib import Path

import pandas as pd

from agent.column_names import AMBIGUOUS, INTENTS, ROLES, available_intents, candidates, normalise, role_of

SAMPLES = Path(__file__).parent.parent / "samples"


def test_each_name_has_one_role():
    seen = {}
    for role, info in ROLES.items():
        for name in info["names"]:
            assert name not in seen, f"{name!r} is in both {seen[name]} and {role}"
            assert name not in AMBIGUOUS, f"{name!r} is listed as ambiguous and as {role}"
            assert name == normalise(name), f"{name!r} isn't in normalised form"
            seen[name] = role


def test_ambiguous_and_intents_point_at_real_roles():
    for options in AMBIGUOUS.values():
        assert set(options) <= set(ROLES)
    for info in INTENTS.values():
        for need in info["needs"]:
            assert set(need.split("|")) <= set(ROLES), need


def test_normalise():
    assert normalise("Harga Jual (Rp)") == "harga_jual"
    assert normalise("Order-Date") == "order_date"
    assert normalise("Discount %") == "discount"
    assert normalise("harga_per_unit") == "harga_per_unit"  # "unit" is part of the name, not a suffix
    assert normalise("Qty") == "qty"


def test_role_of_real_export_style_names():
    assert role_of("Tanggal Transaksi") == "date"
    assert role_of("Nama Barang") == "product"
    assert role_of("Qty Keluar") == "quantity"
    assert role_of("Total Penjualan (Rp)") == "revenue"
    assert role_of("HPP") == "cost"
    assert role_of("Stok Akhir") == "stock"
    assert role_of("Metode Pembayaran") == "payment"
    assert role_of("something_else") is None


def test_ambiguous_names_are_not_guessed():
    assert role_of("Jumlah") is None
    assert candidates("Jumlah") == ["quantity", "revenue"]
    assert candidates("Nama Barang") == ["product"] and candidates("xyz") == []


def test_bundled_samples_are_fully_recognised():
    expected = {
        "toko_sembako.csv": {"tanggal": "date", "produk": "product", "kategori": "category", "terjual": "quantity",
                             "harga_modal": "cost", "harga_jual": "price", "pendapatan": "revenue",
                             "laba": "profit", "stok_akhir": "stock"},
        "sales_id.csv": {"order_id": "transaction_id", "order_date": "date", "city": "city", "channel": "channel",
                         "category": "category", "product": "product", "quantity": "quantity",
                         "unit_price": "price", "discount_pct": "discount", "revenue": "revenue",
                         "payment_method": "payment", "customer_age": "age"},
        "employees.csv": {"employee_id": "employee_id", "department": "department", "city": "city",
                          "hire_date": "hire_date", "years_experience": "experience",
                          "monthly_salary_idr": "salary", "performance_score": "performance",
                          "training_hours": "training", "left_company": "attrition"},
    }
    for name, roles in expected.items():
        cols = pd.read_csv(SAMPLES / name, nrows=1).columns
        assert {c: role_of(c) for c in cols} == roles


def test_available_intents():
    shop = {"date", "product", "category", "quantity", "cost", "price", "revenue", "profit", "stock"}
    assert {"restock", "weekday", "trend_chart", "margin"} <= set(available_intents(shop))
    hr = {"employee_id", "department", "salary", "hire_date"}
    assert "restock" not in available_intents(hr) and "average" in available_intents(hr)

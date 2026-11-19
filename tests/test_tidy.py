import io
from pathlib import Path

import pandas as pd
from openpyxl import Workbook

from agent.tidy import parse_money, tidy
from agent.tools import load_file

SAMPLES = Path(__file__).parent.parent / "samples"


def test_clean_samples_are_not_changed():
    # regression: day-first parsing once turned ISO 2026-01-02 into 1 February
    for name in ("toko_sembako.csv", "sales_id.csv", "employees.csv"):
        raw = pd.read_csv(SAMPLES / name)
        loaded = load_file(name, (SAMPLES / name).read_bytes())
        pd.testing.assert_frame_equal(loaded, raw, check_dtype=False)
        assert loaded.attrs["load_notes"] == []


def test_parse_money():
    assert parse_money("Rp 15.000") == 15000
    assert parse_money("Rp15.000,50") == 15000.5
    assert parse_money("15rb") == 15000
    assert parse_money("1,5jt") == 1_500_000
    assert parse_money("15,000.25") == 15000.25
    assert parse_money(32000) == 32000
    assert parse_money("BRG-001") is None and parse_money("ERROR") is None and parse_money("02/05/2026") is None


def test_dates_day_first_and_month_first():
    id_dates = tidy(pd.DataFrame({"tgl": ["1 Mei 2026", "02/05/2026", "03-05-26", "13/05/2026"]}))
    assert id_dates["tgl"].tolist() == ["2026-05-01", "2026-05-02", "2026-05-03", "2026-05-13"]
    us = tidy(pd.DataFrame({"d": ["2/24/2003 0:00", "5/7/2003 0:00", "12/1/2003 0:00"]}))
    assert us["d"].tolist() == ["2003-02-24", "2003-05-07", "2003-12-01"]
    iso = pd.DataFrame({"d": ["2026-01-02", "2026-01-03 10:15"]})
    assert tidy(iso.copy())["d"].tolist() == iso["d"].tolist()


def test_times_ids_and_phones_stay_text():
    df = tidy(pd.DataFrame({"jam": ["10:01", "20:39"], "kode": ["BRG-001", "BRG-002"], "hp": ["0812345", "0813999"]}))
    assert df["jam"].tolist() == ["10:01", "20:39"] and df["kode"].tolist() == ["BRG-001", "BRG-002"]
    assert df["hp"].tolist() == ["0812345", "0813999"]


def test_bad_values_become_empty_with_a_note():
    df = tidy(pd.DataFrame({"qty": ["2", "3", "ERROR", "4", "5", "1", "2", "3", "4", "5"]}))
    assert df["qty"].isna().sum() == 1 and df["qty"].sum() == 29
    assert "ERROR" in df.attrs["load_notes"][0]


def test_handmade_excel_header_totals_and_money():
    wb = Workbook()
    ws = wb.active
    ws["A1"] = "LAPORAN PENJUALAN"
    ws.append([])
    ws.append(["No", "Tgl", "Barang", "Jml", "Harga", "Total"])
    ws.append([1, "1 Mei 2026", "Gula", 2, "Rp 18.000", "Rp 36.000"])
    ws.append([2, "01/05/2026", "Kopi", 3, "2rb", 6000])
    ws.append(["", "", "TOTAL", "", "", 42000])
    ws.append([])
    ws.append([3, "2 Mei 2026", "Gula", 1, 18000, 18000])
    buf = io.BytesIO()
    wb.save(buf)
    df = load_file("buku.xlsx", buf.getvalue())
    assert list(df.columns) == ["No", "Tgl", "Barang", "Jml", "Harga", "Total"]
    assert df["Total"].sum() == 60000  # the TOTAL row is gone, so nothing is counted twice
    assert df["Harga"].tolist() == [18000, 2000, 18000]
    assert df["Tgl"].tolist() == ["2026-05-01", "2026-05-01", "2026-05-02"]
    assert any("header" in n for n in df.attrs["load_notes"])

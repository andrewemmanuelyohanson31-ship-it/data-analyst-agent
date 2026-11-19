"""Make spreadsheet exports usable right after loading: real header row, no blank/total rows, money written as
text turned into numbers, dates in one format. Every change is noted in df.attrs["load_notes"] for the UI.
"""
from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd

BULAN = {m: i for i, m in enumerate(["januari", "februari", "maret", "april", "mei", "juni", "juli", "agustus",
                                     "september", "oktober", "november", "desember"], 1)}
BULAN.update({m[:3]: i for m, i in list(BULAN.items())})  # jan, feb, mar, apr, mei, jun, jul, agu, sep, okt...
BULAN.update({m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                          "september", "october", "november", "december"], 1)})
BULAN.update({"aug": 8, "oct": 10, "dec": 12, "des": 12, "agt": 8, "ags": 8})
TOTAL_WORDS = re.compile(r"^\s*(grand\s*)?(sub\s*)?total\b|^\s*jumlah\s*$|^\s*total\s*keseluruhan", re.I)
MONEY = re.compile(r"^\s*(rp\.?|idr|us\$|usd|\$)?\s*-?[\d.,]+\s*(rb|ribu|k|jt|juta|m)?\s*$", re.I)
TIME_ONLY = re.compile(r"^\s*\d{1,2}[:.]\d{2}([:.]\d{2})?\s*$")


def _header_row(df: pd.DataFrame, max_scan: int = 15) -> int | None:
    """Row index holding the real header when the file starts with title rows; None if row 0 is the header."""
    unnamed = sum(str(c).startswith("Unnamed") for c in df.columns)
    if unnamed < len(df.columns) / 2:
        return None
    best, best_n = None, 0
    for i in range(min(max_scan, len(df))):
        cells = df.iloc[i].dropna().astype(str).str.strip()
        texty = cells[cells.str.len().between(1, 40) & ~cells.str.fullmatch(r"[\d.,\-/ ]+")]
        if len(texty) >= max(2, best_n + 1) and len(texty) >= 0.6 * df.shape[1]:
            best, best_n = i, len(texty)
    return best


def parse_money(value) -> float | None:
    """'Rp 15.000' -> 15000, '15rb' -> 15000, '1,5jt' -> 1500000, '12.5' -> 12.5, 15000 -> 15000."""
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return None if pd.isna(value) else float(value)
    s = str(value).strip().lower()
    if not s or not MONEY.match(s):
        return None
    mult = 1
    if m := re.search(r"(rb|ribu|k|jt|juta|m)\s*$", s):
        mult = {"rb": 1e3, "ribu": 1e3, "k": 1e3, "jt": 1e6, "juta": 1e6, "m": 1e6}[m.group(1)]
        s = s[: m.start()]
    s = re.sub(r"^(rp\.?|idr|us\$|usd|\$)\s*", "", s).strip()
    neg = s.startswith("-")
    s = s.lstrip("-")
    if re.fullmatch(r"\d{1,3}(\.\d{3})+(,\d+)?", s):  # Indonesian 15.000 / 1.250.000,50
        s = s.replace(".", "").replace(",", ".")
    elif re.fullmatch(r"\d{1,3}(,\d{2,3})*,\d{3}(\.\d+)?", s):  # English 15,000.50 and Indian 42,37,74,881
        s = s.replace(",", "")
    elif re.fullmatch(r"\d+,\d{1,2}", s):  # decimal comma 1,5
        s = s.replace(",", ".")
    try:
        return (-1 if neg else 1) * float(s) * mult
    except ValueError:
        return None


ISO = re.compile(r"^\d{4}-\d{2}-\d{2}([ t]\d{2}:\d{2}(:\d{2})?)?$")


def _parse_date(value, dayfirst: bool):
    s = str(value).strip().lower()
    if not s or s in ("nan", "none"):
        return pd.NaT
    if ISO.match(s):  # year first is never day-first: dayfirst=True would turn 2026-01-02 into 1 February
        return pd.to_datetime(s, format="ISO8601", errors="coerce")
    if m := re.fullmatch(r"(\d{1,2})\s+([a-z]+)\.?\s+(\d{2,4})", s):  # 1 Mei 2026 / 5 Agu 26
        month = BULAN.get(m.group(2))
        if month:
            year = int(m.group(3)) + (2000 if len(m.group(3)) == 2 else 0)
            return pd.Timestamp(year, month, int(m.group(1)))
    with warnings.catch_warnings():  # pandas warns once per value when formats are mixed
        warnings.simplefilter("ignore")
        return pd.to_datetime(s, dayfirst=dayfirst, errors="coerce")


def _as_dates(s: pd.Series) -> pd.Series | None:
    """Parsed dates if >= 80% of the values are dates, trying day-first (Indonesian) and month-first (US)."""
    vals = s.dropna().astype(str)
    digits = vals[vals.str.contains(r"\d")]
    if len(digits) < 0.8 * len(vals) or digits.empty:
        return None
    if digits.str.fullmatch(r"-?\d+(\.\d+)?").mean() > 0.5 or digits.str.match(TIME_ONLY).mean() > 0.5:
        return None  # plain numbers (IDs, amounts) and clock times are not dates
    # Day or month first? pandas silently "fixes" 2/24 under day-first, so decide from evidence instead:
    # a first number over 12 means day-first (24/2), a second number over 12 means month-first (2/24).
    parts = digits.str.extract(r"^\s*(\d{1,2})[/\-.](\d{1,2})[/\-.]\d{2,4}").dropna().astype(int)
    month_first = bool((parts[1] > 12).any()) and not bool((parts[0] > 12).any())
    dayfirst = not month_first  # no evidence either way: Indonesian convention, day first
    parsed = vals.head(300).map(lambda v: _parse_date(v, dayfirst))
    if parsed.notna().mean() < 0.8:
        return None
    return s.map(lambda v: _parse_date(v, dayfirst) if pd.notna(v) else pd.NaT)


def tidy(df: pd.DataFrame) -> pd.DataFrame:
    notes = []
    header = _header_row(df)
    if header is not None:
        cols = [str(c).strip() if pd.notna(c) else f"column_{i + 1}" for i, c in enumerate(df.iloc[header])]
        df = df.iloc[header + 1:].set_axis(cols, axis=1).reset_index(drop=True)
        notes.append(f"used row {header + 2} as the header (the rows above it were titles)")

    empty = [c for c in df.columns if str(c).startswith("Unnamed") and df[c].isna().all()]
    if empty:  # a trailing comma on every line makes an empty, nameless column
        df = df.drop(columns=empty)
        notes.append(f"removed {len(empty)} empty column{'s' if len(empty) > 1 else ''} without a name")

    blank = df.isna().all(axis=1)
    text = df.select_dtypes(exclude="number").astype(str)
    is_total = text.apply(lambda col: col.str.match(TOTAL_WORDS)).any(axis=1) & (df.notna().sum(axis=1) <= 4)
    if (blank | is_total).any():
        df = df[~(blank | is_total)].reset_index(drop=True)
        parts = [f"{int(is_total.sum())} total/subtotal rows" if is_total.any() else "",
                 f"{int(blank.sum())} empty rows" if blank.any() else ""]
        notes.append("removed " + " and ".join(p for p in parts if p) + " (they would be counted twice)")

    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) or s.dropna().empty:
            continue
        vals = s.dropna()
        if vals.astype(str).str.strip().str.lower().str.match(ISO).all():
            continue  # already clean dates: leave exactly as they are
        dates = _as_dates(s)
        if dates is not None:
            has_time = (dates.dropna().dt.hour.ne(0) | dates.dropna().dt.minute.ne(0)).any()
            df[c] = dates.dt.strftime("%Y-%m-%d %H:%M" if has_time else "%Y-%m-%d")
            bad = int(dates[s.notna()].isna().sum())
            if not vals.astype(str).str.fullmatch(r"\d{4}-\d{2}-\d{2}( \d{2}:\d{2}(:\d{2})?)?").all():
                notes.append(f"'{c}': dates -> YYYY-MM-DD" + (f"; {bad} values that aren't dates left empty" if bad else ""))
            continue
        money = vals.map(parse_money)
        if money.notna().mean() < 0.9 or vals.astype(str).str.fullmatch(r"0\d+").any():
            continue  # mostly text, or codes with leading zeros: leave as text
        df[c] = s.map(lambda v: parse_money(v) if pd.notna(v) else np.nan)
        written = vals[money.notna()].astype(str).str.contains(r"[^\d.\-]").any()
        bad = int(money.isna().sum())
        if written:
            notes.append(f"'{c}': amounts written as text (currency signs, separators) -> numbers")
        if bad:
            notes.append(f"'{c}': {bad} values that aren't numbers (e.g. {vals[money.isna()].iloc[0]!r}) left empty")
    df.attrs["load_notes"] = notes
    return df

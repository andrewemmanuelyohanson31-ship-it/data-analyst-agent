"""Read values and periods named in a question, using the data itself. Used by agent/router.py."""
from __future__ import annotations

import re

import pandas as pd

from agent.tools import _date_like

MONTHS = {  # English and Indonesian month names -> number
    **{m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                   "september", "october", "november", "december"], 1)},
    **{m: i for i, m in enumerate(["januari", "februari", "maret", "april", "mei", "juni", "juli", "agustus",
                                   "september", "oktober", "november", "desember"], 1)},
}
# values that are also everyday words: only match them when written with the data's own capitalisation
GENERIC = {"data", "home", "store", "online", "sales", "other", "total", "remote", "cash"}
MAX_CATEGORIES = 50  # text columns with more distinct values are names/IDs, not categories
MIN_VALUE_LEN = 4  # "Yes"/"No"/"Fan" would match ordinary words


# other words for common column names, English <-> Indonesian ("berat badan" asks about the "weight" column)
ALIASES = {
    "weight": ["berat", "berat badan", "bobot"], "height": ["tinggi", "tinggi badan"], "age": ["umur", "usia"],
    "size": ["ukuran"], "gender": ["jenis kelamin", "kelamin", "sex"], "color": ["warna", "colour"],
    "price": ["harga"], "salary": ["gaji"], "score": ["nilai", "skor"], "city": ["kota"], "region": ["wilayah"],
}
for _en, _ids in list(ALIASES.items()):
    for _id in _ids:
        ALIASES.setdefault(_id, []).append(_en)
# comparison words -> operator; "older than 50" also names the column it is about
MORE = r"more than|greater than|higher than|bigger than|larger than|over|above|lebih dari|di atas|lebih besar dari|>"
LESS = r"less than|lower than|smaller than|fewer than|under|below|kurang dari|di bawah|lebih kecil dari|<"
COMPARATIVES = {"older": ("age", ">"), "younger": ("age", "<"), "taller": ("height", ">"), "shorter": ("height", "<"),
                "heavier": ("weight", ">"), "lighter": ("weight", "<"), "lebih tua": ("age", ">"),
                "lebih muda": ("age", "<"), "lebih tinggi": ("height", ">"), "lebih pendek": ("height", "<"),
                "lebih berat": ("weight", ">"), "lebih ringan": ("weight", "<")}
UNITS = r"(?:\s*(?:cm|kg|kilo|years?(?: old)?|yrs?|tahun|thn|orang|people))?"


def column_names(col) -> list[str]:
    """Every way a question might name this column: 'monthly_salary_idr' -> 'monthly salary idr', 'salary', 'gaji'."""
    base = str(col).lower().replace("_", " ").strip()
    names = {base, *base.split()}
    for n in list(names):
        names.update(ALIASES.get(n, []))
    return sorted((n for n in names if len(n) > 1), key=len, reverse=True)  # longest first: "berat badan" before "berat"


def mentioned_values(question: str, df: pd.DataFrame) -> dict:
    """{column: value} for category values named in the question (one value per column, else ambiguous).

    Short values ("M", "XL", "No") only count right after the column's own name ("size M", "ukuran L"):
    on their own they'd match ordinary words.
    """
    found = {}
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) or _date_like(s) or s.nunique() > MAX_CATEGORIES:
            continue
        hits = []
        named = "|".join(re.escape(n) for n in column_names(c))
        for v in s.dropna().astype(str).unique():
            if len(v) < MIN_VALUE_LEN:
                if re.search(rf"\b(?:{named})\s+{re.escape(v)}(?!\w)", question, re.I):
                    hits.append(v)
                elif len(v) >= 3 and v.isupper() and re.search(rf"(?<!\w){re.escape(v)}s?(?!\w)", question):
                    hits.append(v)  # an acronym in capitals ("SUV", "SUVs") can't be an ordinary word
                continue
            generic = v.lower() in GENERIC
            # a plural counts too: "SUVs", "sedans"
            if re.search(rf"(?<!\w){re.escape(v)}(?:s|es)?(?!\w)", question, 0 if generic else re.I):
                hits.append(v)
        hits = [h for h in hits if not any(h != o and h.lower() in o.lower() for o in hits)]  # "Teh" vs "Teh Botol"
        if len(hits) == 1:
            found[c] = hits[0]
    return found


def mentioned_conditions(question: str, df: pd.DataFrame) -> tuple[list[dict], list[str]]:
    """Numeric conditions: 'weight over 70', 'age below 25', 'older than 50', 'tinggi lebih dari 180 cm'.

    Returns (where conditions, the exact text pieces they came from) so the parser can count those words as understood.
    """
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    conds, pieces = [], []
    q = question.lower()
    for c in numeric:
        named = "|".join(re.escape(n) for n in column_names(c))
        for ops, op in ((MORE, ">"), (LESS, "<")):
            pat = rf"\b(?:{named})\s+(?:is\s+|of\s+|nya\s+)?(?:{ops})\s*(\d+(?:[.,]\d+)?){UNITS}"
            for m in re.finditer(pat, q):
                conds.append({"column": c, "op": op, "value": m.group(1).replace(",", ".")})
                pieces.append(m.group(0))
    for word, (concept, op) in COMPARATIVES.items():
        for m in re.finditer(rf"\b{word}\s+(?:than|dari|daripada)\s+(\d+(?:[.,]\d+)?){UNITS}", q):
            col = next((c for c in numeric if concept in column_names(c)), None)
            if col is None:
                return [], []  # "older than 50" but no age column: don't guess
            conds.append({"column": col, "op": op, "value": m.group(1).replace(",", ".")})
            pieces.append(m.group(0))
    return conds, pieces


def mentioned_period(question: str) -> str | None:
    """'2025', '2026-03' or None. One year (and optionally one month) only; two years = a comparison, skip."""
    q = question.lower()
    if m := re.search(r"\b(20\d\d)-(0[1-9]|1[0-2])\b", q):
        return f"{m.group(1)}-{m.group(2)}"
    years = set(re.findall(r"\b(20\d\d)\b", q))
    months = {MONTHS[w] for w in re.findall(r"[a-z]+", q) if w in MONTHS}
    if len(years) != 1 or len(months) > 1:
        return None
    year = years.pop()
    return f"{year}-{months.pop():02d}" if months else year


def date_column(df: pd.DataFrame):
    return next((c for c in df.columns if _date_like(df[c])), None)


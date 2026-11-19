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


def mentioned_values(question: str, df: pd.DataFrame) -> dict:
    """{column: value} for category values named in the question (one value per column, else ambiguous)."""
    found = {}
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) or _date_like(s) or s.nunique() > MAX_CATEGORIES:
            continue
        hits = []
        for v in s.dropna().astype(str).unique():
            if len(v) < MIN_VALUE_LEN:
                continue
            generic = v.lower() in GENERIC
            if re.search(rf"(?<!\w){re.escape(v)}(?!\w)", question, 0 if generic else re.I):
                hits.append(v)
        hits = [h for h in hits if not any(h != o and h.lower() in o.lower() for o in hits)]  # "Teh" vs "Teh Botol"
        if len(hits) == 1:
            found[c] = hits[0]
    return found


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


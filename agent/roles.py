"""Work out what each column means (its role) from its name AND its values.

    detect_roles(df) -> {column: role or None}

Evidence, strongest first: exact known name > arithmetic between columns (quantity x price = revenue) >
key word inside the name > the values alone. Every guess must fit the column's values (a "date" must hold dates,
a "quantity" numbers). One column per role; when two columns claim a role, the stronger evidence wins.
"""
from __future__ import annotations

import itertools
import re
import weakref

import pandas as pd

from agent.column_names import AMBIGUOUS, ROLES, normalise

EXACT, EQUATION, KEYWORD, VALUES = 4, 3, 2, 1

# key words inside longer names, in priority order: the first role whose word appears wins
KEYWORDS: list[tuple[str, set[str]]] = [
    ("date", {"tanggal", "tgl", "date", "waktu", "datetime", "timestamp", "dibuat", "created"}),
    ("transaction_id", {"struk", "faktur", "invoice", "nota", "receipt", "trx"}),
    ("revenue", set()),  # only via the "starts with total" rule below
    ("cost", {"modal", "hpp", "cost", "beli"}),
    ("price", {"harga", "price", "tarif"}),
    ("quantity", {"qty", "quantity", "jml", "units", "pcs", "kuantitas", "terjual", "sold", "jumlah"}),
    ("discount", {"diskon", "discount", "potongan", "promo"}),
    ("status", {"status"}),
    ("customer", {"customer", "pelanggan", "pembeli", "buyer", "client", "konsumen", "member", "username"}),
    ("supplier", {"supplier", "vendor", "pemasok"}),
    ("product", {"produk", "product", "barang", "item", "menu"}),
    ("category", {"kategori", "category", "jenis"}),
    ("payment", {"pembayaran", "payment"}),
    ("stock", {"stok", "stock"}),
    ("month", {"month", "bulan"}),
    ("year", {"year", "tahun"}),
]
TOTAL_FIRST = {"total", "subtotal", "grand", "amount", "nominal", "omzet", "omset"}
CHANNEL_WORDS = {"online", "offline", "in-store", "instore", "in store", "store", "takeaway", "take away", "dine in",
                 "dine-in", "delivery", "pickup", "pick up", "pick-up", "website", "web", "app", "marketplace",
                 "shopee", "tokopedia", "lazada", "tiktok shop", "whatsapp", "gofood", "grabfood", "shopeefood",
                 "walk-in", "walk in"}
JUNK = {"unknown", "error", "n/a", "na", "-", "none", "null", ""}
VOCAB = sorted({t for info in ROLES.values() for n in info["names"] for t in n.split("_")} |
               {t for _, words in KEYWORDS for t in words} | {"each", "ordered", "line", "number", "code", "name",
                                                               "id", "address", "postal", "contact", "first", "last",
                                                               "deal", "size", "qtr", "msrp", "phone"},
               key=len, reverse=True)
_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}")


def split_name(column) -> str:
    """'OrderDate' -> 'order_date', 'QUANTITYORDERED' -> 'quantity_ordered', 'Harga Jual (Rp)' -> 'harga_jual'."""
    camel = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(column))
    name = normalise(camel)
    if "_" in name or len(name) < 7:
        return name
    words = _segment(name)
    return "_".join(words) if words else name


def _segment(s: str) -> list[str] | None:
    """Split a run-together lowercase name into known words, or None if it can't be fully explained."""
    best: dict[int, list[str]] = {0: []}
    for i in range(1, len(s) + 1):
        for w in VOCAB:
            j = i - len(w)
            if j >= 0 and j in best and s[j:i] == w and (i not in best or len(best[j]) + 1 < len(best[i])):
                best[i] = best[j] + [w]
    return best.get(len(s))


def _kind(s: pd.Series) -> str:
    vals = s.dropna()
    if vals.empty:
        return "empty"
    if pd.api.types.is_bool_dtype(s) or vals.astype(str).str.lower().isin({"true", "false", "yes", "no"}).all():
        return "bool"
    if pd.api.types.is_numeric_dtype(s):
        return "number"
    if vals.astype(str).str.match(_ISO).mean() >= 0.9:
        return "date"
    return "text"


FITS = {"date": {"date"}, "number": {"number"}, "money": {"number"}, "text": {"text", "bool"},
        "id": {"text", "number"}, "label": {"text", "number"}}


def _fits(role: str, kind: str) -> bool:
    if role == "discount" and kind == "bool":  # "Discount Applied: True/False"
        return True
    return kind in FITS[ROLES[role]["type"]]


def _rank(role: str, name: str) -> int:
    names = ROLES[role]["names"]
    return names.index(name) if name in names else len(names)


def _name_guess(col) -> tuple[str | None, int, int] | list[str]:
    """(role, strength, rank) from the name, or the candidate list for an ambiguous name."""
    name = split_name(col)
    for n in (name, name.replace("_", "")):
        for role, info in ROLES.items():
            if n in info["names"]:
                return role, EXACT, _rank(role, n)
    if name in AMBIGUOUS:
        return list(AMBIGUOUS[name])
    tokens = name.split("_")
    if tokens[0] in TOTAL_FIRST:
        return "revenue", KEYWORD, 0
    for role, words in KEYWORDS:
        if words & set(tokens):
            return role, KEYWORD, 0
    return None, 0, 0


def _close(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a - b).abs() <= 0.01 * b.abs().clip(lower=1)


def _equations(df: pd.DataFrame, nums: list) -> list[tuple]:
    """(qty, price, total) triples where qty x price = total (allowing a discount column) on >= 80% of rows."""
    d = df[nums].head(500)
    found = []
    for a, b in itertools.combinations(nums, 2):
        for c in nums:
            if c in (a, b):
                continue
            rows = d[[a, b, c]].dropna()
            if len(rows) < 5:
                continue
            ok = _close(rows[a] * rows[b], rows[c])
            if ok.mean() < 0.8:
                for disc in nums:  # qty x price - discount = total
                    if disc not in (a, b, c):
                        dd = d[disc].reindex(rows.index).fillna(0)
                        ok = ok | _close(rows[a] * rows[b] - dd, rows[c])
            if ok.mean() >= 0.8 and rows[c].abs().sum() > 0:
                qty, price = (a, b) if _small_ints(d[a]) and not _small_ints(d[b]) else (b, a)
                found.append((qty, price, c))
    return found


def _small_ints(s: pd.Series) -> bool:
    v = s.dropna()
    return not v.empty and bool((v == v.round()).all()) and v.median() <= 100 and v.min() >= 0


def _channel_like(s: pd.Series) -> bool:
    v = s.dropna().astype(str).str.strip().str.lower()
    v = v[~v.isin(JUNK)]
    return len(v) > 0 and v.isin(CHANNEL_WORDS).mean() >= 0.8


_CACHE: dict[int, tuple] = {}  # id(df) -> (weak reference, roles); tools never change a DataFrame in place


def detect_roles(df: pd.DataFrame) -> dict:
    """Roles for every column; computed once per DataFrame (each cleaned version is a new object)."""
    hit = _CACHE.get(id(df))
    if hit and hit[0]() is df:
        return dict(hit[1])
    roles = _detect(df)
    if len(_CACHE) > 64:
        _CACHE.clear()
    _CACHE[id(df)] = (weakref.ref(df), roles)
    return dict(roles)


def columns_by_role(df: pd.DataFrame) -> dict:
    """{role: column}. When every row is one sale (a price, but no quantity or revenue column, e.g. one car per row),
    the price column also serves as revenue: total sales = sum of prices."""
    by_role = {role: col for col, role in detect_roles(df).items() if role}
    if "revenue" not in by_role and "quantity" not in by_role and "price" in by_role:
        by_role["revenue"] = by_role["price"]
    return by_role


def _detect(df: pd.DataFrame) -> dict:
    kinds = {c: _kind(df[c]) for c in df.columns}
    claims: dict = {}  # column -> (role, strength, rank)

    def claim(col, role, strength, rank=0):
        if role and role in ROLES and _fits(role, kinds[col]):
            if col not in claims or strength > claims[col][1]:
                claims[col] = (role, strength, rank)

    ambiguous = {}
    for c in df.columns:
        guess = _name_guess(c)
        if isinstance(guess, list):
            ambiguous[c] = guess
        else:
            claim(c, *guess)
        if kinds[c] == "text" and _channel_like(df[c]):  # "Location" holding Online / Takeaway is a channel
            claims[c] = ("channel", EXACT + 1, 0)

    nums = [c for c in df.columns if kinds[c] == "number"]
    for qty, price, total in _equations(df, nums):
        claim(qty, "quantity", EQUATION)
        if claims.get(price, (None,))[0] == "cost":  # qty x cost = what was spent, not revenue
            claims.pop(total, None) if claims.get(total, (None, 0))[1] < EXACT else None
            continue
        claim(price, "price", EQUATION)
        claim(total, "revenue", EQUATION)

    has_stock = any(r == "stock" for r, *_ in claims.values())
    for c, options in ambiguous.items():
        if c in claims:
            continue
        if has_stock and "stock_in" in options:
            claim(c, "stock_in", KEYWORD)
        elif has_stock and "quantity" in options and kinds[c] == "number":
            claim(c, "quantity", KEYWORD)
        elif "quantity" in options and kinds[c] == "number" and _small_ints(df[c]):
            claim(c, "quantity", VALUES)

    if not any(r == "date" for r, *_ in claims.values()):  # an unnamed date column is still the date
        for c in df.columns:
            if kinds[c] == "date" and c not in claims:
                claims[c] = ("date", VALUES, 0)
                break

    # one column per role: strongest evidence, then the name listed first, then the leftmost column
    roles: dict = {c: None for c in df.columns}
    order = list(df.columns)
    by_role: dict = {}
    for c, (role, strength, rank) in claims.items():
        by_role.setdefault(role, []).append((-strength, rank, order.index(c), c))
    for role, cands in by_role.items():
        roles[min(cands)[3]] = role
    return roles

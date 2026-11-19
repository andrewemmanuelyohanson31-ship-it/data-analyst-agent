"""Answer the most frequent question types without the model: parse -> run a tested tool -> template answer.

The 3B model gets these wrong in varied ways (wrong tool, dropped filters, misread tables, invented names), so for
them a deterministic parser is more reliable. It is deliberately strict: every word in the question must be
explained (a known column/value/period, an intent word or filler), otherwise it returns None and the model handles
the question as before.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from agent.question import MONTHS, date_column, mentioned_period, mentioned_values
from agent.tools import _date_like, run_tool

# ---- vocabulary (English + Indonesian) ----

FILLER = set("""
what which who is are was were the a an of in on for from by with did does do has have had to at this that these those
there it its me show tell give our my your all how much many total sum overall per each daily data dataset
generate generated generates make makes made bring brings brought earn earns earned gave gives produce produces get
got placed came come comes pay pays paid s
apa yang mana berapa siapa di dari ke pada dengan untuk ini itu saja ada adalah kah punya kita saya kami toko harian
memberi memberikan menghasilkan dapat sebesar seluruh semua keseluruhan nilai sekarang saat dalam
""".split())
TOP = {"most", "highest", "biggest", "largest", "best", "top", "tertinggi", "terbesar", "terbanyak", "terlaris",
       "teratas", "ramai", "teramai"}
BOTTOM = {"least", "lowest", "fewest", "smallest", "worst", "terendah", "terkecil", "tersedikit", "sepi", "jarang",
          "tersepi"}
PALING = {"tinggi": "top", "besar": "top", "banyak": "top", "laris": "top", "ramai": "top", "laku": "top",
          "sering": "top", "rendah": "bottom", "kecil": "bottom", "sedikit": "bottom", "sepi": "bottom",
          "jarang": "bottom"}
AVG = {"average", "avg", "mean", "rata", "rata-rata", "rerata"}
COUNT_NOUNS = {"orders", "order", "pesanan", "transaksi", "transactions", "transaction", "rows", "baris", "employees",
               "employee", "karyawan", "pegawai", "staff", "customers", "pelanggan", "records"}
COUNT_WORDS = {"jumlah", "banyak", "used", "often", "frequently", "common", "popular", "sering", "digunakan", "dipakai"}
MEASURES = {  # concept: (words in questions, regex for column names)
    "revenue": ({"revenue", "pendapatan", "penjualan", "omzet", "omset", "sales", "income", "pemasukan"},
                r"revenue|pendapatan|omzet|omset|^sales$|penjualan"),
    "quantity": ({"units", "unit", "sold", "sell", "sells", "selling", "terjual", "quantity", "qty", "laku", "laris"},
                 r"quantity|qty|terjual|units|sold"),
    "profit": ({"profit", "laba", "untung", "keuntungan"}, r"profit|laba|untung"),
    "salary": ({"salary", "gaji", "upah", "monthly"}, r"salary|gaji|upah"),
    "age": ({"age", "umur", "usia"}, r"age|umur|usia"),
}
GROUPS = {  # concept: (words, regex for column names)
    "city": ({"city", "cities", "kota"}, r"city|kota"),
    "product": ({"product", "products", "produk", "item", "items", "barang"}, r"product|produk|item|barang"),
    "category": ({"category", "categories", "kategori"}, r"category|kategori"),
    "department": ({"department", "departments", "departemen", "divisi"}, r"department|departemen|divisi"),
    "payment": ({"payment", "pembayaran", "metode", "method"}, r"payment|pembayaran"),
    "channel": ({"channel", "channels", "saluran"}, r"channel|saluran"),
}
MONTH_GROUP = re.compile(r"\b(which|what) month\b|\bbulan (apa|mana)\b", re.I)
DAY_GROUP = re.compile(r"\b(which|what) day\b|\bday of (the )?week\b|\bweekday\b|\bhari (apa|mana)\b", re.I)
RESTOCK = re.compile(r"restock|re-stock|reorder|run(s|ning)? out|kehabisan|\bhabis\b", re.I)
STOCK = {"stok", "stock", "sisa", "persediaan"}
STOCK_WORDS = {"restock", "re-stock", "reorder", "run", "runs", "running", "out", "kehabisan", "habis", "hampir",
               "segera", "harus", "perlu", "need", "needs", "will", "akan", "first", "pertama", "duluan", "dulu",
               "minggu", "week", "soon", "di-restock", "direstock", "stok", "stock", "products", "produk", "product",
               "sisa", "persediaan", "left", "low", "items", "barang", "jumlah", "banyak", "many", "should", "i", "what",
               "which", "do", "apa", "saja", "yang", "mana"}
VETO = re.compile(r"\b(or|atau|vs|versus|compare|bandingkan|percent|percentage|persen|correlation|korelasi|outliers?|"
                  r"missing|kosong|duplicates?|duplikat|trend|tren|forecast|prediksi|ramal|chart|grafik|plot|clean|"
                  r"bersihkan|more than|less than|lebih dari|kurang dari|between|antara|median|without|tanpa|except|"
                  r"kecuali|not|bukan|tidak|left the|keluar)\b|[<>%]", re.I)


def _words(q: str) -> list[str]:
    out = []
    for w in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", q.lower()):
        if w.endswith("nya") and len(w) > 5:
            w = w[:-3]  # pendapatannya -> pendapatan, gajinya -> gaji, stoknya -> stok
        out.append(w)
    return out


def indonesian(q: str) -> bool:
    w = set(_words(q))
    return bool(w & {"apa", "mana", "berapa", "yang", "paling", "dari", "bulan", "hari", "saja"}) and \
        not (w & {"what", "which", "how"})


def _pick(df, regex, numeric):
    cols = [c for c in df.columns if re.search(regex, str(c), re.I)
            and (pd.api.types.is_numeric_dtype(df[c]) == numeric) and not _date_like(df[c])]
    return cols[0] if len(cols) == 1 else None  # none or ambiguous (harga_modal vs harga_jual): let the model decide


def _fmt(v) -> str:
    v = float(v)
    return f"{v:,.0f}" if v.is_integer() else f"{v:,.2f}"


def _name_words(col) -> set:
    return set(str(col).lower().replace("_", " ").split())


@dataclass
class Plan:
    tool: str
    args: dict
    render: Callable


def plan(question: str, df: pd.DataFrame) -> Plan | None:
    """A Plan if this is a frequent question type that the parser fully understands, else None."""
    if VETO.search(question):
        return None
    words = _words(question)
    ws = set(words)
    used = set(FILLER)
    idn = indonesian(question)

    values = mentioned_values(question, df)
    for v in values.values():
        used |= set(_words(v))
    period = mentioned_period(question)
    dc = date_column(df)
    if period:
        if dc is None:
            return None
        used |= {w for w in ws if w in MONTHS or re.fullmatch(r"20\d\d(-\d\d)?", w)} | {"bulan", "tahun", "month", "year"}

    direction = None
    for i, w in enumerate(words):
        if w in TOP or w in BOTTOM:
            direction = "top" if w in TOP else "bottom"
            used.add(w)
        if w == "paling" and i + 1 < len(words) and words[i + 1] in PALING:
            direction = PALING[words[i + 1]]
            used |= {"paling", words[i + 1]}

    measure, measure_word = None, None
    for vocab, regex in MEASURES.values():
        if ws & vocab:
            measure = _pick(df, regex, numeric=True)
            if measure is None:
                return None
            measure_word = str(measure).replace("_", " ")
            used |= (ws & vocab) | _name_words(measure)
            break
    if measure is None:  # a numeric column named outright: "performance score", "training hours"
        q = " ".join(words)
        for c in df.columns:
            name = str(c).lower().replace("_", " ")
            if pd.api.types.is_numeric_dtype(df[c]) and re.search(rf"\b{re.escape(name)}\b", q):
                measure, measure_word = c, name
                used |= _name_words(c)
                break
    avg = bool(ws & AVG)
    used |= ws & AVG

    stock_cols = [c for c in df.columns if re.search(r"stok|stock", str(c), re.I) and pd.api.types.is_numeric_dtype(df[c])]
    if (RESTOCK.search(question) or ws & STOCK) and stock_cols:
        return _stock_plan(question, df, ws, used | STOCK_WORDS, stock_cols[0], values, idn)

    group, group_word = None, None
    if MONTH_GROUP.search(question):
        group, group_word = "month", "bulan" if idn else "month"
        used |= {"month", "bulan"}
    elif DAY_GROUP.search(question):
        group, group_word = "weekday", "hari"
        used |= {"day", "week", "weekday", "hari", "of"}
    else:
        found = []  # (position, column, word)
        for vocab, regex in GROUPS.values():
            hits = [(i, w) for i, w in enumerate(words) if w in vocab]
            if hits:
                c = _pick(df, regex, numeric=False)
                if c is None:
                    return None
                found.append((hits[0][0], c, hits[0][1]))
                used |= {w for _, w in hits} | _name_words(c)
        found.sort()
        if len(found) == 2 and found[1][0] - found[0][0] == 1:
            found = [found[0] if idn else found[1]]  # "kategori produk" = kategori; "product category" = category
        if len(found) > 1:
            return None
        if found:
            _, group, group_word = found[0]
    counting = bool(ws & (COUNT_NOUNS | COUNT_WORDS))
    used |= ws & (COUNT_NOUNS | COUNT_WORDS)

    if group in values:
        if direction is not None:
            return None  # "which city ... Jakarta" is contradictory
        group = None  # "karyawan di departemen Marketing": the group word only describes the filter

    if [w for w in ws - used if not w.isdigit()]:
        return None  # a word we don't understand may change the meaning: let the model handle it

    filters = [{"column": c, "op": "==", "value": v} for c, v in values.items()]
    if period:
        filters.append({"column": dc, "op": "==", "value": period})
    where = filters or None

    if group == "weekday":
        m = measure or _pick(df, MEASURES["revenue"][1], numeric=True)
        if dc is None or m is None or direction is None:
            return None
        return Plan("weekday_pattern", {"date_column": dc, "value_column": m, "where": where},
                    lambda r: _weekday_answer(r, direction, idn))

    if group is not None:
        if direction is None or (measure is None and (not counting or group == "month")):
            return None
        agg = "mean" if avg else ("count" if measure is None else "sum")
        args = {"by": [group], "agg": agg, "where": where}
        if measure is not None:
            args["column"] = measure
        if measure is None:
            what = "jumlah" if idn else "count"
        else:
            what = (f"rata-rata {measure_word}" if idn else f"average {measure_word}") if avg else measure_word
        return Plan("group_aggregate", args, lambda r: _group_answer(r, group, group_word, what, direction, idn, filters))

    if measure is not None:
        agg = "mean" if avg else {"top": "max", "bottom": "min"}.get(direction, "sum")
        return Plan("aggregate", {"agg": agg, "column": measure, "where": where},
                    lambda r: _scalar_answer(r, agg, measure_word, idn, filters, None))
    if counting and direction is None:
        noun = next((w for w in words if w in COUNT_NOUNS), None)
        return Plan("aggregate", {"agg": "count", "where": where},
                    lambda r: _scalar_answer(r, "count", None, idn, filters, noun))
    return None


def _where_text(filters, idn):
    if not filters:
        return ""
    return (" untuk " if idn else " for ") + ", ".join(str(f["value"]) for f in filters)


def _group_answer(r, group, group_word, what, direction, idn, filters):
    t = r.table
    if t is None or t.empty:
        return "Tidak ada data yang cocok." if idn else "No rows match."
    key = t.columns[0]
    val = next(c for c in t.columns[1:] if c != "share_pct")
    row, other = (t.iloc[0], t.iloc[-1]) if direction == "top" else (t.iloc[-1], t.iloc[0])
    share = f" ({row['share_pct']}%)" if "share_pct" in t.columns else ""
    w = _where_text(filters, idn)
    if idn:
        word = "tertinggi" if direction == "top" else "terendah"
        return (f"{str(group_word).capitalize()} dengan {what} {word}{w}: **{row[key]}** = {_fmt(row[val])}{share}. "
                f"{'Terendah' if direction == 'top' else 'Tertinggi'}: {other[key]} = {_fmt(other[val])}.")
    word = "highest" if direction == "top" else "lowest"
    return (f"The {group_word} with the {word} {what}{w} is **{row[key]}**: {_fmt(row[val])}{share}. "
            f"{'Lowest' if direction == 'top' else 'Highest'}: {other[key]} = {_fmt(other[val])}.")


def _weekday_answer(r, direction, idn):
    t = r.table
    val = t.columns[1]
    row = t.loc[t[val].idxmax()] if direction == "top" else t.loc[t[val].idxmin()]
    if idn:
        return (f"Hari {'paling ramai' if direction == 'top' else 'paling sepi'}: **{row.weekday}**, rata-rata "
                f"{_fmt(row[val])} per hari ({row.vs_average_pct:+.1f}% dari rata-rata).")
    return (f"The {'busiest' if direction == 'top' else 'quietest'} day is **{row.weekday}**, averaging "
            f"{_fmt(row[val])} per day ({row.vs_average_pct:+.1f}% vs average).")


def _scalar_answer(r, agg, measure_word, idn, filters, noun):
    w = _where_text(filters, idn)
    if agg == "count":
        n = re.search(r"Count of rows: ([\d,]+)", r.text).group(1)
        thing = noun or ("baris" if idn else "rows")
        return f"Jumlah {thing}{w}: **{n}**." if idn else f"There are **{n}** {thing}{w}."
    value = re.search(r": ([\-\d,]+(?:\.\d+)?)\.?$", r.text).group(1)
    rows = re.search(r"over ([\d,]+) rows", r.text).group(1)
    label = ({"sum": "Total", "mean": "Rata-rata", "max": "Tertinggi", "min": "Terendah"} if idn else
             {"sum": "Total", "mean": "Average", "max": "Highest", "min": "Lowest"})[agg]
    return f"{label} {measure_word}{w}: **{value}** ({rows} {'baris' if idn else 'rows'})."


def _stock_plan(question, df, ws, used, stock_col, values, idn):
    dc = date_column(df)
    product = _pick(df, GROUPS["product"][1], numeric=False)
    qty = _pick(df, MEASURES["quantity"][1], numeric=True)
    if dc is None or product is None or qty is None:
        return None
    if [w for w in ws - used if not w.isdigit()]:
        return None
    one = values.get(product)
    args = {"date_column": dc, "product_column": product, "quantity_column": qty, "stock_column": stock_col}
    how_many = bool(re.search(r"\bhow many\b|\bberapa (banyak|jumlah)\b", question, re.I))
    first = bool(ws & {"first", "pertama", "duluan", "dulu"})

    def render(r):
        t = r.table
        soon = t[t.status != "ok"]
        if one is not None:
            row = t[t["product"] == one].iloc[0]
            days = "∞" if row.days_left == float("inf") else f"{row.days_left:g}"
            return (f"Sisa stok **{one}** sekarang: **{row.stock_now:g}** (terjual sekitar {row.avg_sold_per_day:g} per "
                    f"hari, cukup untuk {days} hari)." if idn else
                    f"**{one}** has **{row.stock_now:g}** in stock (sells about {row.avg_sold_per_day:g} per day, "
                    f"{days} days left).")
        listing = ", ".join(f"{p} ({d:g} {'hari' if idn else 'days'})" for p, d in zip(soon["product"], soon.days_left))
        head = t.iloc[0]
        if first:
            return (f"Yang paling cepat habis: **{head['product']}** (stok {head.stock_now:g}, {head.days_left:g} hari lagi)."
                    if idn else f"First to run out: **{head['product']}** (stock {head.stock_now:g}, "
                    f"{head.days_left:g} days left).")
        if soon.empty:
            return "Tidak ada produk yang habis dalam 7 hari." if idn else "No product runs out within 7 days."
        if how_many:
            return (f"**{len(soon)}** produk perlu segera di-restock (habis dalam 7 hari): {listing}." if idn else
                    f"**{len(soon)}** products need restocking within 7 days: {listing}.")
        return f"Produk yang perlu segera di-restock: {listing}." if idn else f"Restock soon: {listing}."

    return Plan("restock", args, render)


def answer(question: str, df: pd.DataFrame):
    """(text, tool, args, ToolResult) or None. Tool errors also return None so the model can try instead."""
    p = plan(question, df)
    if p is None:
        return None
    args = {k: v for k, v in p.args.items() if v is not None}
    try:
        result = run_tool(p.tool, df, args)
        return p.render(result), p.tool, args, result
    except Exception:  # noqa: BLE001 - any failure here just means "let the model handle it"
        return None

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
from agent.roles import detect_roles
from agent.tools import run_tool

# ---- vocabulary (English + Indonesian) ----

FILLER = set("""
what which who is are was were the a an of in on for from by with did does do has have had to at this that these those
there it its me show tell give our my your all how much many total sum overall per each daily data dataset
generate generated generates make makes made bring brings brought earn earns earned gave gives produce produces get
got placed came come comes pay pays paid s
apa yang mana berapa siapa di dari ke pada dengan untuk ini itu saja ada adalah kah punya kita saya kami toko harian
memberi memberikan menghasilkan dapat sebesar seluruh semua keseluruhan nilai sekarang saat dalam
""".split())
TOP = {"most", "highest", "biggest", "largest", "best", "top", "busiest", "tertinggi", "terbesar", "terbanyak",
       "terlaris", "teratas", "ramai", "teramai"}
BOTTOM = {"least", "lowest", "fewest", "smallest", "worst", "quietest", "slowest", "terendah", "terkecil",
          "tersedikit", "sepi", "jarang", "tersepi"}
PALING = {"tinggi": "top", "besar": "top", "banyak": "top", "laris": "top", "ramai": "top", "laku": "top",
          "sering": "top", "rendah": "bottom", "kecil": "bottom", "sedikit": "bottom", "sepi": "bottom",
          "jarang": "bottom"}
AVG = {"average", "avg", "mean", "rata", "rata-rata", "rerata"}
COUNT_NOUNS = {"orders", "order", "pesanan", "transaksi", "transactions", "transaction", "rows", "baris", "employees",
               "employee", "karyawan", "pegawai", "staff", "customers", "pelanggan", "records"}
COUNT_WORDS = {"jumlah", "banyak", "used", "often", "frequently", "common", "popular", "sering", "digunakan", "dipakai"}
# Question words -> column role. Which column has that role comes from agent/roles.py (names + values), so the
# parser works on any export, not just files whose columns happen to be called "revenue" or "produk".
MEASURES = {
    "revenue": {"revenue", "pendapatan", "penjualan", "omzet", "omset", "sales", "income", "pemasukan"},
    "quantity": {"units", "unit", "sold", "sell", "sells", "selling", "terjual", "quantity", "qty", "laku", "laris"},
    "profit": {"profit", "laba", "untung", "keuntungan"},
    "cost": {"cost", "costs", "modal", "hpp"},
    "discount": {"discount", "discounts", "diskon", "potongan"},
    "shipping": {"shipping", "ongkir"},
    "salary": {"salary", "gaji", "upah"},
    "age": {"age", "umur", "usia"},
}
# question words that are verbs/adjectives: show them as a noun in the answer
DISPLAY = {w: {"en": "units sold", "id": "terjual"} for w in ("laris", "laku", "sold", "sell", "sells", "selling", "unit", "units")}
GROUPS = {
    "city": {"city", "cities", "kota"},
    "product": {"product", "products", "produk", "item", "items", "barang", "menu"},
    "category": {"category", "categories", "kategori"},
    "department": {"department", "departments", "departemen", "divisi"},
    "payment": {"payment", "pembayaran", "metode", "method"},
    "channel": {"channel", "channels", "saluran"},
    "store": {"store", "stores", "outlet", "outlets", "cabang", "branch", "branches", "gerai"},
    "customer": {"customer", "customers", "pelanggan", "pembeli", "buyer", "buyers"},
    "supplier": {"supplier", "suppliers", "pemasok", "vendor", "vendors"},
    "cashier": {"cashier", "cashiers", "kasir"},
}
HOW_MANY = re.compile(r"\bhow many\b|\bberapa (banyak|jumlah|kali)\b", re.I)
MONTH_GROUP = re.compile(r"\b(which|what) month\b|\bbulan (apa|mana)\b", re.I)
DAY_GROUP = re.compile(r"\b(which|what) day\b|\bday of (the )?week\b|\bweekday\b|\bhari (apa|mana)\b", re.I)
RESTOCK = re.compile(r"restock|re-stock|reorder|run(s|ning)? out|kehabisan|\bhabis\b", re.I)
STOCK = {"stok", "stock", "sisa", "persediaan"}
STOCK_WORDS = {"restock", "re-stock", "reorder", "run", "runs", "running", "out", "kehabisan", "habis", "hampir",
               "segera", "harus", "perlu", "need", "needs", "will", "akan", "first", "pertama", "duluan", "dulu",
               "minggu", "week", "soon", "di-restock", "direstock", "stok", "stock", "products", "produk", "product",
               "sisa", "persediaan", "left", "low", "items", "barang", "jumlah", "banyak", "many", "should", "i", "what",
               "which", "do", "apa", "saja", "yang", "mana"}
CHART = re.compile(r"\b(chart|graph|plot|grafik|diagram|visuali[sz]e|visualisasi|trends?|tren)\b", re.I)
FREQ = [  # (regex, x for the chart tool): first match wins; no match = monthly
    (re.compile(r"\b(daily|harian|per (hari|day)|each day)\b", re.I), None),  # None = the date column itself
    (re.compile(r"\b(weekly|mingguan|per (minggu|week))\b", re.I), "week"),
    (re.compile(r"\b(yearly|annual|tahunan|per (tahun|year))\b", re.I), "year"),
    (re.compile(r"\b(monthly|bulanan|per (bulan|month))\b", re.I), "month"),
]
CHART_WORDS = set("""chart charts graph plot grafik diagram visualize visualise visualisasi trend trends tren line bar as draw
buat buatkan tampilkan gambar gambarkan over time monthly bulanan bulan month weekly mingguan minggu week daily harian
hari day yearly annual tahunan tahun year""".split())
VETO = re.compile(r"\b(or|atau|vs|versus|compare|bandingkan|percent|percentage|persen|correlation|korelasi|outliers?|"
                  r"missing|kosong|duplicates?|duplikat|forecast|prediksi|ramal|clean|"
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
    return bool(w & {"apa", "mana", "berapa", "yang", "paling", "dari", "bulan", "hari", "saja", "grafik", "buat",
                     "buatkan", "tampilkan", "bulanan", "mingguan", "harian", "tahunan"}) and \
        not (w & {"what", "which", "how"})


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

    col_of = {role: c for c, role in detect_roles(df).items() if role}
    values = mentioned_values(question, df)
    for v in values.values():
        used |= set(_words(v))
    period = mentioned_period(question)
    dc = col_of.get("date") or date_column(df)
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
    for role, vocab in MEASURES.items():
        if ws & vocab and col_of.get(role):
            measure = col_of[role]
            word = sorted(ws & vocab)[0]  # say it the way the question did: "profit", not the column "laba"
            measure_word = DISPLAY.get(word, {}).get("id" if idn else "en", word) if word in DISPLAY else word
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
    # "how many orders had a discount" counts rows that have it, it isn't the sum of the discount column;
    # "how many units were sold" is fine: a quantity is already a count
    if measure is not None and HOW_MANY.search(question) and measure != col_of.get("quantity"):
        return None

    if (RESTOCK.search(question) or ws & STOCK) and col_of.get("stock"):
        return _stock_plan(question, ws, used | STOCK_WORDS, col_of, values, idn)

    group, group_word = None, None
    if MONTH_GROUP.search(question):
        group, group_word = "month", "bulan" if idn else "month"
        used |= {"month", "bulan"}
    elif DAY_GROUP.search(question):
        group, group_word = "weekday", "hari"
        used |= {"day", "week", "weekday", "hari", "of"}
    else:
        found = []  # (position, column, word)
        for role, vocab in GROUPS.items():
            hits = [(i, w) for i, w in enumerate(words) if w in vocab]
            if hits and col_of.get(role):  # no such column here: the word may still be a value ("Store" orders)
                c = col_of[role]
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
    if CHART.search(question):
        return _chart_plan(question, col_of, ws, used | CHART_WORDS, group, group_word, measure, measure_word, avg,
                           values, period, dc, idn)

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
        m = measure or col_of.get("revenue")
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


def _chart_plan(question, col_of, ws, used, group, group_word, measure, measure_word, avg, values, period, dc, idn):
    """'monthly revenue trend' -> line chart by month; 'revenue by city chart' -> bar chart by city."""
    if [w for w in ws - used if not w.isdigit()] or group in values:
        return None
    if measure is None:  # "show the sales trend" / "grafik penjualan": default to the revenue column
        measure = col_of.get("revenue")
        measure_word = "pendapatan" if idn else "revenue"
    if measure is None:
        return None
    filters = [{"column": c, "op": "==", "value": v} for c, v in values.items()]
    if period:
        filters.append({"column": dc, "op": "==", "value": period})
    agg = "mean" if avg else "sum"
    time_x = next((x for rx, x in FREQ if rx.search(question)), "month")
    if group not in (None, "month", "weekday") and not any(rx.search(question) for rx, _ in FREQ):
        kind, x = "bar", group
        title = (f"{measure_word} per {group_word}") if idn else f"{measure_word} by {group_word}"
    else:
        if dc is None:
            return None
        kind, x = "line", time_x or dc
        period_word = {"week": ("mingguan", "weekly"), "year": ("tahunan", "yearly"), "month": ("bulanan", "monthly")}
        title = (f"{measure_word} {period_word.get(time_x, ('harian', 'daily'))[0]}" if idn else
                 f"{period_word.get(time_x, ('harian', 'daily'))[1]} {measure_word}")
    args = {"kind": kind, "x": x, "y": measure, "agg": agg, "where": filters or None}

    def render(r):
        t = r.table
        if t is None or t.empty:
            return "Tidak ada data yang cocok." if idn else "No rows match."
        key, val = t.columns[0], t.columns[1]
        hi, lo = t.loc[t[val].idxmax()], t.loc[t[val].idxmin()]
        w = _where_text(filters, idn)
        if idn:
            return (f"Grafik {'garis' if kind == 'line' else 'batang'} {title}{w}. Tertinggi: **{hi[key]}** = "
                    f"{_fmt(hi[val])}; terendah: {lo[key]} = {_fmt(lo[val])}.")
        return (f"{'Line' if kind == 'line' else 'Bar'} chart of {title}{w}. Highest: **{hi[key]}** = {_fmt(hi[val])}; "
                f"lowest: {lo[key]} = {_fmt(lo[val])}.")

    return Plan("chart", args, render)


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


def _stock_plan(question, ws, used, col_of, values, idn):
    dc, product, qty, stock_col = (col_of.get(r) for r in ("date", "product", "quantity", "stock"))
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

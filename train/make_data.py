"""Build fine-tuning data: question -> tool call -> tool result -> short answer, in the app's exact chat format.

    python train/make_data.py --n 1500          -> train/data/train.jsonl, train/data/dev.jsonl

Every example runs on a freshly generated random table (random column names from agent/column_names.py, English
or Indonesian questions, sometimes typos), never on the eval datasets, so eval/tasks.json stays a fair test.
An example is kept only if (1) the tool result matches a separate pandas calculation and (2) every number in the
written answer appears in the tool output (the app's own grounding check).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from agent.column_names import ROLES  # noqa: E402
from agent.loop import SYSTEM, _for_model, schema_summary  # noqa: E402
from agent.numbers import ungrounded_numbers  # noqa: E402
from agent.tools import run_tool, tool_schemas  # noqa: E402
from run_eval import is_correct  # noqa: E402

R = random.Random()
# words people use for each measure / group (beyond the column name itself): (English, Indonesian)
SAY = {"revenue": (["revenue", "sales", "income", "turnover"], ["pendapatan", "penjualan", "omzet"]),
       "quantity": (["units sold", "quantity", "items sold"], ["jumlah terjual", "barang terjual", "unit terjual"]),
       "price": (["price", "unit price"], ["harga"]), "profit": (["profit"], ["laba", "keuntungan"]),
       "salary": (["salary", "pay"], ["gaji"]), "age": (["age"], ["umur", "usia"]),
       "experience": (["years of experience", "experience"], ["pengalaman", "masa kerja"]),
       "performance": (["performance score", "rating"], ["nilai kinerja", "skor"]),
       "weight": (["weight"], ["berat"]), "height": (["height"], ["tinggi"]),
       "tickets": (["tickets sold"], ["tiket terjual"]), "box_office": (["box office", "gross"], ["pendapatan box office"]),
       "product": (["product", "item"], ["produk", "barang"]), "category": (["category"], ["kategori"]),
       "city": (["city"], ["kota"]), "store": (["store", "branch", "outlet"], ["toko", "cabang"]),
       "payment": (["payment method"], ["metode pembayaran"]), "brand": (["brand", "make"], ["merek"]),
       "department": (["department", "division"], ["departemen", "divisi"]), "gender": (["gender"], ["jenis kelamin"]),
       "size": (["size"], ["ukuran"]), "channel": (["channel"], ["channel"])}
EXTRA_NAMES = {"weight": ["weight", "berat", "weight_kg"], "height": ["height", "tinggi", "height_cm"],
               "size": ["size", "ukuran"], "tickets": ["tickets_sold", "tiket_terjual"],
               "box_office": ["total_box_office", "gross", "box_office"]}
POOL = {"product": ["Kopi Susu", "Teh Manis", "Roti Bakar", "Nasi Goreng", "Mie Ayam", "Es Jeruk", "Sneakers", "Kaos",
                    "Jaket", "Earbuds", "Power Bank", "Beras 5kg", "Minyak Goreng", "Gula Pasir"],
        "category": ["Minuman", "Makanan", "Fashion", "Elektronik", "Sembako", "Snack"],
        "city": ["Jakarta", "Bandung", "Surabaya", "Medan", "Semarang", "Makassar", "Denpasar", "Yogyakarta"],
        "store": ["Cabang Utama", "Kemang", "Dago", "Tunjungan", "Senopati", "Margonda"],
        "payment": ["Cash", "QRIS", "Debit", "Credit Card", "GoPay", "OVO", "Bank Transfer"],
        "brand": ["Toyota", "Honda", "Ford", "Hyundai", "Suzuki", "Daihatsu", "Mitsubishi"],
        "channel": ["Online", "Store", "GoFood", "GrabFood", "Shopee", "Tokopedia"],
        "department": ["Engineering", "Sales", "Finance", "HR", "Marketing", "Operations"],
        "gender": ["Male", "Female"], "size": ["S", "M", "L", "XL", "XXL"]}
MEASURES = ("revenue", "quantity", "price", "profit", "salary", "age", "experience", "performance", "weight",
            "height", "tickets", "box_office")
ADDITIVE = ("revenue", "quantity", "profit", "tickets", "box_office")
AGG_WORDS = {"sum": (["total", "sum of"], ["total", "jumlah"]), "mean": (["average", "mean"], ["rata-rata", "rerata"]),
             "max": (["highest", "maximum", "largest"], ["tertinggi", "terbesar", "maksimum"]),
             "min": (["lowest", "minimum", "smallest"], ["terendah", "terkecil", "minimum"]),
             "median": (["median"], ["median"])}


def _name(role: str) -> str:
    n = R.choice(EXTRA_NAMES.get(role) or ROLES[role]["names"][:10])
    style = R.random()
    return n if style < .5 else n.replace("_", " ").title() if style < .8 else n.upper() if style < .9 else n.replace("_", "")


def make_table():
    """(DataFrame, {role: column}, kind) for one of four kinds of spreadsheet."""
    kind = R.choice(["sales", "sales", "hr", "people", "yearly"])
    n = R.randint(150, 900)
    if kind == "yearly":
        years = list(range(R.randint(1990, 2005), R.randint(2018, 2025) + 1))
        roles = {"year": _name("year"), "tickets": _name("tickets"), "box_office": _name("box_office")}
        t = np.array([R.uniform(1e8, 1.5e9) for _ in years])
        df = pd.DataFrame({roles["year"]: years, roles["tickets"]: t.round(0),
                           roles["box_office"]: (t * R.uniform(6, 9)).round(0)})
        return (df, roles, kind) if df.columns.is_unique else make_table()
    groups = {"sales": ["product", "category", "city", "store", "payment", "brand", "channel"],
              "hr": ["department", "gender", "city"], "people": ["size", "gender"]}[kind]
    picked = R.sample(groups, k=min(len(groups), R.randint(2, 4)))
    roles = {g: _name(g) for g in picked}
    data = {roles[g]: [R.choice(POOL[g][:R.randint(min(3, len(POOL[g])), len(POOL[g]))]) for _ in range(n)] for g in picked}
    if kind == "sales":
        roles["date"] = _name("date")
        start = pd.Timestamp(f"{R.choice([2023, 2024, 2025])}-01-01")
        data[roles["date"]] = sorted((start + pd.to_timedelta(np.random.randint(0, 365 * R.choice([1, 2]), n), "D"))
                                     .strftime("%Y-%m-%d"))
        for r in ("quantity", "price", "revenue"):
            roles[r] = _name(r)
        q = np.random.randint(1, 8, n)
        p = np.random.choice([5000, 12000, 18000, 25000, 150000, 350000, 1200000], n) * 1.0
        data[roles["quantity"]], data[roles["price"]], data[roles["revenue"]] = q, p, q * p
        if R.random() < .5:
            roles["profit"] = _name("profit")
            data[roles["profit"]] = (q * p * np.random.uniform(.1, .35, n)).round(0)
    elif kind == "hr":
        for r in ("age", "salary", "experience", "performance"):
            roles[r] = _name(r)
        data[roles["age"]] = np.random.randint(21, 60, n)
        data[roles["salary"]] = np.random.randint(40, 300, n) * 100000.0
        data[roles["experience"]] = np.random.randint(0, 30, n)
        data[roles["performance"]] = np.random.randint(1, 6, n)
    else:
        for r in ("age", "weight", "height"):
            roles[r] = _name(r)
        data[roles["age"]] = np.random.randint(16, 70, n)
        data[roles["weight"]] = np.random.randint(40, 120, n)
        data[roles["height"]] = np.random.randint(145, 195, n)
    if len(set(roles.values())) < len(roles):  # two roles drew the same name
        return make_table()
    return pd.DataFrame(data), roles, kind


def say(role, col, idn):
    words = SAY.get(role, ([col], [col]))[idn]
    return col.replace("_", " ").lower() if R.random() < .3 else R.choice(words)  # sometimes the column's own name


def typo(q):
    """Sometimes swap two letters or drop one inside a longer word."""
    words = q.split()
    long = [i for i, w in enumerate(words) if len(w) > 5 and w.isalpha()]
    if not long or R.random() > .12:
        return q
    i = R.choice(long)
    w, j = words[i], R.randint(1, len(words[i]) - 3)
    words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:] if R.random() < .5 else w[:j] + w[j + 1:]
    return " ".join(words)


def casual(q):
    q = q if R.random() > .2 else q.lower()
    return q if R.random() > .2 else q.rstrip("?.")


# ---- intents: each returns (question, tool, args, truth, answer writer) or None ----------------------------------
def groups_of(roles):
    return [r for r in roles if r in POOL]


def one_number(df, roles, kind, idn):
    m = R.choice([r for r in roles if r in MEASURES])
    col = roles[m]
    agg = R.choice(["sum", "mean", "max", "min", "median"] if m in ADDITIVE else ["mean", "max", "min", "median"])
    where, cond, d = [], "", df
    if groups_of(roles) and R.random() < .6:
        gr = R.choice(groups_of(roles))
        v = R.choice(sorted(df[roles[gr]].unique()))
        where.append({"column": roles[gr], "op": "==", "value": v})
        cond = R.choice([f" untuk {v}", f" di {v}"] if idn else [f" for {v}", f" in {v}"])
        d = d[d[roles[gr]] == v]
    if "date" in roles and R.random() < .3:
        y = R.choice(sorted(df[roles["date"]].str[:4].unique()))
        where.append({"column": roles["date"], "op": "==", "value": y})
        cond += f" tahun {y}" if idn else f" in {y}"
        d = d[d[roles["date"]].str[:4] == y]
    if d.empty:
        return None
    a, w = R.choice(AGG_WORDS[agg][idn]), say(m, col, idn)
    if kind == "yearly" and agg == "mean" and R.random() < .5:  # "average X per year": every row is a year, no filter
        cond += " per tahun" if idn else " per year"
    if m == "price" and agg in ("min", "max") and not idn and R.random() < .5:
        q = f"What is the {'cheapest' if agg == 'min' else 'most expensive'} price{cond}?"
    else:
        q = R.choice([f"Berapa {a} {w}{cond}?", f"{a.capitalize()} {w}{cond} berapa?"] if idn else
                     [f"What is the {a} {w}{cond}?", f"What's the {a} {w}{cond}?", f"Show me the {a} {w}{cond}."])
    args = {"agg": agg, "column": col, **({"where": where} if where else {})}

    def write(res_text):
        num = res_text.split(": ")[-1].rstrip(".")
        return f"{a.capitalize()} {w}{cond} adalah **{num}**." if idn else f"The {a} {w}{cond} is **{num}**."
    return q, "aggregate", args, getattr(d[col], agg)(), write


def count_rows(df, roles, kind, idn):
    g, m = groups_of(roles), [r for r in roles if r in MEASURES]
    nn = R.choice({"sales": (["orders", "transactions", "sales"], ["transaksi", "pesanan"]),
                   "hr": (["employees", "people", "staff"], ["karyawan", "pegawai"]),
                   "people": (["people", "customers"], ["orang", "pelanggan"]),
                   "yearly": (["years"], ["tahun"])}[kind][idn])
    if m and (not g or R.random() < .5):
        mm = R.choice(m)
        col = roles[mm]
        thr = float(np.percentile(df[col], R.choice([25, 50, 75])))
        digits = len(str(int(abs(thr)))) - 2  # a round threshold, like people write: 949,204,562 -> 950,000,000
        thr = int(round(thr, -digits)) if digits > 0 else int(round(thr))
        op = R.choice([">", "<"])
        word = ({">": "lebih dari", "<": "kurang dari"} if idn else {">": "more than", "<": "less than"})[op]
        w = say(mm, col, idn)
        q = f"Ada berapa {nn} dengan {w} {word} {thr:,}?" if idn else f"How many {nn} have {w} {word} {thr:,}?"
        where = [{"column": col, "op": op, "value": thr}]
        truth = int((df[col] > thr).sum() if op == ">" else (df[col] < thr).sum())
    else:
        gr = R.choice(g)
        v = R.choice(sorted(df[roles[gr]].unique()))
        q = R.choice([f"Berapa banyak {nn} di {v}?", f"Ada berapa {nn} {v}?"] if idn else
                     [f"How many {nn} are in {v}?", f"How many {nn} for {v}?", f"Number of {nn} with {v}?"])
        where = [{"column": roles[gr], "op": "==", "value": v}]
        truth = int((df[roles[gr]] == v).sum())

    def write(res_text):
        num = res_text.split("Count of rows: ")[-1].split(" ")[0]
        return f"Ada **{num}** {nn}." if idn else f"There are **{num}** {nn}."
    return q, "aggregate", {"agg": "count", "where": where}, truth, write


def best_group(df, roles, kind, idn):
    g, m = groups_of(roles), [r for r in roles if r in MEASURES]
    if not g or not m:
        return None
    gr, mm = R.choice(g), R.choice(m)
    gcol, mcol = roles[gr], roles[mm]
    agg = "sum" if mm in ADDITIVE and R.random() < .75 else "mean"
    low = R.random() < .35
    gw, mw = say(gr, gcol, idn), say(mm, mcol, idn)
    if agg == "sum":
        q = R.choice([f"{gw.capitalize()} mana yang {mw}-nya paling {'kecil' if low else 'besar'}?",
                      f"{gw.capitalize()} apa dengan total {mw} {'terendah' if low else 'tertinggi'}?"] if idn else
                     [f"Which {gw} has the {'lowest' if low else 'highest'} total {mw}?",
                      f"Which {gw} brings in the {'least' if low else 'most'} {mw}?"])
        if mm == "quantity" and not low and R.random() < .5:
            q = f"{gw.capitalize()} apa yang paling laris?" if idn else f"What is the best-selling {gw}?"
    else:
        q = (f"{gw.capitalize()} mana yang rata-rata {mw}-nya paling {'rendah' if low else 'tinggi'}?" if idn else
             f"Which {gw} has the {'lowest' if low else 'highest'} average {mw}?")
    s = getattr(df.groupby(gcol)[mcol], agg)()

    def write(res_text):
        name, num = res_text.split("Lowest: " if low else "Highest: ")[-1].split(". ")[0].rstrip(".").rsplit(" = ", 1)
        how = ("total " if agg == "sum" else "rata-rata ") if idn else ("total " if agg == "sum" else "average ")
        return (f"{gw.capitalize()} dengan {how}{mw} {'terendah' if low else 'tertinggi'} adalah **{name}** ({num})." if idn
                else f"The {gw} with the {'lowest' if low else 'highest'} {how}{mw} is **{name}** ({num}).")
    truth = (s.idxmin(), s.min()) if low else (s.idxmax(), s.max())
    return q, "group_aggregate", {"by": [gcol], "column": mcol, "agg": agg}, truth, write


def most_common(df, roles, kind, idn):
    if not groups_of(roles):
        return None
    gr = R.choice(groups_of(roles))
    col = roles[gr]
    gw = say(gr, col, idn)
    who = {"sales": ("customers", "pelanggan"), "hr": ("employees", "karyawan"), "people": ("people", "orang")}[kind][idn]
    chosen = gr in ("payment", "product", "brand", "channel", "size")  # things people pick; not gender or city
    q = R.choice([f"{gw.capitalize()} apa yang paling sering muncul?"] + ([f"{gw.capitalize()} apa yang paling banyak dipilih {who}?"] if chosen else [])
                 if idn else [f"What is the most common {gw}?", f"Which {gw} appears most often?"] + ([f"Which {gw} do {who} prefer?"] if chosen else []))
    vc = df[col].value_counts()

    def write(res_text):
        part = res_text.split("most common: ")[-1].split(";")[0]
        return f"{gw.capitalize()} yang paling sering: **{part}**." if idn else f"The most common {gw} is **{part}**."
    return q, "value_counts", {"column": col}, (vc.idxmax(), int(vc.max())), write


INTENTS = [one_number, one_number, count_rows, best_group, best_group, most_common]


def example(seed: int):
    R.seed(seed)
    np.random.seed(seed)
    df, roles, kind = make_table()
    idn = R.random() < .45
    made = R.choice(INTENTS)(df, roles, kind, idn)
    if made is None:
        return None
    q, tool, args, truth, write = made
    args = json.loads(json.dumps(args, default=lambda o: o.item() if hasattr(o, "item") else str(o)))  # plain JSON
    q = casual(typo(q))
    try:
        res = run_tool(tool, df, args)
        content, final = _for_model(res), write(res.text)
    except Exception:  # noqa: BLE001 - a bad generated case is simply dropped
        return None
    if not all(is_correct(final, t) for t in (truth if isinstance(truth, tuple) else (truth,))) \
            or ungrounded_numbers(final, q + "\n" + content):
        return None  # the tool or the writer got it wrong: never train on it
    return {"messages": [
        {"role": "system", "content": SYSTEM.format(schema=schema_summary(df))},
        {"role": "user", "content": q},
        {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": tool, "arguments": args}}]},
        {"role": "tool", "content": content},
        {"role": "assistant", "content": final}]}


NL = "\n"
HEAD = ("\n\n\nYou may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n<tools>")
TAIL = ("\n</tools>\n\nFor each function call, return a json object with function name and arguments within "
        "<tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": "
        "<args-json-object>}\n</tool_call><|im_end|>\n")


def render(messages, tools) -> str:
    """The exact prompt text Ollama builds for qwen2.5:3b (its TEMPLATE), so training sees what the app sends."""
    out = "<|im_start|>system\n" + messages[0]["content"] + HEAD
    out += "".join('\n{"type": "function", "function": ' + json.dumps(t["function"], ensure_ascii=False) + "}"
                   for t in tools) + TAIL
    for m in messages[1:]:
        if m["role"] == "user":
            out += "<|im_start|>user\n" + m["content"] + "<|im_end|>\n"
        elif m["role"] == "tool":
            out += "<|im_start|>user\n<tool_response>\n" + m["content"] + "\n</tool_response><|im_end|>\n"
        elif m.get("tool_calls"):
            calls = "".join('{"name": "' + c["function"]["name"] + '", "arguments": '
                            + json.dumps(c["function"]["arguments"], ensure_ascii=False) + "}\n" for c in m["tool_calls"])
            out += "<|im_start|>assistant\n<tool_call>\n" + calls + "</tool_call><|im_end|>\n"
        else:
            out += "<|im_start|>assistant\n" + m["content"] + "<|im_end|>\n"
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1500)
    p.add_argument("--dev", type=float, default=0.05)
    a = p.parse_args()
    out = ROOT / "train" / "data"
    out.mkdir(parents=True, exist_ok=True)
    tools = tool_schemas()
    rows, seed, dropped = [], 0, 0
    while len(rows) < a.n:
        seed += 1
        ex = example(seed)
        rows.append(ex) if ex else None
        dropped += ex is None
    n_dev = int(len(rows) * a.dev)
    for name, part in (("dev", rows[:n_dev]), ("train", rows[n_dev:])):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps({**r, "text": render(r["messages"], tools)}, ensure_ascii=False) + "\n" for r in part)
    print(f"{len(rows) - n_dev} train + {n_dev} dev examples ({dropped} dropped by verification) -> {out}")


if __name__ == "__main__":
    main()

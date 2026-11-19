"""Analysis tools the agent can call. Plain pandas functions: tested code does the maths, the model only picks."""
from __future__ import annotations

import inspect
import io
import re
from dataclasses import dataclass
from typing import Callable

import matplotlib

matplotlib.use("Agg")  # no GUI backend; charts are rendered to PNG bytes
import matplotlib.pyplot as plt
import matplotlib.ticker  # noqa: F401  (axis number format)
import numpy as np
import pandas as pd


class ToolError(Exception):
    """Bad arguments or data. The message is shown to the model so it can fix its call."""


@dataclass
class ToolResult:
    text: str
    table: pd.DataFrame | None = None
    chart: bytes | None = None
    new_df: pd.DataFrame | None = None  # set by cleaning tools: becomes a new dataset version


# ---------- helpers ----------

def load_file(name: str, data: bytes) -> pd.DataFrame:
    from agent.tidy import tidy  # local import: tidy is optional for the tools themselves

    if name.lower().endswith((".xlsx", ".xls")):
        return tidy(pd.read_excel(io.BytesIO(data)))
    for enc in ("utf-8-sig", "latin-1"):
        try:
            text = data.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    first = text.splitlines()[0] if text else ""
    sep = max([",", ";", "\t", "|"], key=first.count)
    return tidy(pd.read_csv(io.StringIO(text), sep=sep))


def col(df: pd.DataFrame, name) -> str:
    if isinstance(name, list) and len(name) == 1:  # small models wrap single columns in a list
        name = name[0]
    if isinstance(name, str) and name in df.columns:
        return name
    lookup = {str(c).strip().lower(): c for c in df.columns}
    key = str(name).strip().lower()
    if key in lookup:
        return lookup[key]
    raise ToolError(f"Column '{name}' not found. Columns are: {', '.join(map(str, df.columns))}")


def _cols(df, names) -> list[str]:
    if names is None:
        return []
    if isinstance(names, str):
        names = [n.strip() for n in names.split(",")]
    return [col(df, n) for n in names]


def _numeric(df, name) -> str:
    c = col(df, name)
    if not pd.api.types.is_numeric_dtype(df[c]):
        raise ToolError(f"Column '{c}' is not numeric (type {df[c].dtype}). Use convert_types first or pick a numeric column.")
    return c


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _date_like(s: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    sample = s.dropna().astype(str).head(20)
    return not pd.api.types.is_numeric_dtype(s) and len(sample) > 0 and sample.str.match(r"\d{4}-\d{2}-\d{2}").all()


def apply_where(df: pd.DataFrame, where) -> pd.DataFrame:
    if not where:
        return df
    if isinstance(where, dict):
        where = [where]
    mask = pd.Series(True, index=df.index)
    for cond in where:
        try:
            c, op, v = col(df, cond["column"]), cond.get("op", "=="), cond.get("value")
        except (KeyError, TypeError, AttributeError):
            raise ToolError("Each where condition needs {column, op, value}.")
        s = df[c]
        if op in ("==", "=") and re.fullmatch(r"\d{4}(-\d{2})?", str(v)) and _date_like(s):
            mask &= s.astype(str).str.startswith(str(v)).fillna(False)  # "2025" / "2025-03" = whole year / month
            continue
        if pd.api.types.is_datetime64_any_dtype(s) and op not in ("contains", "in"):
            v = pd.to_datetime(str(v))
        elif pd.api.types.is_numeric_dtype(s) and op not in ("contains", "in"):
            v = _num(v)
        text_col = not pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_datetime64_any_dtype(s)
        if op in ("==", "="):
            m = s.astype(str).str.lower() == str(v).lower() if text_col else s == v
        elif op == "!=":
            m = s.astype(str).str.lower() != str(v).lower() if text_col else s != v
        elif op in (">", ">=", "<", "<="):
            try:
                m = {">": s > v, ">=": s >= v, "<": s < v, "<=": s <= v}[op]
            except TypeError:
                raise ToolError(f"Cannot compare column '{c}' with {v!r}.")
        elif op == "contains":
            m = s.astype(str).str.contains(str(v), case=False, na=False, regex=False)
        elif op == "in":
            vals = v if isinstance(v, list) else [x.strip() for x in str(v).split(",")]
            m = s.astype(str).str.lower().isin([str(x).lower() for x in vals])
        else:
            raise ToolError(f"Unknown op '{op}'. Use one of == != > >= < <= contains in.")
        mask &= m.fillna(False)
    return df[mask]


def _no_rows_hint(df: pd.DataFrame, where) -> str:
    """Why a filter matched nothing, in words the model can act on."""
    conds = [where] if isinstance(where, dict) else list(where or [])
    hints, eq = [], {}
    for cond in conds:
        try:
            c = col(df, cond["column"])
        except (ToolError, KeyError, TypeError):
            continue
        op, v = cond.get("op", "=="), cond.get("value")
        if op in ("==", "="):
            eq.setdefault(c, set()).add(str(v).lower())
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            hints.append(f"'{c}' ranges from {_num(s.min()):g} to {_num(s.max()):g}")
        elif op in ("==", "=") and not s.astype(str).str.lower().str.startswith(str(v).lower()).any():
            sample = ", ".join(map(str, s.dropna().astype(str).unique()[:8]))
            hints.append(f"'{c}' has no value {v!r} (it has: {sample}{', …' if s.nunique() > 8 else ''})")
    for c, vals in eq.items():
        if len(vals) > 1:  # YEAR == 2021 and YEAR == 2020: no row is both
            hints.insert(0, f"the conditions contradict: '{c}' can't equal {' and '.join(sorted(vals))} at the "
                            f"same time (use op 'in' for any of them)")
    return ("No rows match the filter, so there is nothing to calculate. " + "; ".join(dict.fromkeys(hints))
            + ". Fix the filter, or call the tool without `where` to use every row.")


def _filtered(df: pd.DataFrame, where) -> pd.DataFrame:
    """apply_where for tools that calculate: an empty result is an error, never an answer of nan."""
    d = apply_where(df, where)
    if where and d.empty:
        raise ToolError(_no_rows_hint(df, where))
    return d


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return buf.getvalue()


# ---------- profile & cleaning ----------

def profile(df):
    info = pd.DataFrame({
        "type": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "unique": df.nunique(),
        "example": df.iloc[0].astype(str) if len(df) else "",
    })
    dups = int(df.duplicated().sum())
    return ToolResult(f"{len(df)} rows x {df.shape[1]} columns, {int(df.isna().sum().sum())} missing values, "
                      f"{dups} duplicate rows.", table=info.reset_index(names="column"))


def clean_missing(df, strategy, column=None, value=None):
    cols = _cols(df, column) or list(df.columns)
    out = df.copy()
    before = int(out[cols].isna().sum().sum())
    if strategy == "drop":
        out = out.dropna(subset=cols)
    elif strategy in ("mean", "median"):
        for c in cols:
            if out[c].isna().any():
                if column is None and not pd.api.types.is_numeric_dtype(out[c]):
                    continue  # whole-table fill: skip text columns instead of failing
                c = _numeric(out, c)
                out[c] = out[c].fillna(getattr(out[c], strategy)())
    elif strategy == "mode":
        for c in cols:
            if out[c].isna().any() and not out[c].mode().empty:
                out[c] = out[c].fillna(out[c].mode()[0])
    elif strategy == "value":
        if value is None:
            raise ToolError("strategy 'value' needs a value.")
        for c in cols:
            out[c] = out[c].fillna(_num(value) if pd.api.types.is_numeric_dtype(out[c]) else value)
    else:
        raise ToolError("strategy must be one of drop, mean, median, mode, value.")
    after = int(out[cols].isna().sum().sum())
    return ToolResult(f"Missing values in {', '.join(map(str, cols))}: {before} -> {after}. Rows: {len(df)} -> {len(out)}.",
                      new_df=out)


def drop_duplicates(df, subset=None):
    out = df.drop_duplicates(subset=_cols(df, subset) or None)
    return ToolResult(f"Removed {len(df) - len(out)} duplicate rows. Rows: {len(df)} -> {len(out)}.", new_df=out)


def convert_types(df, column, to):
    c, out = col(df, column), df.copy()
    if to == "number":
        cleaned = out[c].astype(str).str.replace(r"[^\d\.\-]", "", regex=True)
        out[c] = pd.to_numeric(cleaned.replace("", np.nan), errors="coerce")
    elif to == "date":
        out[c] = pd.to_datetime(out[c], errors="coerce")
    elif to == "text":
        out[c] = out[c].astype(str)
    else:
        raise ToolError("to must be number, date or text.")
    failed = int(out[c].isna().sum() - df[c].isna().sum())
    return ToolResult(f"Converted '{c}' to {to}. {failed} values could not be converted and became empty.", new_df=out)


def rename_drop_columns(df, rename=None, drop=None):
    out = df.drop(columns=_cols(df, drop))
    if rename:
        out = out.rename(columns={col(out, k): v for k, v in rename.items()})
    return ToolResult(f"Columns now: {', '.join(map(str, out.columns))}.", new_df=out)


# ---------- analysis ----------

def filter_rows(df, where):
    out = apply_where(df, where)
    pct = len(out) / len(df) * 100 if len(df) else 0
    return ToolResult(f"{len(out):,} of {len(df):,} rows match ({pct:.2f}%).", table=out)


def sort_top_n(df, column, n=10, ascending=False, where=None):
    d = _filtered(df, where)
    c = col(d, column)
    out = d.sort_values(c, ascending=str(ascending).lower() == "true").head(int(n))
    return ToolResult(f"{'Bottom' if str(ascending).lower() == 'true' else 'Top'} {len(out)} rows by '{c}'.", table=out)


AGGS = ("sum", "mean", "count", "min", "max", "median")


VIRTUAL = {"month": "M", "bulan": "M", "year": "Y", "tahun": "Y", "quarter": "Q", "kuartal": "Q",
           "weekday": "D", "day_of_week": "D", "hari": "D", "week": "W", "minggu": "W"}


def _with_virtual(d, by):
    """Group by 'month'/'bulan'/'year'/... even when the data only has a date column: derive it."""
    names = [by] if isinstance(by, str) else list(by)
    lookup = {str(c).strip().lower() for c in d.columns}
    missing = [n for n in names if str(n).strip().lower() not in lookup and str(n).strip().lower() in VIRTUAL]
    date_col = next((c for c in d.columns if _date_like(d[c])), None)
    if not missing or date_col is None:
        return d
    dates = pd.to_datetime(d[date_col], errors="coerce")
    derived = {"M": dates.dt.strftime("%Y-%m"), "Y": dates.dt.strftime("%Y"),
               "Q": dates.dt.year.astype("Int64").astype(str) + "-Q" + dates.dt.quarter.astype("Int64").astype(str),
               "D": dates.dt.day_name().map(lambda x: f"{x} ({HARI[x]})" if x in HARI else x),
               "W": dates.dt.strftime("%G-W%V")}
    return d.assign(**{n: derived[VIRTUAL[str(n).strip().lower()]] for n in missing})


def group_aggregate(df, by=None, column=None, agg="sum", where=None):
    if not by:  # small models call this for one overall number
        return aggregate(df, agg, column, where)
    if agg not in AGGS:
        raise ToolError(f"agg must be one of {', '.join(AGGS)}.")
    d = _with_virtual(_filtered(df, where), by)
    keys = _cols(d, by)
    if agg == "count" and column is None:
        out = d.groupby(keys).size().rename("count")
    else:
        if column is None:
            raise ToolError(f"agg '{agg}' needs a column.")
        c = col(d, column) if agg == "count" else _numeric(d, column)
        out = d.groupby(keys)[c].agg(agg).rename(f"{agg}_{c}")
    out = out.sort_values(ascending=False).reset_index()
    val = out.columns[-1]
    if agg in ("sum", "count") and out[val].sum():
        out["share_pct"] = (out[val] / out[val].sum() * 100).round(2)
    if out.empty:
        return ToolResult("No rows match.", table=out)
    top, low = out.iloc[0], out.iloc[-1]
    name = lambda r: ", ".join(str(r[k]) for k in keys)  # noqa: E731
    return ToolResult(f"{len(out)} groups. Highest: {name(top)} = {top[val]:,.2f}. Lowest: {name(low)} = {low[val]:,.2f}.", table=out)


def aggregate(df, agg="count", column=None, where=None):
    if agg not in AGGS:
        raise ToolError(f"agg must be one of {', '.join(AGGS)}.")
    d = apply_where(df, where)
    if agg == "count":
        if not column:
            return ToolResult(f"Count of rows: {len(d):,} (out of {len(df):,} rows).")
        c = col(d, column)
        filled = int(d[c].count())
        return ToolResult(f"Column '{c}' over {len(d):,} rows: {filled:,} filled, {len(d) - filled:,} missing (empty).")
    c = _numeric(d, column) if column else None
    if c is None:
        raise ToolError(f"agg '{agg}' needs a column.")
    if where and d.empty:  # a count of 0 is an answer; the mean of no rows is not ("nan")
        raise ToolError(_no_rows_hint(df, where))
    if d[c].notna().sum() == 0:
        raise ToolError(f"'{c}' has no values in these {len(d):,} rows (all empty), so its {agg} can't be calculated.")
    value = d[c].agg(agg)
    return ToolResult(f"{agg} of '{c}' over {len(d):,} rows: {value:,.2f}.")


def pivot(df, index, columns, values, agg="sum", where=None):
    if agg not in AGGS:
        raise ToolError(f"agg must be one of {', '.join(AGGS)}.")
    d = _filtered(df, where)
    out = pd.pivot_table(d, index=col(d, index), columns=col(d, columns), values=_numeric(d, values), aggfunc=agg)
    return ToolResult(f"Pivot of {agg} of '{col(d, values)}' by '{col(d, index)}' x '{col(d, columns)}'.",
                      table=out.round(2).reset_index())


def describe(df, columns=None, where=None):
    d = _filtered(df, where)
    cols = _cols(d, columns)
    out = (d[cols] if cols else d).describe(include="all").T
    return ToolResult(f"Summary statistics for {len(out)} columns over {len(d)} rows.", table=out.reset_index(names="column"))


def value_counts(df, column, n=10, where=None):
    d = _filtered(df, where)
    c = col(d, column)
    counts = d[c].value_counts()
    if counts.empty:
        return ToolResult(f"'{c}' has no values.")
    out = pd.DataFrame({c: counts.index, "count": counts.values, "pct": (counts.values / counts.sum() * 100).round(2)})
    top, low = out.iloc[0], out.iloc[-1]
    return ToolResult(f"'{c}' has {len(counts)} distinct values; most common: {top[c]} ({top['count']}, {top['pct']}%); "
                      f"least common: {low[c]} ({low['count']}, {low['pct']}%).",
                      table=out.head(int(n)))


def correlation(df, columns=None, where=None):
    d = _filtered(df, where)
    cols = [_numeric(d, c) for c in _cols(d, columns)] or list(d.select_dtypes("number").columns)
    if len(cols) < 2:
        raise ToolError("Need at least two numeric columns for correlation.")
    m = d[cols].corr().round(3)
    pairs = m.where(np.triu(np.ones(m.shape, bool), k=1)).stack()
    a, b = pairs.abs().idxmax()
    return ToolResult(f"Strongest correlation: '{a}' and '{b}' = {m.loc[a, b]}.", table=m.reset_index(names="column"))


def outliers(df, column, method="iqr", where=None):
    d = _filtered(df, where)
    c = _numeric(d, column)
    s = d[c]
    if method == "iqr":
        q1, q3 = s.quantile([0.25, 0.75])
        lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    elif method == "zscore":
        lo, hi = s.mean() - 3 * s.std(), s.mean() + 3 * s.std()
    else:
        raise ToolError("method must be iqr or zscore.")
    out = d[(s < lo) | (s > hi)]
    return ToolResult(f"{len(out)} outliers in '{c}' ({method}); normal range {lo:,.2f} to {hi:,.2f}.", table=out)


FREQS = {"D": "D", "W": "W", "M": "MS", "Q": "QS", "Y": "YS"}


def _series(df, date_column, value_column, freq, agg):
    dc, vc = col(df, date_column), _numeric(df, value_column)
    if freq not in FREQS:
        raise ToolError("freq must be D, W, M, Q or Y.")
    if agg not in AGGS:
        raise ToolError(f"agg must be one of {', '.join(AGGS)}.")
    dates = pd.to_datetime(df[dc], errors="coerce")
    if dates.isna().all():
        raise ToolError(f"Column '{dc}' does not contain dates.")
    return df.assign(**{dc: dates}).dropna(subset=[dc]).set_index(dc)[vc].resample(FREQS[freq]).agg(agg), dc, vc


def time_trend(df, date_column, value_column, freq="M", agg="sum", where=None):
    s, dc, vc = _series(_filtered(df, where), date_column, value_column, freq, agg)
    if s.empty:
        return ToolResult("No rows match.")
    out = s.rename(f"{agg}_{vc}").reset_index()
    out["growth_pct"] = (s.pct_change().values * 100).round(2)
    vals = out.iloc[:, 1]
    return ToolResult(f"{len(out)} periods. First {vals.iloc[0]:,.2f}, last {vals.iloc[-1]:,.2f}; "
                      f"peak {vals.max():,.2f} at {out.loc[vals.idxmax(), dc]:%Y-%m-%d}.", table=out)


def forecast(df, date_column, value_column, freq="M", periods=3, method="linear"):
    s, dc, vc = _series(df, date_column, value_column, freq, "sum")
    if len(s) < 3:
        raise ToolError("Need at least 3 periods of data to forecast.")
    periods = int(periods)
    future = pd.date_range(s.index[-1], periods=periods + 1, freq=FREQS[freq])[1:]
    if method == "linear":
        slope, intercept = np.polyfit(np.arange(len(s)), s.values, 1)
        pred = intercept + slope * np.arange(len(s), len(s) + periods)
    elif method == "moving_average":
        pred = np.repeat(s.tail(3).mean(), periods)
    else:
        raise ToolError("method must be linear or moving_average.")
    out = pd.DataFrame({dc: future, f"forecast_{vc}": pred.round(2)})
    return ToolResult(f"{method} forecast for next {periods} periods: " + ", ".join(f"{v:,.2f}" for v in pred) + ".",
                      table=out)


DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
HARI = dict(zip(DAYS, ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]))  # model mistranslated days itself


def _dates(df, date_column):
    dc = col(df, date_column)
    dates = pd.to_datetime(df[dc], errors="coerce")
    if dates.isna().all():
        raise ToolError(f"Column '{dc}' does not contain dates.")
    return dc, dates


def weekday_pattern(df, date_column, value_column, where=None):
    d = _filtered(df, where)
    vc = _numeric(d, value_column)
    agg = "sum"  # not a parameter: the small model picked "min" here, which answers nothing useful
    dc, dates = _dates(d, date_column)
    # one value per calendar day first, so a weekday with more rows per day isn't counted twice
    daily = d[vc].groupby(dates.dt.normalize()).agg(agg)
    per = daily.groupby(daily.index.day_name()).agg(["mean", "size"]).reindex(DAYS).dropna()
    if per.empty:
        return ToolResult("No rows match.")
    avg = per["mean"].mean()
    out = pd.DataFrame({"weekday": [f"{d} ({HARI[d]})" for d in per.index], f"avg_daily_{agg}_{vc}": per["mean"].round(2).values,
                        "days": per["size"].astype(int).values, "vs_average_pct": ((per["mean"] / avg - 1) * 100).round(1).values})
    val = out.columns[1]
    top, low = out.loc[out[val].idxmax()], out.loc[out[val].idxmin()]
    return ToolResult(f"Average daily {agg} of '{vc}' by day of week. Highest: {top.weekday} = {top[val]:,.2f} "
                      f"({top.vs_average_pct:+.1f}% vs average). Lowest: {low.weekday} = {low[val]:,.2f} "
                      f"({low.vs_average_pct:+.1f}% vs average).", table=out)


def restock(df, date_column, product_column, quantity_column, stock_column, days=28, warn_days=7):
    dc, dates = _dates(df, date_column)
    pc, qc, sc = col(df, product_column), _numeric(df, quantity_column), _numeric(df, stock_column)
    days, warn_days = int(days), int(warn_days)
    d = df.assign(_date=dates).dropna(subset=["_date"]).sort_values("_date", kind="stable")
    stock = d.groupby(pc)[sc].last()  # each product's stock on its latest date
    recent = d[d["_date"] > d["_date"].max() - pd.Timedelta(days=days)]
    per_day = (recent.groupby(pc)[qc].sum() / days).reindex(stock.index, fill_value=0)
    left = (stock / per_day.replace(0, np.nan)).fillna(np.inf)  # never sells -> never runs out
    left[stock <= 0] = 0
    out = pd.DataFrame({"product": stock.index, "stock_now": stock.values, "avg_sold_per_day": per_day.round(2).values,
                        "days_left": left.round(1).values})
    out["status"] = np.select([out.stock_now <= 0, out.days_left <= warn_days], ["out of stock", "restock now"], "ok")
    out = out.sort_values(["days_left", "product"]).reset_index(drop=True)
    soon = out[out.status != "ok"]
    first = out.iloc[0]
    listed = ", ".join(f"{r.product} ({r.days_left:g} days)" for r in soon.itertuples())
    return ToolResult(f"Based on sales in the last {days} days. {len(soon)} of {len(out)} products run out within "
                      f"{warn_days} days{': ' + listed if listed else ''}. First to run out: {first['product']} "
                      f"(stock {first.stock_now:g}, sells about {first.avg_sold_per_day:g} per day, {first.days_left:g} days left).",
                      table=out)


def chart(df, kind, x=None, y=None, agg="sum", where=None):
    if agg not in AGGS:
        raise ToolError(f"agg must be one of {', '.join(AGGS)}.")
    d = _filtered(df, where)
    if x and kind in ("bar", "line"):
        d = _with_virtual(d, [x])  # x='month' / 'bulan' / 'week' / 'year' from the date column
    fig, ax = plt.subplots(figsize=(7, 4))
    data = None
    try:
        if kind == "bar":
            xc = col(d, x)
            data = d[xc].value_counts().head(15) if y is None else d.groupby(xc)[_numeric(d, y)].agg(agg).sort_values(ascending=False).head(15)
            data.plot.bar(ax=ax)
        elif kind == "line":
            data = d[_numeric(d, y)].groupby(d[col(d, x)]).agg(agg)
            data.plot.line(ax=ax, marker="o")
            ax.tick_params(axis="x", labelrotation=45)
        elif kind == "hist":
            d[_numeric(d, x or y)].plot.hist(ax=ax, bins=30)
        elif kind == "scatter":
            xc, yc = _numeric(d, x), _numeric(d, y)
            ax.scatter(d[xc], d[yc], s=10, alpha=0.6)
            ax.set_xlabel(xc)
            ax.set_ylabel(yc)
        elif kind == "box":
            yc = _numeric(d, y or x)
            if x and y:
                d.boxplot(column=yc, by=col(d, x), ax=ax, rot=45)
            else:
                d[[yc]].boxplot(ax=ax)
        elif kind == "heatmap":
            m = d.select_dtypes("number").corr()
            im = ax.imshow(m, cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_xticks(range(len(m)), m.columns, rotation=45, ha="right")
            ax.set_yticks(range(len(m)), m.columns)
            fig.colorbar(im)
        else:
            raise ToolError("kind must be bar, line, hist, scatter, box or heatmap.")
    except Exception:
        plt.close(fig)
        raise
    ax.set_title(f"{kind}: {y or ''} {'by ' + str(x) if x and y else x or ''}".strip())
    if kind in ("bar", "line"):  # 70,000,000 instead of "7.0 1e7"
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    if data is None or data.empty:
        return ToolResult(f"{kind} chart created.", chart=_png(fig))
    label = f"{agg} of '{y}'" if y else "count"
    return ToolResult(f"{kind} chart of {label} by '{x}', {len(data)} points. Highest: {data.idxmax()} = {data.max():,.2f}. "
                      f"Lowest: {data.idxmin()} = {data.min():,.2f}.",
                      table=data.rename(label).reset_index(), chart=_png(fig))


# ---------- registry ----------

_STR, _INT, _BOOL = {"type": "string"}, {"type": "integer"}, {"type": "boolean"}
_COLS = {"type": "array", "items": {"type": "string"}}
_WHERE = {"type": "array", "description": "Optional row filters, ONLY for conditions stated in the question. "
          "For a year or month on a date column use op '==' with value '2025' or '2025-03'.",
          "items": {"type": "object", "properties": {"column": _STR, "op": {"type": "string", "enum": ["==", "!=", ">", ">=", "<", "<=", "contains", "in"]}, "value": _STR}}}
_AGG = {"type": "string", "enum": list(AGGS)}


def _enum(*v):
    return {"type": "string", "enum": list(v)}


def _schema(name, desc, props=None, required=()):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props or {}, "required": list(required)}}}


TOOLS: dict[str, tuple[Callable, dict]] = {
    "profile": (profile, _schema("profile", "Overview of the dataset: columns, types, missing values, duplicates.")),
    "clean_missing": (clean_missing, _schema("clean_missing", "Handle missing values. Creates a new dataset version.",
                      {"strategy": _enum("drop", "mean", "median", "mode", "value"), "column": _STR, "value": _STR}, ["strategy"])),
    "drop_duplicates": (drop_duplicates, _schema("drop_duplicates", "Remove duplicate rows. Creates a new dataset version.", {"subset": _COLS})),
    "convert_types": (convert_types, _schema("convert_types", "Convert a column to number, date or text. Creates a new dataset version.",
                      {"column": _STR, "to": _enum("number", "date", "text")}, ["column", "to"])),
    "rename_drop_columns": (rename_drop_columns, _schema("rename_drop_columns", "Rename and/or drop columns. Creates a new dataset version.",
                            {"rename": {"type": "object", "description": "old name -> new name"}, "drop": _COLS})),
    "filter_rows": (filter_rows, _schema("filter_rows", "Show rows matching conditions.", {"where": _WHERE}, ["where"])),
    "sort_top_n": (sort_top_n, _schema("sort_top_n", "Top (or bottom) N rows by a column.",
                   {"column": _STR, "n": _INT, "ascending": _BOOL, "where": _WHERE}, ["column"])),
    "group_aggregate": (group_aggregate, _schema("group_aggregate",
                        "Compare groups: which city/department/product has the highest or lowest total, average or count. "
                        "Reports highest and lowest. E.g. 'which city has the most revenue' -> by=['city'], column='revenue', agg='sum'; "
                        "'which department has the most employees' -> by=['department'], agg='count'; "
                        "'produk paling laris' / 'which product sells the fewest units' -> by=['produk'], column='terjual', agg='sum'.",
                        {"by": _COLS, "column": _STR, "agg": _AGG, "where": _WHERE}, ["by", "agg"])),
    "aggregate": (aggregate, _schema("aggregate",
                  "One overall number: total, average, count, min, max, median. "
                  "E.g. 'total revenue' -> agg='sum', column='revenue'; 'how many orders' -> agg='count'; "
                  "'how many employees left' -> agg='count', where=[{column:'left_company', op:'==', value:'Yes'}]; "
                  "'how many missing values in X' -> agg='count', column='X' (reports filled and missing); "
                  "'total laba bulan Maret 2026' -> agg='sum', column='laba', where=[{column:'tanggal', op:'==', value:'2026-03'}].",
                  {"agg": _AGG, "column": _STR, "where": _WHERE}, ["agg"])),
    "pivot": (pivot, _schema("pivot", "Pivot table: rows = index, columns = columns, cells = agg of values.",
              {"index": _STR, "columns": _STR, "values": _STR, "agg": _AGG, "where": _WHERE}, ["index", "columns", "values"])),
    "describe": (describe, _schema("describe", "Summary statistics (count, mean, std, min, quartiles, max).", {"columns": _COLS, "where": _WHERE})),
    "value_counts": (value_counts, _schema("value_counts", "Count of each distinct value in a column, with %.",
                     {"column": _STR, "n": _INT, "where": _WHERE}, ["column"])),
    "correlation": (correlation, _schema("correlation", "Correlation between numeric columns; names the strongest pair.", {"columns": _COLS, "where": _WHERE})),
    "outliers": (outliers, _schema("outliers", "Find outlier rows in a numeric column.",
                 {"column": _STR, "method": _enum("iqr", "zscore"), "where": _WHERE}, ["column"])),
    "time_trend": (time_trend, _schema("time_trend", "Aggregate a value over time (D/W/M/Q/Y) with growth %.",
                   {"date_column": _STR, "value_column": _STR, "freq": _enum(*FREQS), "agg": _AGG, "where": _WHERE}, ["date_column", "value_column"])),
    "forecast": (forecast, _schema("forecast", "Forecast future totals of a value over time.",
                 {"date_column": _STR, "value_column": _STR, "freq": _enum(*FREQS), "periods": _INT, "method": _enum("linear", "moving_average")},
                 ["date_column", "value_column"])),
    "weekday_pattern": (weekday_pattern, _schema("weekday_pattern",
                        "Which day of the week is busiest or quietest (hari apa paling ramai / sepi), e.g. to pick a promo day. "
                        "Averages the daily total for each weekday; reports highest and lowest.",
                        {"date_column": _STR, "value_column": _STR, "where": _WHERE}, ["date_column", "value_column"])),
    "restock": (restock, _schema("restock",
                "Stock planning (kapan harus restock, stok habis): for each product, current stock, average sold per day "
                "and days until it runs out. Needs a date, product, quantity sold and stock column. "
                "Use for 'which product runs out first', 'what do I need to restock', 'how much stock is left'.",
                {"date_column": _STR, "product_column": _STR, "quantity_column": _STR, "stock_column": _STR,
                 "days": {"type": "integer", "description": "days of recent sales to average (default 28)"},
                 "warn_days": {"type": "integer", "description": "flag products running out within this many days (default 7)"}},
                ["date_column", "product_column", "quantity_column", "stock_column"])),
    "chart": (chart, _schema("chart", "Draw a chart (grafik). bar: x category (y optional, aggregated). line: x vs y. hist: x numeric. scatter: x,y numeric. box: y numeric, x optional group. heatmap: correlations. "
              "Trend over time: kind='line', x='month' (or 'week', 'year', or the date column for daily), y=the value, agg='sum'.",
              {"kind": _enum("bar", "line", "hist", "scatter", "box", "heatmap"), "x": _STR, "y": _STR, "agg": _AGG, "where": _WHERE}, ["kind"])),
}


def _run_python(df, code):
    from agent.sandbox import run_python  # local import: sandbox imports this module

    return run_python(df, code)


TOOLS["run_python"] = (_run_python, _schema(
    "run_python", "LAST RESORT when no other tool fits. Pandas code using `df`, `pd`, `np`; assign the answer to `result`.",
    {"code": _STR}, ["code"]))


def tool_schemas() -> list[dict]:
    return [schema for _, schema in TOOLS.values()]


def run_tool(name: str, df: pd.DataFrame, args: dict | None) -> ToolResult:
    if name not in TOOLS:
        raise ToolError(f"Unknown tool '{name}'. Available: {', '.join(TOOLS)}.")
    fn = TOOLS[name][0]
    params = inspect.signature(fn).parameters
    clean = {}
    for k, v in (args or {}).items():
        if v in (None, "", [], {}):  # small models send empty optionals
            continue
        if k not in params:  # "column" vs "columns" slips
            k = next((p for p in (k + "s", k[:-1]) if p in params), k)
        clean[k] = v
    try:
        return fn(df, **clean)
    except ToolError:
        raise
    except TypeError as e:
        raise ToolError(f"Bad arguments for {name}: {e}")
    except (ValueError, KeyError, IndexError) as e:
        raise ToolError(f"{name} failed: {e}")

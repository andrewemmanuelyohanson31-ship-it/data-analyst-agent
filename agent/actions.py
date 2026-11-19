"""One-click actions for the most common jobs. Each button runs its tools directly, with arguments built from the
dataset's own columns: no question text, nothing for a model or parser to interpret.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from agent.loop import AgentAnswer, Step
from agent.roles import columns_by_role
from agent.router import _fmt
from agent.state import DatasetState
from agent.tools import run_tool

NOTE = "one-click action, no model call"


@dataclass
class Action:
    label: str
    calls: list[tuple[str, dict]]  # (tool, args), run in order on the current dataset version
    summary: Callable[[list], str] = field(default=lambda results: " ".join(r.text for r in results))


NEEDED = ("date", "revenue", "quantity", "profit", "product", "category", "city", "department", "salary", "stock",
          "supplier")


def _columns(df: pd.DataFrame) -> dict:
    """The column for each role the actions need (None when the data has no such column)."""
    by_role = columns_by_role(df)
    return {role: by_role.get(role) for role in NEEDED}


def _high_low(r, label=""):
    t = r.table
    key, val = t.columns[0], t.columns[1]
    hi, lo = t.loc[t[val].idxmax()], t.loc[t[val].idxmin()]
    return f"{label}Highest: **{hi[key]}** = {_fmt(hi[val])}. Lowest: **{lo[key]}** = {_fmt(lo[val])}."


def _clean_summary(results):
    removed = sum(int(m.group(1)) for r in results if (m := re.search(r"Removed (\d+) duplicate", r.text)))
    filled = sum(int(a) - int(b) for r in results for a, b in re.findall(r": (\d+) -> (\d+)\. Rows", r.text))
    if not removed and not filled:
        return "Nothing to clean: no duplicate rows and no missing values."
    return (f"Cleaned: removed **{removed}** duplicate rows and filled **{filled}** missing values (numbers with the "
            f"column median, text with the most common value). Use **Undo** to go back.")


def available(df: pd.DataFrame) -> list[Action]:
    """The actions that make sense for this dataset, in button order."""
    c = _columns(df)
    out = [Action("📋 Overview", [("profile", {})]),
           Action("🧹 Clean data", [("drop_duplicates", {}), ("clean_missing", {"strategy": "median"}),
                                    ("clean_missing", {"strategy": "mode"})],
                  _clean_summary)]
    money = c["revenue"] or c["profit"]
    word = "revenue" if c["revenue"] else "profit"  # label by meaning: "Total revenue", not "Total Subtotal"
    if money:
        out.append(Action(f"💰 Total {word}", [("aggregate", {"agg": "sum", "column": money})]))
    if c["date"] and money:
        out.append(Action(f"📈 Monthly {word} chart", [("chart", {"kind": "line", "x": "month", "y": money, "agg": "sum"})],
                          lambda rs: _high_low(rs[0], f"Monthly {money}. ")))
    purchases = c["supplier"] and not c["revenue"]  # a purchase log: quantities were bought, not sold
    if c["product"] and c["quantity"] and purchases:
        out.append(Action("📦 Most bought items", [("group_aggregate", {"by": [c["product"]], "column": c["quantity"],
                                                                        "agg": "sum"})],
                          lambda rs: _high_low(rs[0], f"Units bought ({c['quantity']}) per {c['product']}. ")))
    elif c["product"] and c["quantity"]:
        by_product = {"by": [c["product"]], "column": c["quantity"], "agg": "sum"}
        out.append(Action("🏆 Best sellers", [("group_aggregate", by_product)],
                          lambda rs: _high_low(rs[0], f"Units sold ({c['quantity']}) per {c['product']}. ")))
        out.append(Action("🐢 Slow sellers", [("group_aggregate", by_product)],
                          lambda rs: f"Slowest seller: **{rs[0].table.iloc[-1, 0]}** = {_fmt(rs[0].table.iloc[-1, 1])} "
                                     f"{c['quantity']} (table: best to slowest)."))
    group = c["category"] or c["department"] or c["city"] or c["product"]
    if group and money:
        out.append(Action(f"📊 {word.capitalize()} by {group}", [("chart", {"kind": "bar", "x": group, "y": money, "agg": "sum"})],
                          lambda rs: _high_low(rs[0], f"{money} by {group}. ")))
    elif group and c["salary"]:  # e.g. an employee list: no sales, but pay per group is the usual question
        out.append(Action(f"📊 Average salary by {group}",
                          [("chart", {"kind": "bar", "x": group, "y": c["salary"], "agg": "mean"})],
                          lambda rs: _high_low(rs[0], f"Average {c['salary']} by {group}. ")))
    if c["date"] and money:
        out.append(Action("📅 Busy & quiet days", [("weekday_pattern", {"date_column": c["date"], "value_column": money})]))
    if c["date"] and c["product"] and c["quantity"] and c["stock"]:
        out.append(Action("📦 Restock list", [("restock", {"date_column": c["date"], "product_column": c["product"],
                                                         "quantity_column": c["quantity"], "stock_column": c["stock"]})]))
    return out


def run(action: Action, state: DatasetState) -> AgentAnswer:
    """Run the action's tools in order; cleaning tools add dataset versions, so Undo works as usual."""
    steps, results = [], []
    for tool, args in action.calls:
        result = run_tool(tool, state.current, dict(args))
        if result.new_df is not None and not result.new_df.equals(state.current):  # no empty versions in Undo
            state.apply(tool, result.new_df)
        steps.append(Step(tool, dict(args), result, fixes=[NOTE]))
        results.append(result)
    return AgentAnswer(action.summary(results), steps)

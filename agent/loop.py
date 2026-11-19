"""The agent loop: a local model picks tools, tested code runs them, results go back until it answers."""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Protocol

import pandas as pd

from agent.numbers import ungrounded_numbers
from agent.router import answer as route
from agent.state import DatasetState
from agent.tools import TOOLS, ToolError, ToolResult, run_tool, tool_schemas

DEFAULT_MODEL = "qwen2.5:3b"
MAX_ROWS_TO_MODEL = 20

SYSTEM = """You are a careful data analyst. A pandas DataFrame is loaded; its schema is below.
Answer the user's question by calling tools. Rules:
- Every number in your answer must come from a tool result. Never guess or compute numbers yourself.
- Copy numbers exactly as the tool wrote them. Do not add currency symbols that are not in the data.
- For one overall number (total, average, how many rows/orders) use the aggregate tool.
- Add a `where` filter for every condition written in the question (a city, product, month, year, "with a discount"),
  and never for a condition that is not written. Never filter on a value you are guessing.
- "Most / least / biggest / best-selling / most profit" of a product, city or category means its TOTAL: agg="sum".
  The same in Indonesian: paling laris, paling banyak, paling besar, terbesar, paling sedikit, paling kecil, terkecil.
  Use agg="max" or "min" only when the question asks for the single largest or smallest value.
- If the question asks for the lowest / fewest / smallest, use the "Lowest" part of the result.
- Use exact column names from the schema.
- Use one tool call when one is enough. Use `where` filters instead of separate filter steps.
- If the question asks for a chart, graph, plot or grafik, you MUST call the chart tool (a table is not a chart).
- Use run_python only if no other tool can do it.
- After you have the result, answer in 1-3 short sentences in the same language as the question (English or Indonesian).

Dataset schema:
{schema}"""

NUDGE = "You answered with numbers without calling a tool. Call the right tool first, then answer from its result."


class ChatModel(Protocol):
    def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        """Return {"content": str, "tool_calls": [{"name": str, "arguments": dict}]}."""


class OllamaModel:
    def __init__(self, model: str | None = None, host: str | None = None):
        import ollama

        self.model = model or os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)
        self.client = ollama.Client(host=host or os.getenv("OLLAMA_HOST"))

    def chat(self, messages, tools):
        # num_ctx: Ollama defaults to 4096 and silently truncates; prompt + tool schemas alone are ~3.6k tokens.
        msg = self.client.chat(model=self.model, messages=messages, tools=tools,
                               options={"temperature": 0, "num_ctx": 8192}).message
        calls = [{"name": c.function.name, "arguments": dict(c.function.arguments or {})} for c in (msg.tool_calls or [])]
        return {"content": msg.content or "", "tool_calls": calls}


@dataclass
class Step:
    tool: str
    args: dict
    result: ToolResult | None = None
    error: str | None = None
    seconds: float = 0.0
    fixes: list[str] = field(default_factory=list)  # notes shown in the Steps panel, e.g. answered by the question parser


@dataclass
class AgentAnswer:
    text: str
    steps: list[Step] = field(default_factory=list)
    note: str = ""  # set when the model's own wording was replaced by the tools' verified summary


def _params(tool: str) -> dict:
    return TOOLS[tool][1]["function"]["parameters"]["properties"] if tool in TOOLS else {}


def schema_summary(df: pd.DataFrame) -> str:
    from agent.roles import detect_roles  # local import: roles imports column lists only, keep loop import-light

    roles = detect_roles(df)
    cols = "\n".join(f"- {c} ({t}, {df[c].isna().sum()} missing{', means: ' + roles[c] if roles.get(c) else ''})"
                     for c, t in df.dtypes.astype(str).items())
    return f"{len(df)} rows. Columns:\n{cols}\nFirst rows:\n{df.head(5).to_csv(index=False)}"


def _text_tool_call(content: str) -> dict | None:
    """Small models sometimes write the tool call as JSON text instead of a real call."""
    m = re.search(r"\{.*\}", content, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict) and "name" in obj and isinstance(obj.get("arguments", {}), dict):
        return {"name": obj["name"], "arguments": obj.get("arguments", {})}
    return None


def _for_model(result: ToolResult) -> str:
    out = result.text
    if result.table is not None:
        t = result.table
        out += f"\nTable ({len(t)} rows{', first ' + str(MAX_ROWS_TO_MODEL) if len(t) > MAX_ROWS_TO_MODEL else ''}):\n"
        out += t.head(MAX_ROWS_TO_MODEL).to_csv(index=False)
    return out


def run_agent(question: str, state: DatasetState, model: ChatModel, history: list[dict] | None = None,
              max_steps: int = 6) -> AgentAnswer:
    t0 = time.perf_counter()
    if (routed := route(question, state.current)) is not None:  # frequent question types: no model needed
        text, tool, args, result = routed
        step = Step(tool, args, result, seconds=time.perf_counter() - t0,
                    fixes=["answered by the question parser (agent/router.py), no model call"])
        return AgentAnswer(text, [step])
    messages = [{"role": "system", "content": SYSTEM.format(schema=schema_summary(state.current))},
                *(history or []), {"role": "user", "content": question}]
    steps: list[Step] = []
    nudged = False
    tools = tool_schemas()
    evidence = question  # everything the model was shown; answer numbers must appear here
    active_where = None
    while len(steps) < max_steps:
        reply = model.chat(messages, tools)
        calls = reply["tool_calls"] or ([c] if (c := _text_tool_call(reply["content"])) else [])
        if not calls:
            text = reply["content"].strip()
            ok_steps = [s for s in steps if s.error is None]
            bad = ungrounded_numbers(text, evidence)
            if (not text or bad) and not ok_steps and not nudged:
                nudged = True
                messages += [{"role": "assistant", "content": text}, {"role": "user", "content": NUDGE}]
                continue
            if ok_steps and (not text or bad):
                note = f"The model's wording had numbers not found in the results ({', '.join(bad)}); showing the verified tool output instead." if bad else ""
                return AgentAnswer(" ".join(s.result.text for s in ok_steps[-2:]), steps, note)
            if bad or not text:
                return AgentAnswer("I couldn't get a verified answer. Try rephrasing the question or naming the column.", steps)
            return AgentAnswer(text, steps)
        messages.append({"role": "assistant", "content": reply["content"],
                         "tool_calls": [{"function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls]})
        for call in calls:
            args = dict(call["arguments"])
            # Small models filter in one step and forget the filter in the next: carry it over.
            if call["name"] == "filter_rows":
                active_where = args.get("where") or active_where
            elif active_where and not args.get("where") and "where" in _params(call["name"]):
                args["where"] = active_where
            step = Step(call["name"], args)
            t0 = time.perf_counter()
            try:
                step.result = run_tool(call["name"], state.current, args)
                if step.result.new_df is not None:
                    state.apply(call["name"], step.result.new_df)
                content = _for_model(step.result)
            except ToolError as e:
                step.error = str(e)
                content = f"ERROR: {e} Fix the arguments and try again."
            step.seconds = time.perf_counter() - t0
            steps.append(step)
            evidence += "\n" + content
            messages.append({"role": "tool", "content": content, "tool_name": call["name"]})
    last = next((s.result.text for s in reversed(steps) if s.result), "I could not finish this question.")
    return AgentAnswer(f"(Stopped after {max_steps} steps.) {last}", steps)

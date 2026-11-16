# Data Analyst Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Offline Streamlit chat agent that answers data-analyst questions on an uploaded CSV/Excel using a small local Ollama model choosing from tested pandas tools.

**Architecture:** `agent/tools.py` holds pure pandas tools + JSON schemas; `agent/loop.py` runs a tool-calling loop against a `ChatModel` (Ollama in prod, fake in tests); `agent/state.py` keeps dataset versions; `app.py` is the UI; `eval/` measures success rate.

**Tech Stack:** Python 3.12, pandas, numpy, matplotlib, openpyxl, streamlit, ollama (python client), pytest.

**Spec:** `docs/superpowers/specs/2026-11-16-data-analyst-agent-design.md`

## Global Constraints
- No external API; model via Ollama, default `qwen2.5:3b`, overridable by env `OLLAMA_MODEL`, host by `OLLAMA_HOST`.
- Cross-platform (Windows + macOS): `pathlib`, `sys.executable`, no shell-specific code.
- Max 6 agent steps per question; tool results sent to the model truncated to 20 rows.
- Sandbox timeout 10 s; only `pandas`, `numpy`, `math` importable.
- Original dataset never modified; cleaning creates a new version; undo available.
- Commit titles ~6 words.

## Review Focus
- Column names with spaces/mixed case, or the model using wrong case → tools resolve columns case-insensitively, else ToolError listing real columns.
- Non-numeric column passed to numeric tools (mean, correlation, outliers) → clear ToolError, not a pandas traceback.
- Model answers with numbers without calling any tool → loop rejects once and asks it to use a tool.
- Model emits an unknown tool name or malformed args → error fed back, loop continues.
- Excel upload / CSV with `;` separator or non-UTF8 encoding → `load_file` sniffs separator and falls back to latin-1.

---

### Task 1: Tools core + cleaning/profile tools

**Files:** Create `requirements.txt`, `.gitignore`, `agent/__init__.py`, `agent/tools.py`, `tests/test_tools.py`

**Interfaces — Produces:**
- `class ToolError(Exception)`
- `@dataclass ToolResult(text: str, table: pd.DataFrame|None=None, chart: bytes|None=None, new_df: pd.DataFrame|None=None)`
- `load_file(name: str, data: bytes) -> pd.DataFrame`
- `col(df, name) -> str` case-insensitive column resolve
- `apply_where(df, where: list[dict]|None) -> pd.DataFrame`; each condition `{column, op, value}`, ops `== != > >= < <= contains in`
- tools: `profile(df)`, `clean_missing(df, strategy, column=None, value=None)` strategy in drop/mean/median/mode/value, `drop_duplicates(df, subset=None)`, `convert_types(df, column, to)` to in number/date/text, `rename_drop_columns(df, rename=None, drop=None)`
- `TOOLS: dict[str, tuple[Callable, dict]]` (function, Ollama tool schema), `tool_schemas() -> list[dict]`, `run_tool(name, df, args) -> ToolResult` (unknown name / TypeError → ToolError)

Tests: profile reports missing count + duplicates; clean_missing median fills and returns new_df without mutating input; `col(df,"CITY")` → `"city"`; bad column → ToolError message lists columns; `apply_where` with `>` and `contains`; load_file parses `;` CSV; run_tool unknown → ToolError.

### Task 2: Analysis + chart tools

**Files:** Modify `agent/tools.py`; Test `tests/test_tools.py`

**Produces:** `filter_rows(df, where)`, `sort_top_n(df, column, n=10, ascending=False, where=None)`, `group_aggregate(df, by, column=None, agg="sum", where=None)` (adds `share_pct`, sorted desc; `agg="count"` allows no column), `pivot(df, index, columns, values, agg="sum", where=None)`, `describe(df, columns=None, where=None)`, `value_counts(df, column, n=10, where=None)` (with pct), `correlation(df, columns=None, where=None)` (text names strongest pair), `outliers(df, column, method="iqr", where=None)`, `time_trend(df, date_column, value_column, freq="M", agg="sum", where=None)` (adds `growth_pct`), `forecast(df, date_column, value_column, freq="M", periods=3, method="linear")`, `chart(df, kind, x=None, y=None, agg="sum", where=None)` kinds bar/line/hist/scatter/box/heatmap → PNG bytes.
Numeric tools on non-numeric column → ToolError.

Tests: group_aggregate top city and share sums to 100; outliers IQR finds planted value; time_trend monthly sums; forecast linear on perfect line predicts next value; chart returns bytes starting `\x89PNG`; correlation on text column → ToolError.

### Task 3: Dataset state

**Files:** Create `agent/state.py`, `tests/test_state.py`
**Produces:** `class DatasetState(df, name)`: `.current`, `.original`, `.versions: list[tuple[str, DataFrame]]`, `.apply(label, df)`, `.undo() -> bool`.
Tests: apply then undo restores original; undo on original returns False.

### Task 4: Agent loop

**Files:** Create `agent/loop.py`, `tests/test_loop.py`
**Consumes:** `run_tool`, `tool_schemas`, `ToolError`, `DatasetState`.
**Produces:** `class ChatModel(Protocol): chat(messages, tools) -> dict` returning `{"content": str, "tool_calls": [{"name": str, "arguments": dict}]}`; `class OllamaModel(model=None, host=None)`; `@dataclass Step(tool, args, result, error, seconds)`; `@dataclass AgentAnswer(text, steps)`; `schema_summary(df) -> str`; `run_agent(question, state, model, history=None, max_steps=6) -> AgentAnswer`.
Rules: tool results with `new_df` call `state.apply`; ToolError fed back as tool message; answer with digits and zero successful tool calls → one nudge retry; step cap returns best-effort text.
Tests (FakeModel with scripted replies): tool call then answer → 1 step; error then corrected call → 2 steps, second ok; no-tool numeric answer triggers nudge; clean tool creates version.

### Task 5: Sandbox fallback `run_python`

**Files:** Create `agent/sandbox.py`, `tests/test_sandbox.py`; register `run_python(df, code)` in `TOOLS`.
**Produces:** `check_code(code) -> None` (ToolError), `run_python(df, code, timeout=10) -> ToolResult`; code must assign `result`.
Runs `sys.executable -I` child, df passed via pickle in temp dir.
Tests: `result = df['a'].sum()` works; `import os` rejected; `open(...)` rejected; `__class__` rejected; `while True: pass` → ToolError timeout.

### Task 6: Samples + report

**Files:** Create `samples/make_samples.py` (seeded; writes `samples/sales_id.csv` with planted missing/duplicates/outliers, `samples/employees.csv`), `agent/report.py`, `tests/test_report.py`.
**Produces:** `build_report(dataset_name, turns: list[tuple[str, AgentAnswer]]) -> str` (self-contained HTML, charts base64).
Test: report contains question text, `<table`, and `data:image/png;base64`.

### Task 7: Streamlit UI

**Files:** Create `app.py`, `.env.example`.
Sidebar: upload / sample picker, rows×cols, version list, Undo, Download CSV, Export report, model name. Chat: history, per answer text + table + chart + "Steps" expander. Ollama-down → friendly error with setup hint.
Verify: `streamlit run app.py` starts; manual question on sample with Ollama.

### Task 8: Eval + README

**Files:** Create `eval/tasks.json` (~40 tasks: `{dataset, question, expected}` where `expected` is a pandas expression on `df`), `eval/run_eval.py` (`--model`, prints success %, avg steps, avg s; correct when answer text contains expected string or any number within 1%), `README.md` (Windows + macOS setup, usage, eval table, design link).
Verify: `python eval/run_eval.py --model qwen2.5:3b --limit 3` runs.

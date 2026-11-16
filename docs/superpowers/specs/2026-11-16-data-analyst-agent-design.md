# Data Analyst Agent — Design

Date: 2026-11-16 · Status: draft, awaiting approval

## 1. Intent

A portfolio project: an AI agent that does the everyday work of a junior data analyst on a dataset you
upload, answering questions in English or Indonesian. It must run **fully offline on modest hardware**
(Windows PC with GTX 1650 4 GB / 16 GB RAM, and a MacBook), use **no external API**, and be impressive
and reliable enough to demo to recruiters.

## 2. Goals, non-goals, anti-goals

**Goals — the common data-analyst jobs it can do**
1. **Load & profile:** CSV/Excel → rows, columns, types, missing values, duplicates, sample rows.
2. **Clean:** drop duplicates, handle missing values (drop/fill mean/median/mode/value), fix types (dates, numbers), rename/drop columns.
3. **Filter & sort:** conditions on columns, top-N / bottom-N.
4. **Aggregate:** group-by with sum/mean/count/min/max/median, pivot tables, percentages / share of total.
5. **Describe:** summary statistics, value counts, distributions.
6. **Relationships:** correlation matrix, strongest correlations.
7. **Outliers:** IQR / z-score detection per column.
8. **Trends & forecast:** resample over time (day/week/month), growth rates, simple forecast (linear trend / moving average).
9. **Charts:** bar, line, histogram, scatter, box, heatmap.
10. **Report:** export the session (questions, answers, tables, charts) to a single HTML file.

**Non-goals:** databases/SQL connections, dashboards that update live, multi-user, cloud hosting,
fine-tuning a model, big data (> ~1M rows).

**Anti-goals (failure even if it "works")**
- **Made-up numbers.** Every number in an answer must come from a tool result, never from the model's head.
  The UI shows the exact table/chart the answer is based on.
- Silently changing the user's data: cleaning creates a new version; the original is kept and undo is possible.
- Running harmful code on the user's machine (see §6.3).

## 3. Constraints

- Local model via **Ollama** (Windows + macOS, one installer). No internet needed after setup.
- Default model **`qwen2.5:3b`** (~2 GB, supports tool calling, fits the 4 GB GPU; runs on Apple Silicon or CPU).
  Swappable in `.env` (e.g. `qwen2.5:7b` / `qwen2.5-coder:7b` on stronger machines). Final default is
  confirmed by the evaluation in §7.
- Python 3.12, cross-platform paths, no Windows-only or Mac-only code.

## 4. Approaches

| | Approach | Trade-off |
|---|---|---|
| A | Local app + bigger model (7B) writing pandas code | Best quality, too heavy for 4 GB GPU / older Mac |
| B | In-browser (WebLLM + Pyodide), always online | Free hosting, but needs WebGPU; many visitors can't load it |
| **C (chosen)** | Local app + **small model (3B) choosing from ready-made analysis tools**, code-writing only as fallback | Runs on both machines; reliable because tools are tested code; less flexible for exotic questions |

Andrew asked for the lowest compute that runs on his PC or MacBook → **C**. Demo for recruiters = README
GIF/video + evaluation numbers.

## 5. User experience

Streamlit app, `streamlit run app.py` → browser opens.

```
┌ Sidebar ─────────────┐┌ Chat ────────────────────────────────────────────┐
│ [Upload CSV/Excel]   ││ You: which city had the highest revenue in 2025? │
│ or [Sample: sales ▾] ││ Agent: Jakarta, Rp 1.24 B (31% of total).        │
│                      ││   [table: revenue by city]  [bar chart]          │
│ Dataset: sales.csv   ││   ▸ Steps (2): filter_rows → group_aggregate     │
│ 5,000 rows × 9 cols  ││                                                  │
│ Versions: v1 → v2    ││ You: clean the missing values                    │
│ [Undo] [Download]    ││ Agent: Filled 42 missing "price" with median…    │
│ [Export report]      ││                                                  │
│ Model: qwen2.5:3b    ││ [ Ask about your data…                     ]     │
└──────────────────────┘└──────────────────────────────────────────────────┘
```

Each answer shows: short text answer, the result table and/or chart, and an expandable **Steps** trace
(each tool called, its arguments, how long it took).

## 6. Architecture

```
data-analyst-agent/
  app.py              Streamlit UI (chat, upload, versions, export)
  agent/
    loop.py           agent loop: Ollama chat with tools, max 6 steps, retries on tool errors
    tools.py          the analysis tools (plain pandas/numpy/matplotlib functions + JSON schemas)
    sandbox.py        fallback "run_python" executor (restricted, subprocess, timeout)
    state.py          dataset versions (original + each cleaning step), undo
    report.py         HTML report export
  samples/            bundled datasets (synthetic Indonesian retail sales + 2 open datasets)
  eval/
    tasks.yaml        ~40 questions with expected answers
    run_eval.py       success rate + latency per model
  tests/              pytest for every tool, sandbox, state
  .env.example        OLLAMA_MODEL, OLLAMA_HOST
```

### 6.1 Agent loop
1. System prompt = role + dataset schema (column names, types, 5 sample rows) — never the whole dataset,
   so it stays small and fast.
2. Model replies with tool calls → run them → feed results back (tables truncated to ~20 rows).
3. Tool error → error message fed back so the model fixes its arguments (self-correction).
4. Stop at a final text answer or after 6 steps.
5. Answer language follows the question (English / Indonesian).

### 6.2 Tools (the core)
One function per goal in §2: `profile`, `clean_missing`, `drop_duplicates`, `convert_types`,
`rename_drop_columns`, `filter_rows`, `sort_top_n`, `group_aggregate`, `pivot`, `describe`,
`value_counts`, `correlation`, `outliers`, `time_trend`, `forecast`, `chart`.
Each validates its arguments (column exists, numeric where needed) and returns a clear error otherwise.

### 6.3 Fallback `run_python`
For questions no tool covers. Model-written pandas code, which is checked before running:
- runs in a separate process with a timeout (10 s), on a copy of the data;
- AST check blocks imports other than pandas/numpy/math, and blocks `open`, `exec`, `eval`, `__` attributes, file/network/OS access.
This is a guard for mistakes, not a security boundary against a hostile user — fine for a local, single-user app.

## 7. Evaluation (the CV numbers)
- ~40 tasks across the sample datasets, each with an expected answer (number, top item, or column set).
- `python eval/run_eval.py --model qwen2.5:3b` → success rate, average steps, average seconds per task.
- Run for 2–3 models (e.g. qwen2.5:1.5b / 3b / 7b) → a table in the README:
  "3B model: X% of tasks correct, ~Y s per task on a GTX 1650".

## 8. Testing
- Every tool: unit tests on small DataFrames (normal case + bad argument).
- Sandbox: blocked code is rejected; timeout works.
- State: clean → undo restores the original.
- Agent loop: tested with a fake model that returns scripted tool calls (no Ollama needed in tests).

## 9. Build order
1. Tools + tests.  2. State/versions.  3. Agent loop with fake-model tests.  4. Streamlit UI.
5. Sandbox fallback.  6. Sample datasets + eval + model comparison.  7. Report export.
8. README: setup for Windows and macOS, demo GIF, eval table.

## 10. Calls made by Claude (check these)
- Repo `data-analyst-agent`, **public** (it's a portfolio piece; contains no personal data).
- Streamlit for the UI (least code for chat + upload + charts; works the same on Windows and Mac).
- Default model `qwen2.5:3b`, confirmed or changed by the eval.
- Matplotlib for charts (static, easy to embed in the report).
- Bundled samples: a synthetic Indonesian retail sales dataset (generated by a script, with deliberate
  missing values/duplicates/outliers to show cleaning) + open datasets with permissive licences.

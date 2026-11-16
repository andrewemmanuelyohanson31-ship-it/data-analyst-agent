# 📊 Data Analyst Agent

An AI agent that does the everyday work of a data analyst on your CSV/Excel files — cleaning, summarising,
grouping, finding outliers, trends, forecasts and charts — answering questions in **English or Indonesian**.

It runs **100% offline on a normal laptop** (tested target: GTX 1650 4 GB / 16 GB RAM, and MacBook) using a
small local model through [Ollama](https://ollama.com). No API key, no data leaves your machine.

> *“Which city had the highest revenue in 2025?”* → **Jakarta, Rp 811.6 million (38.2% of total)** + table + bar chart + the steps it took.

## Why it works on a small model

Small models (3B parameters) are bad at writing long pandas code, but good at **choosing a tool and filling
in its arguments**. So the agent gets 16 tested analysis tools and only writes its own code as a last resort:

```
question ─► model picks a tool + arguments ─► tested pandas code runs ─► result back to model ─► answer
                     ▲                                   │
                     └──── error message (self-correct) ◄┘     (max 6 steps)
```

**No made-up numbers:** every number in the model's answer is checked against what the tools actually
returned (English and Indonesian number formats, "juta"/"miliar"/"billion"). If one doesn't match, the answer
is replaced by the tool's own verified summary, with a warning. Every answer also shows the table/chart it came from.

## What it can do

| Job | Tools |
|---|---|
| Profile data | `profile` — types, missing values, duplicates |
| Clean data | `clean_missing`, `drop_duplicates`, `convert_types`, `rename_drop_columns` (each creates a new version — **undo** anytime, original never changed) |
| Filter & rank | `filter_rows`, `sort_top_n` |
| Aggregate | `group_aggregate` (with share %), `pivot` |
| Describe | `describe`, `value_counts` |
| Relationships | `correlation`, `outliers` (IQR / z-score) |
| Time | `time_trend` (day/week/month/quarter/year, growth %), `forecast` (linear / moving average) |
| Charts | `chart` — bar, line, histogram, scatter, box, heatmap |
| Anything else | `run_python` — model-written pandas in a separate process with a 10 s limit and an import/file-access blocklist |
| Share | Export the whole session as one HTML report; download the cleaned CSV |

## Setup

**1. Install Ollama** — https://ollama.com/download (Windows or macOS), then:

```bash
ollama pull qwen2.5:3b
```

**2. Install the app**

Windows (PowerShell):
```powershell
git clone https://github.com/andrewemmanuelyohanson31-ship-it/data-analyst-agent.git
cd data-analyst-agent
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

macOS:
```bash
git clone https://github.com/andrewemmanuelyohanson31-ship-it/data-analyst-agent.git
cd data-analyst-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**3. Run**
```bash
streamlit run app.py
```
Your browser opens. Upload a CSV/Excel file, or pick a sample dataset in the sidebar.

Try:
- *Give me an overview of this data*
- *Clean the missing values and remove duplicates*
- *Which city has the highest total revenue?*
- *Show the monthly revenue trend as a line chart*
- *Forecast revenue for the next 3 months*
- *Kota mana dengan penjualan tertinggi?*

On a stronger machine, use a bigger model: `ollama pull qwen2.5:7b` and type it in the sidebar's model box
(or set `OLLAMA_MODEL`).

## Sample datasets

`samples/make_samples.py` generates two seeded synthetic datasets:
- `sales_id.csv` — 3,025 Indonesian retail orders (7 cities, 4 categories, IDR prices), with **deliberate dirt**:
  46 missing prices, 30 missing ages, 25 duplicate rows, 6 bulk-order outliers.
- `employees.csv` — 400 employees (department, salary, experience, performance, attrition).

## Evaluation

40 questions with known answers (25 sales, 15 HR, 3 in Indonesian) in `eval/tasks.json`.
Expected answers are pandas expressions, so they're always computed from the actual data. For "which X…"
questions the name must be right **and** any figure quoted must be X's real value.

```bash
python eval/run_eval.py --model qwen2.5:3b
```

| Model | Size | Correct | Avg steps | Avg time / question | Hardware |
|---|---|---|---|---|---|
| qwen2.5:3b | 1.9 GB | **87.5%** (35/40) | 1.05 | 3.3 s | GTX 1650 4 GB, i5-9400F, 16 GB RAM |

**How it got there** — each step came from reading the agent's step traces on the failures:

| Version | Correct | What changed |
|---|---|---|
| First run (3 questions) | 33% | No tool for a single overall number; model did maths in its head |
| + `aggregate` tool, number verification, 8k context | 62.5% | Ollama's default 4k context was silently cutting off the tool list |
| + tolerate small-model slips | 87.5% | Filters carry over between steps; results state highest **and** lowest, filled **and** missing; `column`/`columns` and list-wrapped names accepted; year filters on date columns |

Remaining failures: picking an exact date instead of a whole year, quoting the grand total instead of the
group's value, and one non-answer.

## Tests

```bash
pytest
```
60 tests: every tool (normal + bad input + small-model slips), dataset versions/undo, the agent loop (with a
scripted fake model: self-correction, number verification, filter carry-over, step cap), the code sandbox
(blocked imports/file access, timeout), the report, the eval answer checker, and the Streamlit app driven headlessly.

## Project structure

```
app.py              Streamlit UI
agent/tools.py      the analysis tools + their JSON schemas
agent/loop.py       agent loop (Ollama tool calling, self-correction, guards)
agent/sandbox.py    restricted fallback code runner
agent/state.py      dataset versions + undo
agent/report.py     HTML report export
eval/               40-question benchmark
samples/            sample dataset generator
docs/superpowers/   design and implementation plan
```

## Limits

- The code sandbox protects against model mistakes, not a hostile user — it's built for local, single-user use.
- Very large files (> ~1M rows) will be slow; the model only ever sees the schema and ≤ 20 result rows.
- A 3B model sometimes picks the wrong tool on ambiguous questions; the Steps panel shows exactly what it did.

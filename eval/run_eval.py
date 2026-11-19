"""Measure how often the agent answers correctly. Run: python eval/run_eval.py --model qwen2.5:3b

Each task's `expected` is a pandas expression on the raw dataset, so answers stay right if samples are regenerated.
An answer is correct if it contains the expected text, or any number in it is within 1% of the expected number.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.numbers import close, numbers_in  # noqa: E402  (needs ROOT on sys.path)


def load_dataset(name: str) -> pd.DataFrame:
    """Load a sample exactly as the app does (tidying included), from samples/ or samples/messy/."""
    from agent.tools import load_file

    path = next(p for p in (ROOT / "samples" / name, ROOT / "samples" / "messy" / name) if p.exists())
    return load_file(name, path.read_bytes())


def is_correct(answer: str, expected) -> bool:
    options = expected if isinstance(expected, list) else [expected]
    for exp in options:
        if isinstance(exp, (int, float, np.integer, np.floating)) and not isinstance(exp, bool):
            if any(close(n, float(exp)) for n in numbers_in(answer)):
                return True
        elif str(exp).lower() in answer.lower():
            return True
    return False


def main():
    from agent.loop import OllamaModel, run_agent
    from agent.state import DatasetState

    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen2.5:3b")
    p.add_argument("--limit", type=int, default=0, help="run only the first N tasks")
    args = p.parse_args()

    tasks = json.loads((ROOT / "eval" / "tasks.json").read_text(encoding="utf-8"))
    tasks = tasks[: args.limit] if args.limit else tasks
    model = OllamaModel(args.model)
    model.chat([{"role": "user", "content": "hi"}], [])  # warm-up: loading the model into memory isn't answer time
    data = {}
    rows = []
    for i, t in enumerate(tasks, 1):
        if t["dataset"] not in data:
            data[t["dataset"]] = load_dataset(t["dataset"])
        df = data[t["dataset"]]
        expected = eval(t["expected"], {"df": df, "pd": pd, "np": np})  # trusted: expressions from our own tasks.json
        t0 = time.perf_counter()
        try:
            ans = run_agent(t["question"], DatasetState(df.copy(), t["dataset"]), model)
            text, steps = ans.text, len(ans.steps)
        except Exception as e:
            text, steps = f"CRASH: {e}", 0
        secs = time.perf_counter() - t0
        ok = is_correct(text, expected)
        if ok and t.get("value") and numbers_in(text):  # "which X" answers: a quoted figure must also be X's real value
            ok = is_correct(text, eval(t["value"], {"df": df, "pd": pd, "np": np}))
        rows.append({"question": t["question"], "expected": str(expected), "answer": text, "ok": ok, "steps": steps,
                     "seconds": round(secs, 1), "core": t.get("core", False), "heldout": t.get("heldout", False)})
        print(f"[{i}/{len(tasks)}] {'PASS' if ok else 'FAIL'} {secs:5.1f}s  {t['question']}\n      -> {text[:140]}")

    res = pd.DataFrame(rows)
    subset = lambda m: f"{int(res.ok[m].sum())}/{int(m.sum())}"  # noqa: E731
    summary = {"model": args.model, "tasks": len(res), "success_pct": round(res.ok.mean() * 100, 1),
               "core": subset(res.core), "core_heldout": subset(res.heldout), "other": subset(~res.core),
               "avg_steps": round(res.steps.mean(), 2), "avg_seconds": round(res.seconds.mean(), 1)}
    print("\n" + json.dumps(summary, indent=2))
    out = ROOT / "eval" / "results"
    out.mkdir(exist_ok=True)
    (out / f"{args.model.replace(':', '_')}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

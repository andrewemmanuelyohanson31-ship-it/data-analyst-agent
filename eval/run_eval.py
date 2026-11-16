"""Measure how often the agent answers correctly. Run: python eval/run_eval.py --model qwen2.5:3b

Each task's `expected` is a pandas expression on the raw dataset, so answers stay right if samples are regenerated.
An answer is correct if it contains the expected text, or any number in it is within 1% of the expected number.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SCALES = {"ribu": 1e3, "thousand": 1e3, "k": 1e3, "juta": 1e6, "million": 1e6, "m": 1e6,
          "miliar": 1e9, "milyar": 1e9, "billion": 1e9, "bn": 1e9, "b": 1e9}
NUM = re.compile(r"(\d[\d.,]*)\s*(ribu|thousand|juta|million|miliar|milyar|billion|bn|[kmb])?(?![a-z])", re.I)


def numbers_in(text: str) -> list[float]:
    """All plausible readings of each number: '1,234.5' (English) and '1.234,5' (Indonesian), with scale words."""
    out = []
    for token, scale in NUM.findall(text):
        token = token.rstrip(".,")
        mult = SCALES.get(scale.lower(), 1) if scale else 1
        for cand in (token.replace(",", ""), token.replace(".", "").replace(",", ".")):
            try:
                out.append(float(cand) * mult)
            except ValueError:
                pass
    return out


def is_correct(answer: str, expected) -> bool:
    options = expected if isinstance(expected, list) else [expected]
    for exp in options:
        if isinstance(exp, (int, float, np.integer, np.floating)) and not isinstance(exp, bool):
            exp = float(exp)
            if any(abs(n - exp) <= max(abs(exp) * 0.01, 1e-9) for n in numbers_in(answer)):
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
    data = {}
    rows = []
    for i, t in enumerate(tasks, 1):
        if t["dataset"] not in data:
            data[t["dataset"]] = pd.read_csv(ROOT / "samples" / t["dataset"])
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
        rows.append({"question": t["question"], "expected": str(expected), "answer": text, "ok": ok, "steps": steps, "seconds": round(secs, 1)})
        print(f"[{i}/{len(tasks)}] {'PASS' if ok else 'FAIL'} {secs:5.1f}s  {t['question']}\n      -> {text[:140]}")

    res = pd.DataFrame(rows)
    summary = {"model": args.model, "tasks": len(res), "success_pct": round(res.ok.mean() * 100, 1),
               "avg_steps": round(res.steps.mean(), 2), "avg_seconds": round(res.seconds.mean(), 1)}
    print("\n" + json.dumps(summary, indent=2))
    out = ROOT / "eval" / "results"
    out.mkdir(exist_ok=True)
    (out / f"{args.model.replace(':', '_')}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

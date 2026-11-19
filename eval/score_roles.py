"""Score column-role detection against eval/column_roles.json.

Run: python eval/score_roles.py [--real-dir D:/dataset] [--names-only]
Each file is loaded exactly as the app loads it, then every column's detected role is compared with the key.
A column that doesn't come out of loading (e.g. the header row wasn't found) counts as wrong.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.column_names import role_of  # noqa: E402
from agent.tools import load_file  # noqa: E402

KEY = json.loads((ROOT / "eval" / "column_roles.json").read_text(encoding="utf-8"))
DIRS = {"samples": ROOT / "samples", "messy": ROOT / "samples" / "messy",
        "heldout": ROOT / "samples" / "messy" / "heldout"}


def detector(names_only: bool):
    if names_only:
        return lambda df: {c: role_of(c) for c in df.columns}
    from agent.roles import detect_roles  # noqa: PLC0415 - optional until it exists
    return detect_roles


def score(detect, real_dir: Path | None, verbose=True, groups=None):
    total = right = 0
    misses = []
    for group, files in KEY.items():
        if group.startswith("_") or (groups and group not in groups):
            continue
        folder = real_dir if group == "real" else DIRS[group]
        for name, truth in files.items():
            path = folder / name if folder else None
            if path is None or not path.exists():
                print(f"  skip {name} (not found)")
                continue
            df = load_file(name, path.read_bytes())
            got = detect(df)
            ok = 0
            for col, want in truth.items():
                allowed = set(want if isinstance(want, list) else [want])
                found = got.get(col, "MISSING") if col in df.columns else "MISSING"
                found = "other" if found is None else found
                if found in allowed:
                    ok += 1
                else:
                    misses.append((name, col, found, "/".join(sorted(allowed))))
            total += len(truth)
            right += ok
            if verbose:
                print(f"  {ok:2d}/{len(truth):2d}  {name}")
    print(f"\nColumns with the right role: {right}/{total} ({right / max(total, 1):.1%})")
    if verbose and misses:
        print("\nWrong:")
        for name, col, found, want in misses:
            print(f"  {name:28s} {col!r:26} got {found:15s} want {want}")
    return right, total, misses


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-dir", type=Path, help="folder with the public datasets listed under 'real'")
    ap.add_argument("--names-only", action="store_true", help="baseline: column-name lookup only")
    ap.add_argument("--heldout", action="store_true", help="score ONLY the held-out files (final check)")
    a = ap.parse_args()
    score(detector(a.names_only), a.real_dir,
          groups={"heldout"} if a.heldout else {"samples", "messy", "real"})

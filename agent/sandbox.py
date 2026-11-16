"""Fallback for questions no tool covers: run model-written pandas code in a separate, time-limited process.

shortcut: AST allow-list + subprocess guards against model mistakes, not a hostile user; fine for a local
single-user app, upgrade to a container/VM sandbox before ever hosting this for others.
"""
from __future__ import annotations

import ast
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

from agent.tools import ToolError, ToolResult

ALLOWED_IMPORTS = {"pandas", "numpy", "math"}
BLOCKED_NAMES = {"open", "exec", "eval", "compile", "__import__", "input", "globals", "locals", "vars",
                 "getattr", "setattr", "delattr", "breakpoint", "exit", "quit", "help"}
# Methods that touch files: to_csv("x") / read_csv etc. would let code read or write the disk.
BLOCKED_ATTRS = {"to_csv", "to_excel", "to_json", "to_pickle", "to_parquet", "to_sql", "to_html", "to_feather",
                 "read_csv", "read_excel", "read_json", "read_pickle", "read_parquet", "read_sql", "read_html",
                 "read_table", "read_fwf", "read_clipboard", "to_clipboard", "save", "load", "fromfile", "tofile"}

RUNNER = """
import pickle, sys
import pandas as pd, numpy as np, math
src, dst = sys.argv[1], sys.argv[2]
with open(src, "rb") as f:
    df, code = pickle.load(f)
scope = {"df": df, "pd": pd, "np": np, "math": math}
exec(code, scope)
if "result" not in scope:
    raise SystemExit("Code must assign the answer to a variable named result.")
with open(dst, "wb") as f:
    pickle.dump(scope["result"], f)
"""


def check_code(code: str) -> None:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ToolError(f"Syntax error: {e}")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            bad = [m for m in mods if m.split(".")[0] not in ALLOWED_IMPORTS]
            if bad:
                raise ToolError(f"Import of {', '.join(bad)} is not allowed. Only pandas, numpy, math.")
        elif isinstance(node, ast.Name) and node.id in BLOCKED_NAMES:
            raise ToolError(f"'{node.id}' is not allowed.")
        elif isinstance(node, ast.Attribute) and (node.attr.startswith("__") or node.attr in BLOCKED_ATTRS):
            raise ToolError(f"'.{node.attr}' is not allowed (no file access or dunder attributes).")


def run_python(df: pd.DataFrame, code: str, timeout: float = 10) -> ToolResult:
    check_code(code)
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.pkl", Path(tmp) / "out.pkl"
        src.write_bytes(pickle.dumps((df, code)))
        try:
            proc = subprocess.run([sys.executable, "-I", "-c", RUNNER, str(src), str(dst)],
                                  capture_output=True, text=True, timeout=timeout, cwd=tmp)
        except subprocess.TimeoutExpired:
            raise ToolError(f"Code took longer than {timeout:g} s and was stopped.")
        if proc.returncode != 0:
            err = (proc.stderr.strip().splitlines() or ["unknown error"])[-1]
            raise ToolError(f"Code failed: {err}")
        result = pickle.loads(dst.read_bytes())  # written by our own runner process above
    if isinstance(result, pd.Series):
        result = result.reset_index()
    if isinstance(result, pd.DataFrame):
        return ToolResult(f"Code returned a table with {len(result)} rows.", table=result)
    return ToolResult(f"Result: {result}")

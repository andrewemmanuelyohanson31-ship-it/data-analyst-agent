import pandas as pd
import pytest

from agent.sandbox import check_code, run_python
from agent.tools import ToolError, run_tool

DF = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "x"]})


def test_scalar_and_table():
    assert run_python(DF, "result = df['a'].sum()").text == "Result: 6"
    assert len(run_python(DF, "result = df.groupby('b')['a'].sum()").table) == 2


def test_registered_as_tool():
    assert run_tool("run_python", DF, {"code": "result = len(df)"}).text == "Result: 3"


@pytest.mark.parametrize("code", ["import os", "from subprocess import run", "open('x')", "result = df.__class__",
                                  "df.to_csv('x.csv')", "result = pd.read_csv('secret')", "eval('1')"])
def test_blocked(code):
    with pytest.raises(ToolError):
        check_code(code)


def test_timeout():
    with pytest.raises(ToolError, match="longer than"):
        run_python(DF, "while True: pass", timeout=2)


def test_missing_result_and_runtime_error():
    with pytest.raises(ToolError, match="named result"):
        run_python(DF, "x = 1")
    with pytest.raises(ToolError, match="Code failed"):
        run_python(DF, "result = df['nope']")

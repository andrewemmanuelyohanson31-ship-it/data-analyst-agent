import pandas as pd

from agent.loop import AgentAnswer, Step
from agent.report import build_report
from agent.tools import run_tool


def test_report_has_question_table_and_chart():
    df = pd.DataFrame({"city": ["A", "B"], "v": [1, 2]})
    steps = [Step("group_aggregate", {}, run_tool("group_aggregate", df, {"by": "city", "column": "v"})),
             Step("chart", {}, run_tool("chart", df, {"kind": "bar", "x": "city", "y": "v"}))]
    out = build_report("data.csv", [("Which <city>?", AgentAnswer("B is highest.", steps))])
    assert "Which &lt;city&gt;?" in out and "<table" in out and "data:image/png;base64" in out

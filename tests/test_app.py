"""Drives the real Streamlit script headlessly with a fake model (no Ollama needed)."""
from pathlib import Path

import agent.loop
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


class FakeOllama:
    def __init__(self, model=None, host=None):
        self.replies = [{"content": "", "tool_calls": [{"name": "group_aggregate", "arguments": {"by": ["city"], "column": "revenue", "agg": "sum"}}]},
                        {"content": "Jakarta has the highest revenue.", "tool_calls": []}]

    def chat(self, messages, tools):
        return self.replies.pop(0)


def test_sample_question_flow(monkeypatch):
    monkeypatch.setattr(agent.loop, "OllamaModel", FakeOllama)
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    at.selectbox[0].select("sales_id.csv").run()
    assert not at.exception and "3,025 rows" in at.sidebar.caption[0].value
    at.chat_input[0].set_value("which city has the highest revenue?").run()
    assert not at.exception
    assert any("**Jakarta**" in m.value for m in at.markdown)  # answered by the question parser

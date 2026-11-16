import pandas as pd

from agent.loop import AgentAnswer, run_agent, schema_summary
from agent.state import DatasetState


class FakeModel:
    """Replays scripted replies; records what it was sent."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.seen = []

    def chat(self, messages, tools):
        self.seen.append(list(messages))
        return self.replies.pop(0)


def call(name, **args):
    return {"content": "", "tool_calls": [{"name": name, "arguments": args}]}


def say(text):
    return {"content": text, "tool_calls": []}


def state():
    return DatasetState(pd.DataFrame({"city": ["Jakarta", "Bandung", "Jakarta"], "revenue": [10.0, None, 30.0]}), "t")


def test_tool_then_answer():
    m = FakeModel(call("group_aggregate", by=["city"], column="revenue", agg="sum"), say("Jakarta has 40."))
    ans = run_agent("top city?", state(), m)
    assert isinstance(ans, AgentAnswer) and ans.text == "Jakarta has 40."
    assert len(ans.steps) == 1 and ans.steps[0].error is None
    assert "Jakarta" in m.seen[1][-1]["content"]  # tool result fed back


def test_error_then_self_correct():
    m = FakeModel(call("group_aggregate", by=["kota"], column="revenue", agg="sum"),
                  call("group_aggregate", by=["city"], column="revenue", agg="sum"), say("Jakarta."))
    ans = run_agent("top city?", state(), m)
    assert [s.error is None for s in ans.steps] == [False, True]
    assert "Columns are" in m.seen[1][-1]["content"]


def test_numbers_without_tool_get_nudged_once():
    m = FakeModel(say("Revenue is 999."), call("describe"), say("Mean revenue is 20."))
    ans = run_agent("mean revenue?", state(), m)
    assert ans.text == "Mean revenue is 20." and len(ans.steps) == 1


def test_cleaning_creates_version():
    s = state()
    run_agent("fill missing", s, FakeModel(call("clean_missing", strategy="median"), say("Done.")))
    assert len(s.versions) == 2 and s.current["revenue"].isna().sum() == 0
    assert s.original["revenue"].isna().sum() == 1


def test_json_text_tool_call_is_parsed():
    m = FakeModel(say('{"name": "profile", "arguments": {}}'), say("3 rows."))
    assert len(run_agent("overview", state(), m).steps) == 1


def test_step_cap():
    m = FakeModel(*[call("profile")] * 6)
    ans = run_agent("loop", state(), m, max_steps=6)
    assert ans.text.startswith("(Stopped after 6 steps.)")


def test_schema_summary_lists_columns():
    s = schema_summary(state().current)
    assert "city" in s and "1 missing" in s

import pandas as pd
import pytest

import agent.loop
from agent.loop import AgentAnswer, run_agent, schema_summary
from agent.state import DatasetState


@pytest.fixture(autouse=True)
def no_router(request, monkeypatch):
    """These tests drive the model loop; the question parser would answer their simple questions first."""
    if "router" not in request.node.name:
        monkeypatch.setattr(agent.loop, "route", lambda question, df: None)


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


def test_garbled_number_replaced_by_tool_summary():
    m = FakeModel(call("aggregate", agg="sum", column="revenue"), say("Total revenue is $11,486,290.00."))
    ans = run_agent("total revenue?", state(), m)
    assert ans.text == "sum of 'revenue' over 3 rows: 40.00." and "11,486,290.00" in ans.note


def test_rounded_number_is_accepted():
    m = FakeModel(call("aggregate", agg="sum", column="revenue"), say("Total revenue is about 40."))
    assert run_agent("total revenue?", state(), m).text == "Total revenue is about 40."


def test_empty_answer_after_tool_uses_tool_text():
    m = FakeModel(call("aggregate"), say(""))
    assert run_agent("how many rows?", state(), m).text.startswith("Count of rows: 3")


def test_filter_carries_over_to_next_tool():
    m = FakeModel(call("filter_rows", where=[{"column": "city", "op": "==", "value": "Jakarta"}]),
                  call("aggregate", agg="sum", column="revenue"), say("Jakarta revenue is 40."))
    ans = run_agent("Jakarta revenue?", state(), m)
    assert "over 2 rows: 40.00" in ans.steps[1].result.text and ans.steps[1].args["where"][0]["value"] == "Jakarta"


def test_schema_summary_lists_columns():
    s = schema_summary(state().current)
    assert "city" in s and "1 missing" in s


def test_router_answers_frequent_question_without_model():
    model = FakeModel()  # no scripted replies: calling the model would fail the test
    ans = run_agent("Which city has the highest total revenue?", state(), model)
    assert "**Jakarta**" in ans.text and model.seen == []
    assert ans.steps[0].tool == "group_aggregate" and "question parser" in ans.steps[0].fixes[0]


def test_router_falls_back_to_model_when_unsure():
    model = FakeModel(call("profile"), say("The data has 3 rows."))
    ans = run_agent("Give me an overview of this data", state(), model)
    assert len(model.seen) == 2 and ans.steps[0].tool == "profile"


def test_question_cannot_change_the_data():
    model = FakeModel(call("drop_duplicates"), call("profile"), say("There are 3 rows."))
    st = state()
    ans = run_agent("How many duplicate rows are there?", st, model)
    assert ans.steps[0].error and "changes the dataset" in ans.steps[0].error
    assert len(st.versions) == 1  # nothing was removed


def test_asking_to_clean_may_change_the_data():
    model = FakeModel(call("drop_duplicates"), say("Done."))
    st = state()
    run_agent("Remove the duplicate rows", st, model)
    assert len(st.versions) == 2

import pandas as pd

from agent.state import DatasetState


def test_apply_and_undo():
    df = pd.DataFrame({"a": [1, 2]})
    s = DatasetState(df, "x")
    assert s.undo() is False
    s.apply("drop", df.head(1))
    assert len(s.current) == 1
    assert s.undo() is True and s.current is df

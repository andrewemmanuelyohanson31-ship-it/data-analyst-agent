import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
import make_data  # noqa: E402

from agent.tools import tool_schemas  # noqa: E402


def test_examples_are_verified_and_rendered_like_ollama():
    made = [e for e in map(make_data.example, range(1, 40)) if e]
    assert len(made) >= 20  # some question types don't fit some tables (no date, no groups)
    m = made[0]["messages"]
    assert [x["role"] for x in m] == ["system", "user", "assistant", "tool", "assistant"]
    text = make_data.render(m, tool_schemas())
    assert text.startswith("<|im_start|>system\n") and text.count("<tool_call>\n{\"name\": \"") == 1
    assert "<tool_response>\n" in text and text.endswith("<|im_end|>\n")

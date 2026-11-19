"""Streamlit UI. Run: streamlit run app.py"""
import os
from pathlib import Path

import streamlit as st

from agent.loop import DEFAULT_MODEL, AgentAnswer, OllamaModel, run_agent
from agent.report import build_report
from agent.state import DatasetState
from agent.tools import load_file

SAMPLES = Path(__file__).parent / "samples"
HISTORY_TURNS = 3  # earlier Q&A sent to the model for follow-ups; small to keep a 3B model fast

st.set_page_config(page_title="Data Analyst Agent", page_icon="📊", layout="wide")
ss = st.session_state


def load(name: str, data: bytes):
    ss.state = DatasetState(load_file(name, data), name)
    ss.turns = []
    ss.source = name


def show_answer(ans: AgentAnswer):
    st.markdown(ans.text)
    if ans.note:
        st.caption(f"⚠️ {ans.note}")
    for s in ans.steps:
        if s.result is not None and s.result.table is not None and s.result.new_df is None:
            st.dataframe(s.result.table.head(200), hide_index=True)
        if s.result is not None and s.result.chart:
            st.image(s.result.chart)
    if ans.steps:
        with st.expander(f"Steps ({len(ans.steps)}): " + " → ".join(s.tool for s in ans.steps)):
            for i, s in enumerate(ans.steps, 1):
                st.markdown(f"**{i}. `{s.tool}`** · {s.seconds:.2f}s")
                st.json(s.args, expanded=False)
                st.caption(f"Error: {s.error}" if s.error else s.result.text)


# ---------- sidebar ----------
with st.sidebar:
    st.header("📊 Data Analyst Agent")
    up = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
    if up is not None and ss.get("source") != up.name:
        try:
            load(up.name, up.getvalue())
        except Exception as e:  # unreadable file: tell the user instead of crashing the app
            st.error(f"Could not read {up.name}: {e}")
    samples = sorted(p.name for p in SAMPLES.glob("*.csv"))
    pick = st.selectbox("…or try a sample", ["—"] + samples)
    if pick != "—" and ss.get("source") != pick and up is None:
        load(pick, (SAMPLES / pick).read_bytes())

    if "state" in ss:
        state: DatasetState = ss.state
        st.caption(f"**{state.name}** · {len(state.current):,} rows × {state.current.shape[1]} columns")
        st.caption("Versions: " + " → ".join(label for label, _ in state.versions))
        c1, c2 = st.columns(2)
        if c1.button("Undo", disabled=len(state.versions) == 1, use_container_width=True):
            state.undo()
            st.rerun()
        c2.download_button("CSV", state.current.to_csv(index=False), f"cleaned_{Path(state.name).stem}.csv",
                           use_container_width=True)
        if ss.turns:
            st.download_button("Export report (HTML)", build_report(state.name, ss.turns),
                               f"report_{Path(state.name).stem}.html", use_container_width=True)
    model_name = st.text_input("Ollama model", value=os.getenv("OLLAMA_MODEL", DEFAULT_MODEL))

# ---------- chat ----------
if "state" not in ss:
    st.info("Upload a CSV/Excel file or pick a sample dataset in the sidebar to start.")
    st.markdown("Try: *“which city had the highest revenue?”*, *“clean the missing values”*, "
                "*“show monthly revenue trend”*, *“kota mana dengan penjualan tertinggi?”*")
    st.markdown("For a shop (sample **toko_sembako.csv**): *“produk apa yang harus segera di-restock?”*, "
                "*“hari apa penjualan paling sepi?”*, *“produk mana yang paling jarang terjual?”*")
    st.stop()

with st.expander("Preview data", expanded=not ss.turns):
    st.dataframe(ss.state.current.head(100), hide_index=True)

for q, ans in ss.turns:
    st.chat_message("user").markdown(q)
    with st.chat_message("assistant"):
        show_answer(ans)

if question := st.chat_input("Ask about your data…"):
    st.chat_message("user").markdown(question)
    history = [m for q, a in ss.turns[-HISTORY_TURNS:] for m in
               ({"role": "user", "content": q}, {"role": "assistant", "content": a.text})]
    with st.chat_message("assistant"):
        with st.spinner("Analysing…"):
            try:
                ans = run_agent(question, ss.state, OllamaModel(model_name), history)
            except Exception as e:  # Ollama not running / model not pulled
                st.error(f"Could not reach the model `{model_name}`: {e}\n\n"
                         f"Make sure Ollama is running and the model is downloaded: `ollama pull {model_name}`")
                st.stop()
        show_answer(ans)
    ss.turns.append((question, ans))
    st.rerun()

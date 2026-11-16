"""Export a chat session to one self-contained HTML file (charts embedded as base64)."""
import base64
import html
from datetime import datetime

from agent.loop import AgentAnswer

CSS = """body{font-family:system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;color:#1f2937}
h1{font-size:1.5rem}.q{margin-top:2rem;font-weight:600;border-left:4px solid #2563eb;padding-left:.6rem}
table{border-collapse:collapse;font-size:.85rem;margin:.6rem 0}td,th{border:1px solid #d1d5db;padding:.25rem .5rem;text-align:right}
th{background:#f3f4f6}img{max-width:100%}.steps{color:#6b7280;font-size:.8rem}"""


def build_report(dataset_name: str, turns: list[tuple[str, AgentAnswer]]) -> str:
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>Analysis report: {html.escape(dataset_name)}</title>",
             f"<style>{CSS}</style></head><body><h1>Analysis report: {html.escape(dataset_name)}</h1>",
             f"<p class='steps'>Generated {datetime.now():%Y-%m-%d %H:%M} by Data Analyst Agent</p>"]
    for question, ans in turns:
        parts.append(f"<p class='q'>{html.escape(question)}</p><p>{html.escape(ans.text)}</p>")
        for s in ans.steps:
            if s.result is None:
                continue
            if s.result.table is not None:
                parts.append(s.result.table.head(50).to_html(index=False, border=0))
            if s.result.chart:
                parts.append(f"<img alt='{html.escape(s.tool)} chart' src='data:image/png;base64,{base64.b64encode(s.result.chart).decode()}'>")
        if ans.steps:
            parts.append("<p class='steps'>Steps: " + " → ".join(html.escape(s.tool) for s in ans.steps) + "</p>")
    parts.append("</body></html>")
    return "\n".join(parts)

"""Read numbers out of free text, in English (1,234.5) and Indonesian (1.234,5) styles, with scale words."""
import re

SCALES = {"ribu": 1e3, "thousand": 1e3, "k": 1e3, "juta": 1e6, "million": 1e6, "m": 1e6,
          "miliar": 1e9, "milyar": 1e9, "billion": 1e9, "bn": 1e9, "b": 1e9}
# 1,234,567.8 | 1.234.567,8 | plain 12.5 / 12,5. Thousands groups must be exactly 3 digits so CSV rows
# like "20.0,14.14,10.0" split into separate numbers instead of merging into one.
NUM = re.compile(r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?(?!\d)|\d{1,3}(?:\.\d{3})+(?:,\d+)?(?!\d)|\d+(?:[.,]\d+)?)"
                 r"\s*(ribu|thousand|juta|million|miliar|milyar|billion|bn|[kmb])?(?![a-z])", re.I)


def numbers_in(text: str) -> list[float]:
    """All plausible readings of each number in the text (both decimal styles, scaled)."""
    out = []
    for t, scale in NUM.findall(text):
        mult = SCALES.get(scale.lower(), 1) if scale else 1
        if re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", t):  # English thousands: 1,234,567.8
            cands = [t.replace(",", "")] + ([t.replace(",", ".")] if t.count(",") == 1 and "." not in t else [])
        elif re.fullmatch(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?", t):  # Indonesian thousands: 1.234.567,8
            cands = [t.replace(".", "").replace(",", ".")] + ([t] if t.count(".") == 1 and "," not in t else [])
        else:  # plain: 12.5 or Indonesian decimal comma 12,5
            cands = [t.replace(",", ".")]
        out += [float(c) * mult for c in cands]
    return out


def close(a: float, b: float, rel: float = 0.01) -> bool:
    return abs(a - b) <= max(abs(b) * rel, 1e-9)


def ungrounded_numbers(answer: str, evidence: str) -> list[str]:
    """Numbers in the answer that match nothing in the evidence (tool results + question).

    Each number is checked on its own so one bad figure can't hide behind a good one.
    Small integers (<= 12) are allowed: "top 3", months, ordinals.
    """
    known = numbers_in(evidence)
    bad = []
    for m in NUM.finditer(answer):
        readings = numbers_in(m.group(0))
        if not readings or (readings[0] <= 12 and readings[0] == int(readings[0])):
            continue
        if not any(close(r, k) for r in readings for k in known):
            bad.append(m.group(0).strip())
    return bad

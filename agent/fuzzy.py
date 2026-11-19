"""Second chance for questions the parser can't read as written: chat abbreviations, typos, close synonyms.

    normalize(question, known, targets) -> (rewritten question, ["'hihgest' -> 'highest'", ...])

Only words the parser doesn't already know are touched, and only when the match is clear; the rewritten question
still has to pass the parser's own "every word understood" check, so a doubtful rewrite just goes to the model.
"""
from __future__ import annotations

import json
import math
import re
import urllib.request

from rapidfuzz import fuzz, process
from rapidfuzz.distance import OSA

# Indonesian chat abbreviations (fixed list: more reliable than a model for these)
SLANG = {"yg": "yang", "brp": "berapa", "rata2": "rata-rata", "utk": "untuk", "dgn": "dengan", "bln": "bulan",
         "thn": "tahun", "tgl": "tanggal", "jml": "jumlah", "tdk": "tidak", "gak": "tidak", "ga": "tidak",
         "krn": "karena", "dr": "dari", "aja": "saja", "pd": "pada", "sdh": "sudah", "blm": "belum",
         "hrg": "harga", "brg": "barang", "dlm": "dalam", "sbg": "sebagai", "tsb": "tersebut", "gmn": "bagaimana",
         "plg": "paling", "bnyk": "banyak", "byk": "banyak", "sblm": "sebelum"}
# OSA similarity 0-1 counts swapped neighbours as ONE typo: "weigth"->"weight" 0.83 vs "with" 0.67
TYPO_MIN, TYPO_MARGIN = 0.80, 0.08
SYN_MIN, SYN_MARGIN = 0.60, 0.10  # cosine: "income"->"revenue" 0.66 vs 0.36 passes; "prefer"->"least" 0.44 doesn't
# real words with their own meaning: never "correct" them ("more"->"most" and "median"->"mean" changed answers)
PROTECTED = {"more", "less", "fewer", "median", "mode", "than", "between", "each", "every", "except", "without",
             "over", "under", "above", "below", "first", "last", "next", "previous", "same", "other", "only",
             "lebih", "kurang", "selain", "tiap", "setiap", "antara", "kecuali"}
EMBED_URL, EMBED_MODEL ="http://localhost:11434/api/embed", "all-minilm"
_VECTORS: dict[str, list[float]] = {}


def _embed(words: list[str]) -> list[list[float]] | None:
    todo = [w for w in words if w not in _VECTORS]
    if todo:
        try:
            req = urllib.request.Request(EMBED_URL, json.dumps({"model": EMBED_MODEL, "input": todo}).encode())
            with urllib.request.urlopen(req, timeout=10) as resp:
                _VECTORS.update(zip(todo, json.loads(resp.read())["embeddings"]))
        except Exception:  # noqa: BLE001 - no Ollama or no embedding model: skip synonyms, typos still work
            return None
    return [_VECTORS[w] for w in words]


def _cos(a, b) -> float:
    return sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a) * sum(y * y for y in b)) or 1)


def _synonym(word: str, targets: list[str]) -> str | None:
    vecs = _embed([word, *targets])
    if not vecs or len(targets) < 2:
        return None
    scores = sorted(((_cos(vecs[0], v), t) for v, t in zip(vecs[1:], targets)), reverse=True)
    (best, target), (second, _) = scores[0], scores[1]
    return target if best >= SYN_MIN and best - second >= SYN_MARGIN else None


def normalize(question: str, known: set[str], targets: set[str], use_embeddings: bool = True):
    """Rewrite unknown words; returns (question, notes). Notes are empty when nothing changed."""
    vocab = sorted(known)
    notes, out = [], question
    for word in dict.fromkeys(re.findall(r"[A-Za-z][A-Za-z0-9-]*", question)):
        w = word.lower()
        if w in known or w in PROTECTED:
            continue
        new = SLANG.get(w)
        if new is None and len(w) >= 4:
            # two measures catch different typos: OSA counts swapped letters as one ("weigth"), the plain ratio
            # separates a missing letter better ("wich" -> "which", not "with"); either must give a CLEAR winner
            for scorer in (OSA.normalized_similarity, lambda a, b, **_: fuzz.ratio(a, b) / 100):
                hits = process.extract(w, vocab, scorer=scorer, limit=2)
                if hits and hits[0][1] >= TYPO_MIN and (len(hits) < 2 or hits[0][1] - hits[1][1] >= TYPO_MARGIN):
                    new = hits[0][0]
                    break
        if new is None and use_embeddings and len(w) >= 4:
            new = _synonym(w, sorted(targets))
        if new and new != w:
            out = re.sub(rf"(?<![\w-]){re.escape(word)}(?![\w-])", new, out)
            notes.append(f"'{word}' → '{new}'")
    return out, notes

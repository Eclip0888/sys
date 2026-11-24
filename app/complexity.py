"""Answer complexity features (Chapter 3, Mathematical Model 2).

FKGL   Flesch-Kincaid Grade Level of the answer [16]
LEN    natural log of the answer's word count
TECH   technical-term ratio: share of words of two or more syllables that are not on the Dale-Chall list of
       familiar English words (data/dale_chall_easy_words.txt, from the textstat package, MIT licence),
       i.e. words "not found in a list of common English words"
LEVEL  cognitive level of the generated question (1 recall, 2 understanding, 3 application/inference),
       added after the question is generated
"""
from __future__ import annotations

import math
import os
import re
from functools import lru_cache

FEATURES = ("fkgl", "len", "tech", "level")


def plain_text(answer: str) -> str:
    """Strip Markdown and code so readability is measured on the prose the user reads."""
    t = re.sub(r"```.*?```", " ", answer, flags=re.S)          # code blocks
    t = re.sub(r"`[^`]*`", " ", t)                             # inline code
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", t)                # images
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)             # links -> text
    t = re.sub(r"^\s{0,3}#{1,6}\s*", "", t, flags=re.M)        # headings
    t = re.sub(r"^\s*([-*+]|\d+[.)])\s+", "", t, flags=re.M)   # list markers
    t = re.sub(r"[*_~>|]", " ", t)                             # emphasis, quotes, tables
    lines = [ln.strip() for ln in t.splitlines()]
    # treat each non-empty line as a sentence if it lacks end punctuation (lists, headings)
    lines = [ln if re.search(r"[.!?:]$", ln) else ln + "." for ln in lines if ln]
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


WORD = re.compile(r"[A-Za-z][A-Za-z'-]*|\d+(?:\.\d+)?")


@lru_cache(maxsize=1)
def easy_words() -> frozenset:
    path = os.path.join(os.path.dirname(__file__), "..", "data", "dale_chall_easy_words.txt")
    with open(path, encoding="utf-8") as f:
        return frozenset(w.strip().lower() for w in f if w.strip())


def syllables(word: str) -> int:
    """Heuristic English syllable count: vowel groups, minus silent endings (-e, -ed, -es)."""
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 1                                   # numbers count as one syllable
    if len(w) <= 3:
        return 1
    n = len(re.findall(r"[aeiouy]+", w))
    n += len(re.findall(r"[aeiou](?:a|o)(?!u)", w)) if re.search(r"(ia|io|ea|eo)", w) and not w.endswith(("tion", "sion", "cian")) else 0
    if w.endswith("e") and not w.endswith(("le", "ee", "ie", "ye")):
        n -= 1
    elif w.endswith("ed") and not w.endswith(("ted", "ded")):
        n -= 1
    elif w.endswith("es") and not w.endswith(("ses", "zes", "ces", "ges", "xes", "shes", "ches")):
        n -= 1
    return max(1, n)


def word_count(text: str) -> int:
    return len(WORD.findall(text))


def fkgl(text: str) -> float:
    """Flesch-Kincaid Grade Level: 0.39 (words / sentences) + 11.8 (syllables / words) - 15.59 [16]."""
    words = WORD.findall(text)
    sentences = max(1, len([s for s in re.split(r"[.!?]+", text) if WORD.search(s)]))
    syl = sum(syllables(w) for w in words)
    return 0.39 * len(words) / sentences + 11.8 * syl / max(len(words), 1) - 15.59


def technical_ratio(text: str) -> float:
    words = [w.lower().strip("'-") for w in WORD.findall(text) if not w[0].isdigit()]
    easy = easy_words()
    hard = [w for w in words if w not in easy and syllables(w) >= 2]
    return len(hard) / max(len(words), 1)


def answer_features(answer: str) -> dict:
    text = plain_text(answer)
    words = max(word_count(text), 1)
    if words < 3:
        return {"fkgl": 0.0, "len": math.log(words), "tech": 0.0, "words": words}
    return {"fkgl": round(fkgl(text), 3), "len": round(math.log(words), 4), "tech": round(technical_ratio(text), 4),
            "words": words}


def question_length(question: str, options: list[str]) -> int:
    """L_j: words in the question and its options (used for the expected reading time)."""
    return word_count(question) + sum(word_count(o) for o in options)

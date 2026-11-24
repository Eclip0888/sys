"""Question generation with validation and the self-consistency check (Chapter 3, Step 3)."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

MAX_ATTEMPTS = 3


@dataclass
class QuestionResult:
    ok: bool
    question: str = ""
    options: list = field(default_factory=list)
    key: int = -1
    level: int = 0
    attempts: int = 0
    valid_outputs: int = 0          # attempts whose output passed the format check
    first_try: bool = False         # accepted on the first attempt
    seconds: float = 0.0
    error: str = ""


def parse_question(text: str):
    """Return (question, options, key, level) or raise ValueError if the output is not valid."""
    d = json.loads(text)
    q = str(d.get("question", "")).strip()
    opts = [str(o).strip() for o in d.get("options", [])]
    key, level = int(d.get("answer_index", -1)), int(d.get("level", 0))
    if not q or len(opts) != 4 or any(not o for o in opts) or len({o.lower() for o in opts}) != 4:
        raise ValueError("question must have text and 4 different, non-empty options")
    if not 0 <= key <= 3 or level not in (1, 2, 3):
        raise ValueError("answer_index must be 0-3 and level 1-3")
    return q, opts, key, level


def parse_check(text: str) -> int:
    return int(json.loads(text).get("answer_index", -1))


def generate_question(client, prompt: str, answer: str) -> QuestionResult:
    """Up to 3 attempts; accept a question only if its output is valid AND Gemini, answering from the text
    alone, picks the same option as the key."""
    t0 = time.monotonic()
    res = QuestionResult(ok=False)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        res.attempts = attempt
        try:
            q, opts, key, level = parse_question(client.question(prompt, answer))
        except (ValueError, TypeError, json.JSONDecodeError) as e:
            res.error = f"invalid output: {e}"
            continue
        res.valid_outputs += 1
        try:
            picked = parse_check(client.check(answer, q, opts))
        except (ValueError, TypeError, json.JSONDecodeError) as e:
            res.error = f"invalid check output: {e}"
            continue
        if picked == key:
            res.ok, res.question, res.options, res.key, res.level = True, q, opts, key, level
            res.first_try = attempt == 1
            res.error = ""
            break
        res.error = f"self-consistency failed (key {key}, check picked {picked})"
    res.seconds = round(time.monotonic() - t0, 2)
    return res

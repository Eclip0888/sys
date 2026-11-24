"""Question generation: validation and self-consistency (Table 8, Correctness and Robustness)."""
import json

import pytest

from app.gemini import FakeGemini
from app.qgen import generate_question, parse_question


def q(**kw):
    d = {"question": "What?", "options": ["a", "b", "c", "d"], "answer_index": 1, "level": 2}
    d.update(kw)
    return json.dumps(d)


def test_parse_valid():
    assert parse_question(q()) == ("What?", ["a", "b", "c", "d"], 1, 2)


@pytest.mark.parametrize("bad", [q(options=["a", "b", "c"]), q(options=["a", "a", "c", "d"]), q(answer_index=4),
                                 q(level=0), q(question=""), "{not json"])
def test_parse_rejects_invalid(bad):
    with pytest.raises(Exception):
        parse_question(bad)


def test_fake_question_accepted_first_try():
    r = generate_question(FakeGemini(), "What is an algorithm?", "An algorithm is a list of steps.")
    assert r.ok and r.first_try and r.attempts == 1 and len(r.options) == 4


def test_self_consistency_failure_retries_three_times():
    r = generate_question(FakeGemini(), "prompt", "inconsistent answer text")
    assert not r.ok and r.attempts == 3 and r.valid_outputs == 3 and "self-consistency" in r.error


def test_invalid_output_is_counted():
    r = generate_question(FakeGemini(), "broken-json prompt", "text")
    assert not r.ok and r.valid_outputs == 0 and r.attempts == 3


class Flaky(FakeGemini):
    """First output invalid, second valid."""
    def __init__(self):
        self.n = 0

    def question(self, prompt, answer):
        self.n += 1
        return "{bad" if self.n == 1 else super().question(prompt, answer)


def test_recovers_on_second_attempt():
    r = generate_question(Flaky(), "p", "a")
    assert r.ok and r.attempts == 2 and not r.first_try

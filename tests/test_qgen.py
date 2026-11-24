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


def test_thinking_config_per_model():
    from app.gemini import DEFAULT_MODEL, GeminiClient
    c3 = GeminiClient("dummy-key", model="gemini-3.8-flash")
    assert c3._config().thinking_config.thinking_level is not None
    c25 = GeminiClient("dummy-key", model="gemini-2.5-flash")
    assert c25._config().thinking_config.thinking_budget == 0
    c3.use_thinking_config = False
    assert c3._config().thinking_config is None
    assert GeminiClient("dummy-key", model="").model == DEFAULT_MODEL


class _Resp:
    text = "an answer"


def _client_with(errors_by_model):
    """GeminiClient whose fake transport raises the given HTTP codes per model, then succeeds."""
    from google.genai import errors

    from app.gemini import GeminiClient
    c = GeminiClient("dummy-key", model="main-model")
    c.models = ["main-model", "backup-model"]
    c.limiter.interval = 0
    calls = []

    class Models:
        def generate_content(self, model, contents, config):
            calls.append(model)
            codes = errors_by_model.get(model, [])
            if codes:
                code = codes.pop(0)
                raise errors.APIError(code, {"error": {"code": code, "message": "busy", "status": "X"}})
            return _Resp()

    c.client = type("C", (), {"models": Models()})()
    return c, calls


def test_busy_model_falls_back(monkeypatch):
    import time as _t
    monkeypatch.setattr(_t, "sleep", lambda s: None)
    c, calls = _client_with({"main-model": [503, 503]})
    assert c.answer("hi") == "an answer"
    assert calls == ["main-model", "main-model", "backup-model"] and c.last_model == "backup-model"


def test_quota_moves_to_next_model_immediately():
    c, calls = _client_with({"main-model": [429]})
    c.answer("hi")
    assert calls == ["main-model", "backup-model"]


def test_all_models_busy_gives_clear_error(monkeypatch):
    import time as _t

    import pytest as _p

    from app.gemini import GeminiError
    monkeypatch.setattr(_t, "sleep", lambda s: None)
    c, _ = _client_with({"main-model": [503, 503], "backup-model": [503, 503]})
    with _p.raises(GeminiError, match="busy or unavailable"):
        c.answer("hi")


def _groq(responses):
    """OpenAICompatClient with a fake HTTP transport: responses is a list of (status, json) per request."""
    import httpx

    from app.gemini import OpenAICompatClient
    seen = []

    def handler(request):
        import json as _j
        seen.append(_j.loads(request.content)["model"])
        status, body = responses.pop(0)
        return httpx.Response(status, json=body)

    c = OpenAICompatClient("k", ["model-a", "model-b"], http=httpx.Client(transport=httpx.MockTransport(handler)))
    c.limiter.interval = 0
    return c, seen


def _ok(text):
    return (200, {"choices": [{"message": {"content": text}}]})


def test_groq_answers_and_skips_retired_model():
    c, seen = _groq([(404, {"error": "model not found"}), _ok("hello")])
    assert c.answer("hi") == "hello" and seen == ["model-a", "model-b"] and c.last_model == "groq/model-b"


def test_groq_question_json_passes_validation():
    from app.qgen import parse_question
    c, _ = _groq([_ok(q())])
    assert parse_question(c.question("p", "a"))[2] == 1


def test_chain_uses_backup_when_gemini_fails():
    from app.gemini import ChainClient, GeminiError

    class Down:
        model = "gemini-x"

        def answer(self, prompt):
            raise GeminiError("busy")

    backup, _ = _groq([_ok("from groq")])
    chain = ChainClient([Down(), backup])
    assert chain.answer("hi") == "from groq" and chain.last_model == "groq/model-a"
    assert "groq backup" in chain.model


def test_qgen_through_chain_with_backup():
    from app.gemini import ChainClient, GeminiError
    from app.qgen import generate_question

    class Down:
        model = "gemini-x"

        def question(self, *a):
            raise GeminiError("busy")

        def check(self, *a):
            raise GeminiError("busy")

    backup, _ = _groq([_ok(q()), _ok('{"answer_index": 1}')])
    r = generate_question(ChainClient([Down(), backup]), "p", "a")
    assert r.ok and r.key == 1

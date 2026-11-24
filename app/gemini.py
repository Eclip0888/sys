"""Google Gemini calls: the AI answer, the generated question and the self-consistency check.

GEMINI_API_KEY   your key from Google AI Studio (never commit it)
GEMINI_MODEL     model name, default "gemini-3.8-flash" (see ai.google.dev/gemini-api/docs/models)
GEMINI_RPM       maximum requests per minute (default 8) so the free tier is not exceeded

Without a key the app uses FakeGemini, an offline stand-in with fixed answers, so the website,
the tests and the simulation can run before the key is set.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time

from pydantic import BaseModel

ANSWER_SYSTEM = ("You are a helpful tutor for college students in the Philippines. Answer the question clearly and "
                 "accurately in English. Use at most 250 words. Use short paragraphs or a short list when it helps.")

QUESTION_PROMPT = """Write ONE multiple-choice question that checks whether a reader understood the MAIN POINT of the
AI answer below.

Rules:
- The question must be answerable from the AI answer alone, without outside knowledge.
- Exactly 4 options, exactly one correct; wrong options must be plausible but clearly wrong according to the answer.
- Keep the question under 30 words and each option under 20 words. No "all of the above" or "none of the above".
- level: 1 = recall a fact stated in the answer, 2 = understand or explain an idea, 3 = apply or infer from the answer.
- answer_index is the 0-based position of the correct option.

USER'S PROMPT:
{prompt}

AI ANSWER:
{answer}"""

CHECK_PROMPT = """Answer the multiple-choice question using ONLY the text below. Return the 0-based index of the best option.

TEXT:
{answer}

QUESTION: {question}
OPTIONS:
{options}"""


DEFAULT_MODEL = "gemini-3.8-flash"


class QuestionOut(BaseModel):
    question: str
    options: list[str]
    answer_index: int
    level: int


class CheckOut(BaseModel):
    answer_index: int


class GeminiError(RuntimeError):
    pass


class _RateLimiter:
    def __init__(self, rpm: int):
        self.interval = 60.0 / max(rpm, 1)
        self.lock = threading.Lock()
        self.next_time = 0.0

    def wait(self):
        with self.lock:
            now = time.monotonic()
            if now < self.next_time:
                time.sleep(self.next_time - now)
            self.next_time = max(now, self.next_time) + self.interval


class GeminiClient:
    fake = False

    def __init__(self, api_key: str, model: str | None = None, rpm: int | None = None):
        from google import genai
        self.genai = genai
        self.client = genai.Client(api_key=api_key)
        self.model = (model or os.environ.get("GEMINI_MODEL", "") or DEFAULT_MODEL).strip()
        self.use_thinking_config = True
        self.limiter = _RateLimiter(rpm or int(os.environ.get("GEMINI_RPM", "8")))

    def _config(self, **kw):
        """Keep answers fast: little or no internal "thinking" (Gemini 2.5 uses a budget, Gemini 3+ a level)."""
        from google.genai import types
        if self.use_thinking_config:
            if self.model.startswith("gemini-2.5") and "flash" in self.model:
                kw.setdefault("thinking_config", types.ThinkingConfig(thinking_budget=0))
            elif not self.model.startswith(("gemini-2.", "gemini-1.")):
                kw.setdefault("thinking_config", types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW))
        return types.GenerateContentConfig(**kw)

    def _call(self, contents, config, retries=4):
        from google.genai import errors
        delay = 8.0
        for attempt in range(retries + 1):
            self.limiter.wait()
            try:
                return self.client.models.generate_content(model=self.model, contents=contents, config=config)
            except errors.APIError as e:
                code = getattr(e, "code", None)
                msg = str(getattr(e, "message", e))
                if code == 400 and "thinking" in msg.lower() and getattr(config, "thinking_config", None):
                    self.use_thinking_config = False            # this model rejects the setting: drop it
                    config.thinking_config = None
                    continue
                if code == 404:
                    raise GeminiError(f"the model '{self.model}' is not available ({msg}). Set GEMINI_MODEL to a "
                                      f"current model, for example {DEFAULT_MODEL}") from e
                if code in (429, 500, 503) and attempt < retries:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise GeminiError(f"Gemini error {code}: {getattr(e, 'message', e)}") from e
        raise GeminiError("Gemini did not respond")

    def answer(self, prompt: str) -> str:
        r = self._call(prompt, self._config(system_instruction=ANSWER_SYSTEM, temperature=0.7))
        text = (r.text or "").strip()
        if not text:
            raise GeminiError("Gemini returned an empty answer")
        return text

    def question(self, prompt: str, answer: str) -> str:
        r = self._call(QUESTION_PROMPT.format(prompt=prompt, answer=answer),
                       self._config(response_mime_type="application/json", response_schema=QuestionOut,
                                    temperature=0.9))
        return r.text or ""

    def check(self, answer: str, question: str, options: list[str]) -> str:
        opts = "\n".join(f"{i}. {o}" for i, o in enumerate(options))
        r = self._call(CHECK_PROMPT.format(answer=answer, question=question, options=opts),
                       self._config(response_mime_type="application/json", response_schema=CheckOut,
                                    temperature=0.0))
        return r.text or ""


class FakeGemini:
    """Offline stand-in used for tests and for running the site before a key is set."""
    fake = True
    model = "fake-offline"

    def answer(self, prompt: str) -> str:
        topic = re.sub(r"[?.!]+$", "", prompt.strip()) or "your question"
        return (f"Here is a short explanation of {topic.lower()}. The main idea is that it describes how a system "
                "takes inputs, applies a clear set of rules, and produces a result. For example, a simple recipe "
                "turns ingredients into a dish by following steps in order.\n\n"
                "- The inputs are what you start with.\n- The rules decide what happens.\n- The output is the result.\n\n"
                "(Offline demo answer: set GEMINI_API_KEY to get real answers from Google Gemini.)")

    def question(self, prompt: str, answer: str) -> str:
        if "broken-json" in prompt:
            return "{not json"
        return json.dumps({"question": "According to the answer, what decides what happens to the inputs?",
                           "options": ["The rules", "The output", "The example recipe", "The number of steps"],
                           "answer_index": 0, "level": 1})

    def check(self, answer: str, question: str, options: list[str]) -> str:
        return json.dumps({"answer_index": 1 if "inconsistent" in answer else 0})


_client = None


def get_client():
    """Real Gemini if GEMINI_API_KEY is set, otherwise the offline FakeGemini."""
    global _client
    if _client is None:
        key = os.environ.get("GEMINI_API_KEY", "").strip()
        _client = GeminiClient(key) if key else FakeGemini()
    return _client


def set_client(client):
    global _client
    _client = client

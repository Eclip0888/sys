"""Google Gemini calls: the AI answer, the generated question and the self-consistency check.

GEMINI_API_KEY   your key from Google AI Studio (never commit it)
GEMINI_MODEL     model name, default "gemini-3.8-flash" (see ai.google.dev/gemini-api/docs/models)
GEMINI_RPM       maximum requests per minute (default 8) so the free tier is not exceeded
GEMINI_FALLBACK_MODELS  comma-separated models to try when the main model is busy (503) or out of quota (429),
                 default "gemini-3.7-flash,gemini-3.5-flash-lite"

GROQ_API_KEY     optional backup provider (console.groq.com): used when every Gemini model is busy or out of quota
GROQ_MODELS      comma-separated Groq models to try, default "openai/gpt-oss-120b,openai/gpt-oss-20b"

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
- Keep the question under 30 words and each option under 20 words. No "all of the above" or "none of the above",
  and do not refer to other options by letter or position (the options are shown in random order).
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
DEFAULT_FALLBACKS = "gemini-3.7-flash,gemini-3.5-flash-lite"


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
        fb = os.environ.get("GEMINI_FALLBACK_MODELS", DEFAULT_FALLBACKS)
        self.models = [self.model] + [m.strip() for m in fb.split(",") if m.strip() and m.strip() != self.model]
        self.no_thinking: set[str] = set()      # models that rejected the thinking setting
        self.last_model = self.model
        self.limiter = _RateLimiter(rpm or int(os.environ.get("GEMINI_RPM", "8")))

    @property
    def use_thinking_config(self):            # kept for compatibility with older tests
        return self.model not in self.no_thinking

    @use_thinking_config.setter
    def use_thinking_config(self, value):
        (self.no_thinking.discard if value else self.no_thinking.add)(self.model)

    def _config(self, model: str | None = None, **kw):
        """Keep answers fast: little or no internal "thinking" (Gemini 2.5 uses a budget, Gemini 3+ a level)."""
        from google.genai import types
        model = model or self.model
        if model not in self.no_thinking:
            if model.startswith("gemini-2.5") and "flash" in model:
                kw.setdefault("thinking_config", types.ThinkingConfig(thinking_budget=0))
            elif not model.startswith(("gemini-2.", "gemini-1.")):
                kw.setdefault("thinking_config", types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW))
        return types.GenerateContentConfig(**kw)

    def _call(self, contents, tries_per_model=2, **cfg):
        """Try the main model, then the fallback models when one is busy (503/500), out of quota (429) or gone (404)."""
        from google.genai import errors
        last = None
        for model in self.models:
            delay = 2.0
            attempt = 0
            while attempt < tries_per_model:
                self.limiter.wait()
                try:
                    r = self.client.models.generate_content(model=model, contents=contents,
                                                            config=self._config(model, **cfg))
                    self.last_model = model
                    return r
                except errors.APIError as e:
                    code = getattr(e, "code", None)
                    msg = str(getattr(e, "message", e))
                    last = f"{model}: error {code}: {msg}"
                    if code == 400 and "thinking" in msg.lower() and model not in self.no_thinking:
                        self.no_thinking.add(model)              # this model rejects the setting: drop it
                        continue
                    if code in (500, 503):
                        attempt += 1
                        if attempt < tries_per_model:
                            time.sleep(delay)
                            delay *= 2
                        continue
                    if code in (404, 429):
                        break                                     # next model
                    raise GeminiError(f"Gemini error {code}: {msg}") from e
        raise GeminiError(f"all models are busy or unavailable right now (last: {last}). Try again in a few minutes, "
                          f"or set GEMINI_MODEL to another current model")

    def answer(self, prompt: str) -> str:
        r = self._call(prompt, system_instruction=ANSWER_SYSTEM, temperature=0.7)
        text = (r.text or "").strip()
        if not text:
            raise GeminiError("Gemini returned an empty answer")
        return text

    def question(self, prompt: str, answer: str) -> str:
        r = self._call(QUESTION_PROMPT.format(prompt=prompt, answer=answer),
                       response_mime_type="application/json", response_schema=QuestionOut, temperature=0.9)
        return r.text or ""

    def check(self, answer: str, question: str, options: list[str]) -> str:
        opts = "\n".join(f"{i}. {o}" for i, o in enumerate(options))
        r = self._call(CHECK_PROMPT.format(answer=answer, question=question, options=opts),
                       response_mime_type="application/json", response_schema=CheckOut, temperature=0.0)
        return r.text or ""


GROQ_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODELS = "openai/gpt-oss-120b,openai/gpt-oss-20b"
JSON_QUESTION = ("\n\nRespond with ONLY a JSON object with these keys: question (string), options (array of exactly 4 "
                 "strings), answer_index (integer 0-3), level (integer 1-3).")
JSON_CHECK = '\n\nRespond with ONLY a JSON object of the form {"answer_index": <integer 0-3>}.'


class OpenAICompatClient:
    """Backup provider with an OpenAI-compatible chat API (Groq by default; works with OpenRouter too)."""
    fake = False

    def __init__(self, api_key: str, models: list[str], base_url: str = GROQ_URL, name: str = "groq", http=None):
        import httpx
        self.name, self.models, self.base_url = name, models, base_url.rstrip("/")
        self.http = http or httpx.Client(timeout=90.0, headers={"Authorization": f"Bearer {api_key}"})
        self.model = models[0]
        self.last_model = f"{name}/{self.model}"
        self.limiter = _RateLimiter(int(os.environ.get("GROQ_RPM", "20")))

    def _chat(self, messages, json_mode=False, temperature=0.7):
        last = None
        for model in self.models:
            body = {"model": model, "messages": messages, "temperature": temperature}
            if json_mode:
                body["response_format"] = {"type": "json_object"}
            for attempt in range(2):
                self.limiter.wait()
                try:
                    r = self.http.post(f"{self.base_url}/chat/completions", json=body)
                except Exception as e:                    # network error: try again, then the next model
                    last = f"{model}: {e}"
                    continue
                if r.status_code == 200:
                    text = (r.json()["choices"][0]["message"].get("content") or "").strip()
                    if text:
                        self.last_model = f"{self.name}/{model}"
                        return text
                    last = f"{model}: empty answer"
                    break
                last = f"{model}: error {r.status_code}: {r.text[:200]}"
                if r.status_code in (500, 502, 503) and attempt == 0:
                    time.sleep(2)
                    continue
                break                                     # 400/404 (model gone), 429 (quota): next model
        raise GeminiError(f"backup provider {self.name} could not answer (last: {last})")

    def answer(self, prompt: str) -> str:
        return self._chat([{"role": "system", "content": ANSWER_SYSTEM}, {"role": "user", "content": prompt}])

    def question(self, prompt: str, answer: str) -> str:
        content = QUESTION_PROMPT.format(prompt=prompt, answer=answer) + JSON_QUESTION
        return self._chat([{"role": "user", "content": content}], json_mode=True, temperature=0.9)

    def check(self, answer: str, question: str, options: list[str]) -> str:
        opts = "\n".join(f"{i}. {o}" for i, o in enumerate(options))
        content = CHECK_PROMPT.format(answer=answer, question=question, options=opts) + JSON_CHECK
        return self._chat([{"role": "user", "content": content}], json_mode=True, temperature=0.0)


class ChainClient:
    """Gemini first; when every Gemini model fails, the backup provider answers."""
    fake = False

    def __init__(self, providers):
        self.providers = providers
        self.model = providers[0].model + "".join(f" → {getattr(p, 'name', 'backup')} backup" for p in providers[1:])
        self.last_model = providers[0].model

    def _try(self, method, *args):
        errors = []
        for p in self.providers:
            try:
                out = getattr(p, method)(*args)
                self.last_model = getattr(p, "last_model", p.model)
                return out
            except GeminiError as e:
                errors.append(str(e))
        raise GeminiError(" | ".join(errors))

    def answer(self, prompt):
        return self._try("answer", prompt)

    def question(self, prompt, answer):
        return self._try("question", prompt, answer)

    def check(self, answer, question, options):
        return self._try("check", answer, question, options)


class FakeGemini:
    """Offline stand-in used for tests and for running the site before a key is set."""
    fake = True
    model = "fake-offline"
    last_model = "fake-offline"

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
        providers = []
        key = os.environ.get("GEMINI_API_KEY", "").strip()
        if key:
            providers.append(GeminiClient(key))
        gkey = os.environ.get("GROQ_API_KEY", "").strip()
        if gkey:
            models = [m.strip() for m in os.environ.get("GROQ_MODELS", DEFAULT_GROQ_MODELS).split(",") if m.strip()]
            providers.append(OpenAICompatClient(gkey, models))
        if not providers:
            _client = FakeGemini()
        elif len(providers) == 1:
            _client = providers[0]
        else:
            _client = ChainClient(providers)
    return _client


def set_client(client):
    global _client
    _client = client

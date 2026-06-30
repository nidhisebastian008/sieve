"""LLM-as-judge scorer via Ollama, Anthropic, or OpenAI.

Based on: "Judging LLM-as-a-Judge with MT-Bench" (Zheng et al., 2023).
GPT-4 class models achieve >80% agreement with human annotators.
"""

import httpx

from sieve.models import Interaction
from sieve.score.base import BaseScorer

_RUBRIC = """You are a strict evaluator of AI assistant responses.

Rate this response on a scale of 0 to 5:
5 — Perfect: accurate, complete, well-structured, directly answers the question
4 — Good: minor gaps or verbosity but overall high quality
3 — Acceptable: answers the core question but has notable issues
2 — Poor: partially addresses the question or contains errors
1 — Bad: mostly unhelpful or significantly wrong
0 — Unacceptable: refusal without reason, harmful, or completely wrong

USER MESSAGE:
{user}

ASSISTANT RESPONSE:
{assistant}

Reply with ONLY a single integer 0-5. No explanation."""


def _extract_score(text: str) -> float | None:
    for ch in text.strip():
        if ch.isdigit():
            v = int(ch)
            if 0 <= v <= 5:
                return v / 5.0
    return None


class OllamaJudgeScorer(BaseScorer):
    """Score via a locally running Ollama model. No API key needed."""

    def __init__(
        self,
        model: str = "llama3.2",
        host: str = "http://localhost:11434",
        timeout: int = 60,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    def _judge(self, user: str, assistant: str) -> float | None:
        prompt = _RUBRIC.format(user=user, assistant=assistant)
        try:
            resp = httpx.post(
                f"{self.host}/api/chat",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": 0, "num_predict": 4},
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return _extract_score(resp.json()["message"]["content"])
        except Exception:
            return None

    def score(self, interaction: Interaction) -> float:
        msgs = interaction.messages or []
        users = [m for m in msgs if m.get("role") == "user"]
        assts = [m for m in msgs if m.get("role") == "assistant"]
        if not users or not assts:
            return 0.0
        user = str(users[-1].get("content", ""))
        asst = str(assts[-1].get("content", ""))
        result = self._judge(user, asst)
        return result if result is not None else 0.0


class AnthropicJudgeScorer(BaseScorer):
    """Score via Claude API. Requires ANTHROPIC_API_KEY env var."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        try:
            import anthropic
            import os
            self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            self.model = model
        except ImportError:
            raise ImportError("pip install 'trainsieve[anthropic]'")
        except KeyError:
            raise EnvironmentError("ANTHROPIC_API_KEY not set")

    def score(self, interaction: Interaction) -> float:
        msgs = interaction.messages or []
        users = [m for m in msgs if m.get("role") == "user"]
        assts = [m for m in msgs if m.get("role") == "assistant"]
        if not users or not assts:
            return 0.0
        user = str(users[-1].get("content", ""))
        asst = str(assts[-1].get("content", ""))
        prompt = _RUBRIC.format(user=user, assistant=asst)
        try:
            msg = self._client.messages.create(
                model=self.model,
                max_tokens=4,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            return _extract_score(msg.content[0].text) or 0.0
        except Exception:
            return 0.0


class OpenAIJudgeScorer(BaseScorer):
    """Score via OpenAI API. Requires OPENAI_API_KEY env var."""

    def __init__(self, model: str = "gpt-4o-mini"):
        try:
            import openai
            import os
            self._client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            self.model = model
        except ImportError:
            raise ImportError("pip install 'trainsieve[openai]'")
        except KeyError:
            raise EnvironmentError("OPENAI_API_KEY not set")

    def score(self, interaction: Interaction) -> float:
        msgs = interaction.messages or []
        users = [m for m in msgs if m.get("role") == "user"]
        assts = [m for m in msgs if m.get("role") == "assistant"]
        if not users or not assts:
            return 0.0
        user = str(users[-1].get("content", ""))
        asst = str(assts[-1].get("content", ""))
        prompt = _RUBRIC.format(user=user, assistant=asst)
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                temperature=0,
                max_tokens=4,
                messages=[{"role": "user", "content": prompt}],
            )
            return _extract_score(resp.choices[0].message.content) or 0.0
        except Exception:
            return 0.0

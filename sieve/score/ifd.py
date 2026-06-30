"""Instruction Following Difficulty (IFD) scorer.

Based on: "From Quantity to Quality: Boosting LLM Performance with Self-Guided
Data Selection for Instruction Tuning" (Li et al., 2023 — Cherry LLM, arxiv 2308.12032).

IFD = loss(response | instruction) / loss(response | no instruction)
- LOW IFD  → instruction helps the model generate the response → high-value example
- HIGH IFD → response could be generated without instruction → low-value example

Requires a local Ollama model to compute token-level log probabilities.
"""

import math
import httpx

from sieve.models import Interaction
from sieve.score.base import BaseScorer


def _compute_logprob(host: str, model: str, prompt: str, timeout: int) -> float | None:
    """Return mean log probability of tokens in the prompt via Ollama."""
    try:
        resp = httpx.post(
            f"{host}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "logprobs": True,
                "options": {"temperature": 0},
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        # Ollama >=0.4 returns logprobs in response
        logprobs = data.get("logprobs", [])
        if logprobs and isinstance(logprobs, list):
            values = [lp for lp in logprobs if isinstance(lp, (int, float))]
            if values:
                return sum(values) / len(values)

        # Fallback: estimate via response length (crude but functional)
        # If logprobs unavailable, return None to signal fallback needed
        return None
    except Exception:
        return None


class IFDScorer(BaseScorer):
    """Instruction Following Difficulty scorer via Ollama.

    Scores 0–1 where HIGH score = low IFD = instruction is necessary = good training example.
    """

    def __init__(
        self,
        model: str = "llama3.2",
        host: str = "http://localhost:11434",
        timeout: int = 90,
        ifd_low_threshold: float = 0.3,
        ifd_high_threshold: float = 0.95,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        # examples with IFD below low_threshold are the most valuable
        self.ifd_low = ifd_low_threshold
        self.ifd_high = ifd_high_threshold

    def _format_prompt_with_instruction(self, user: str, assistant: str) -> str:
        return f"### Instruction:\n{user}\n\n### Response:\n{assistant}"

    def _format_prompt_without_instruction(self, assistant: str) -> str:
        return f"### Response:\n{assistant}"

    def score(self, interaction: Interaction) -> float:
        msgs = interaction.messages or []
        users = [m for m in msgs if m.get("role") == "user"]
        assts = [m for m in msgs if m.get("role") == "assistant"]
        if not users or not assts:
            return 0.0

        user = str(users[-1].get("content", ""))
        asst = str(assts[-1].get("content", ""))

        with_instruction = self._format_prompt_with_instruction(user, asst)
        without_instruction = self._format_prompt_without_instruction(asst)

        lp_with = _compute_logprob(self.host, self.model, with_instruction, self.timeout)
        lp_without = _compute_logprob(self.host, self.model, without_instruction, self.timeout)

        if lp_with is None or lp_without is None:
            # Ollama logprobs unavailable — fall back to length-based heuristic
            return self._fallback_score(user, asst)

        # Convert log probs to losses (negative log prob = loss)
        loss_with = -lp_with
        loss_without = max(-lp_without, 1e-8)
        ifd = loss_with / loss_without

        # Map IFD to 0–1 quality score: low IFD → high quality
        return self._ifd_to_quality(ifd)

    def _ifd_to_quality(self, ifd: float) -> float:
        """Map IFD ratio to 0–1 quality. Low IFD = high quality."""
        if ifd <= self.ifd_low:
            return 1.0
        if ifd >= self.ifd_high:
            return 0.0
        # linear interpolation in between
        return 1.0 - (ifd - self.ifd_low) / (self.ifd_high - self.ifd_low)

    def _fallback_score(self, user: str, assistant: str) -> float:
        """Approximate IFD when logprobs unavailable.

        Heuristic: if assistant response is long relative to user prompt,
        the response likely requires the instruction context → higher value.
        """
        u_len = len(user.split())
        a_len = len(assistant.split())
        if u_len == 0:
            return 0.0
        ratio = a_len / u_len
        # ratio > 3 → detailed response relative to short prompt → likely high value
        return min(1.0, ratio / 5.0)

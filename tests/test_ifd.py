"""Tests for IFD scorer."""

from unittest.mock import patch

from sieve.models import Interaction
from sieve.score.ifd import IFDScorer


def _make(user: str, assistant: str) -> Interaction:
    return Interaction(
        source="test",
        messages=[
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
    )


scorer = IFDScorer()


def test_ifd_fallback_detailed_response_scores_high():
    # long assistant response relative to short user prompt → high value
    i = _make("Why?", "This is because of several interconnected reasons. " * 10)
    with patch("sieve.score.ifd._compute_logprob", return_value=None):
        s = scorer.score(i)
    assert s > 0.5


def test_ifd_fallback_short_response_scores_low():
    i = _make("Explain quantum mechanics in great detail please", "Yes.")
    with patch("sieve.score.ifd._compute_logprob", return_value=None):
        s = scorer.score(i)
    assert s < 0.3


def test_ifd_with_logprobs_low_ifd_scores_high():
    # low IFD: loss_with << loss_without → instruction helps a lot → high quality
    i = _make("What is 2+2?", "4")
    call_count = 0

    def mock_logprob(host, model, prompt, timeout):
        nonlocal call_count
        call_count += 1
        # first call = with instruction (lower loss = higher logprob)
        # second call = without instruction (higher loss = lower logprob)
        return -0.3 if call_count == 1 else -1.5

    with patch("sieve.score.ifd._compute_logprob", side_effect=mock_logprob):
        s = scorer.score(i)
    assert s > 0.5


def test_ifd_with_logprobs_high_ifd_scores_low():
    # high IFD: loss_with ≈ loss_without → instruction doesn't help → low quality
    i = _make("Tell me something", "Hello world.")
    call_count = 0

    def mock_logprob(host, model, prompt, timeout):
        nonlocal call_count
        call_count += 1
        return -1.0  # same loss with and without instruction

    with patch("sieve.score.ifd._compute_logprob", side_effect=mock_logprob):
        s = scorer.score(i)
    assert s < 0.5


def test_ifd_empty_messages_returns_zero():
    i = Interaction(source="test", messages=[])
    assert scorer.score(i) == 0.0

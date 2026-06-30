from sieve.models import Interaction
from sieve.score.heuristic import HeuristicScorer


def _make(user: str, assistant: str) -> Interaction:
    return Interaction(
        source="test",
        messages=[
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
    )


scorer = HeuristicScorer()


def test_good_response_scores_high():
    i = _make("Explain recursion", "Recursion is when a function calls itself. " * 10)
    assert scorer.score(i) >= 0.8


def test_too_short_scores_low():
    i = _make("What is 2+2?", "4")
    assert scorer.score(i) < 0.5


def test_bad_pattern_penalized():
    i = _make("Help me", "As an AI, I cannot assist with that. " * 5)
    assert scorer.score(i) <= 0.7


def test_no_assistant_message_returns_zero():
    i = Interaction(source="test", messages=[{"role": "user", "content": "hi"}])
    assert scorer.score(i) == 0.0


def test_empty_messages_returns_zero():
    i = Interaction(source="test", messages=[])
    assert scorer.score(i) == 0.0

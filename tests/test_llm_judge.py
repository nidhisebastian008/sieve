"""Tests for LLM judge scorer — mocks Ollama/API calls."""

from unittest.mock import MagicMock, patch

from sieve.models import Interaction
from sieve.score.llm_judge import OllamaJudgeScorer, _extract_score


def _make(user: str, assistant: str) -> Interaction:
    return Interaction(
        source="test",
        messages=[
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
    )


def test_extract_score_valid():
    assert _extract_score("4") == 4 / 5.0
    assert _extract_score("  3\n") == 3 / 5.0
    assert _extract_score("Score: 5") == 1.0
    assert _extract_score("0") == 0.0


def test_extract_score_invalid():
    assert _extract_score("no digit here") is None
    assert _extract_score("") is None


def test_ollama_judge_scores_correctly():
    scorer = OllamaJudgeScorer()
    mock_response = MagicMock()
    mock_response.json.return_value = {"message": {"content": "4"}}
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.post", return_value=mock_response):
        score = scorer.score(_make("What is Python?", "Python is a high-level programming language."))

    assert score == 4 / 5.0


def test_ollama_judge_returns_zero_on_connection_error():
    scorer = OllamaJudgeScorer()
    with patch("httpx.post", side_effect=Exception("connection refused")):
        score = scorer.score(_make("hi", "hello"))
    assert score == 0.0


def test_ollama_judge_zero_on_empty_messages():
    scorer = OllamaJudgeScorer()
    i = Interaction(source="test", messages=[])
    assert scorer.score(i) == 0.0


def test_ollama_judge_zero_on_missing_assistant():
    scorer = OllamaJudgeScorer()
    i = Interaction(source="test", messages=[{"role": "user", "content": "hi"}])
    assert scorer.score(i) == 0.0

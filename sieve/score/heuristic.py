from sieve.models import Interaction
from sieve.score.base import BaseScorer

_BAD_PATTERNS = [
    "i cannot",
    "i'm unable",
    "i don't have access",
    "as an ai",
    "i apologize, but",
    "i'm sorry, but i can't",
]


class HeuristicScorer(BaseScorer):
    def __init__(self, min_response_len: int = 50, max_response_len: int = 4000):
        self.min_response_len = min_response_len
        self.max_response_len = max_response_len

    def score(self, interaction: Interaction) -> float:
        messages = interaction.messages or []
        assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
        user_msgs = [m for m in messages if m.get("role") == "user"]

        if not assistant_msgs or not user_msgs:
            return 0.0

        scores = []
        for msg in assistant_msgs:
            content = msg.get("content", "")
            if not isinstance(content, str):
                content = str(content)

            length = len(content)
            if length < self.min_response_len:
                length_score = length / self.min_response_len
            elif length > self.max_response_len:
                overage = (length - self.max_response_len) / self.max_response_len
                length_score = max(0.5, 1.0 - overage)
            else:
                length_score = 1.0

            lowered = content.lower()
            penalty = sum(0.15 for p in _BAD_PATTERNS if p in lowered)
            scores.append(max(0.0, length_score - penalty))

        return sum(scores) / len(scores)

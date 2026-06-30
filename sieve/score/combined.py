"""Combined scorer: weighted average of heuristic + LLM judge + IFD.

Weights from research:
- LLM judge is most reliable (>80% human agreement) → highest weight
- IFD captures instruction necessity → medium weight
- Heuristic is fast but crude → lowest weight
"""

from sieve.models import Interaction
from sieve.score.base import BaseScorer
from sieve.score.heuristic import HeuristicScorer


class CombinedScorer(BaseScorer):
    def __init__(
        self,
        judge_scorer: BaseScorer | None = None,
        ifd_scorer: BaseScorer | None = None,
        weights: dict[str, float] | None = None,
    ):
        self.heuristic = HeuristicScorer()
        self.judge = judge_scorer
        self.ifd = ifd_scorer
        self.weights = weights or {
            "heuristic": 0.2,
            "judge": 0.6,
            "ifd": 0.2,
        }

    def score(self, interaction: Interaction) -> float:
        scores: dict[str, float] = {}
        total_weight = 0.0

        h = self.heuristic.score(interaction)
        scores["heuristic"] = h
        total_weight += self.weights["heuristic"]

        if self.judge is not None:
            j = self.judge.score(interaction)
            scores["judge"] = j
            total_weight += self.weights["judge"]

        if self.ifd is not None:
            d = self.ifd.score(interaction)
            scores["ifd"] = d
            total_weight += self.weights["ifd"]

        if total_weight == 0:
            return 0.0

        weighted = sum(scores[k] * self.weights[k] for k in scores)
        return weighted / total_weight

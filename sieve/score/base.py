from abc import ABC, abstractmethod

from sieve.models import Interaction


class BaseScorer(ABC):
    @abstractmethod
    def score(self, interaction: Interaction) -> float:
        """Return quality score 0.0–1.0."""
        pass

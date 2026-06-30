from abc import ABC, abstractmethod
from typing import Iterator

from sieve.models import Interaction


class BaseIngester(ABC):
    @abstractmethod
    def ingest(self) -> Iterator[Interaction]:
        pass

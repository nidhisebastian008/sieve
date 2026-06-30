from pathlib import Path
from typing import Iterator

import jsonlines

from sieve.ingest.base import BaseIngester
from sieve.models import Interaction


class JSONLIngester(BaseIngester):
    def __init__(self, path: Path):
        self.path = path

    def ingest(self) -> Iterator[Interaction]:
        with jsonlines.open(self.path) as reader:
            for obj in reader:
                messages = obj.get("messages") or obj.get("conversations") or []

                if not messages and "prompt" in obj and "response" in obj:
                    messages = [
                        {"role": "user", "content": obj["prompt"]},
                        {"role": "assistant", "content": obj["response"]},
                    ]

                if not messages:
                    continue

                skip_keys = {"messages", "conversations", "prompt", "response"}
                meta = {k: v for k, v in obj.items() if k not in skip_keys}

                yield Interaction(
                    source="jsonl",
                    messages=messages,
                    metadata_={"source_file": str(self.path), **meta},
                )

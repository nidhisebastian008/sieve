import json
import tempfile
from pathlib import Path

from sieve.ingest.jsonl import JSONLIngester


def _write_jsonl(path: Path, rows: list) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_ingest_messages_format():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
        path = Path(f.name)

    _write_jsonl(path, [
        {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]},
    ])
    results = list(JSONLIngester(path).ingest())
    assert len(results) == 1
    assert results[0].messages[0]["role"] == "user"
    assert results[0].source == "jsonl"


def test_ingest_prompt_response_format():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
        path = Path(f.name)

    _write_jsonl(path, [
        {"prompt": "What is Python?", "response": "A programming language."},
    ])
    results = list(JSONLIngester(path).ingest())
    assert len(results) == 1
    assert results[0].messages[0]["content"] == "What is Python?"
    assert results[0].messages[1]["role"] == "assistant"


def test_ingest_skips_empty_messages():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
        path = Path(f.name)

    _write_jsonl(path, [
        {"no_messages": True},
        {"messages": [{"role": "user", "content": "valid"}, {"role": "assistant", "content": "ok"}]},
    ])
    results = list(JSONLIngester(path).ingest())
    assert len(results) == 1

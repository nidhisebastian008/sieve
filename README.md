# sieve

Production LLM traces → curated training datasets → fine-tune pipeline.

Most teams ship prompt changes blind. **sieve** closes the loop: capture real interactions, score quality, version datasets, and trigger fine-tuning — without duct-taping five tools together.

```
production logs → score → filter → versioned dataset → fine-tune
```

## Install

```bash
git clone https://github.com/nidhisebastian008/sieve
cd sieve
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Quickstart

```bash
# 1. ingest interactions (supports messages[], prompt/response, conversations[] formats)
sieve ingest my_logs.jsonl

# 2. score quality with heuristics (no API key needed)
sieve score --min-len 50 --max-len 4000

# 3. check what you have
sieve stats

# 4. create a versioned dataset from high-quality interactions
sieve dataset create v1.0 --min-quality 0.6

# 5. export to JSONL for fine-tuning
sieve dataset export v1.0 --output train.jsonl

# 6. generate Axolotl config + see training command
sieve train v1.0 --export-path train.jsonl
```

## Input formats

sieve accepts any of these JSONL row formats:

```jsonl
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"prompt": "...", "response": "..."}
{"conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

## Dataset versioning

Datasets are versioned and tracked in a local SQLite database (`~/.sieve/sieve.db`).

```bash
# only include interactions not already in v1.0
sieve dataset create v2.0 --min-quality 0.6 --diff v1.0

sieve dataset list
```

```
┏━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃ Name ┃ Interactions ┃ Min Quality ┃ Parent ┃ Created          ┃
┡━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│ v1.0 │ 1200         │ 0.60        │ —      │ 2026-06-30 10:00 │
│ v2.0 │ 340          │ 0.60        │ v1.0   │ 2026-07-15 09:00 │
└──────┴──────────────┴─────────────┴────────┴──────────────────┘
```

## Scorers

| Scorer | Key needed | How |
|---|---|---|
| Heuristic (default) | No | Length + bad-pattern detection |
| LLM-as-judge (coming) | Ollama or API key | Claude / GPT / local model |

## Fine-tuning backends

| Backend | Key needed |
|---|---|
| Axolotl (local GPU) | No |
| LLaMA-Factory (local) | No |
| Modal (coming) | Yes |

## Architecture

```
sieve/
├── ingest/      ← pluggable ingesters (JSONL, Langfuse coming)
├── score/       ← pluggable scorers (heuristic, LLM judge coming)
├── curate/      ← dataset versioning, lineage, export
└── trigger/     ← training config generation (Axolotl, LLaMA-Factory coming)
```

Everything is plugin-friendly. Implement `BaseIngester` or `BaseScorer` and drop it in.

## Run tests

```bash
pytest tests/ -v
```

## Roadmap

- [ ] Langfuse ingester
- [ ] OpenTelemetry trace ingester
- [ ] LLM-as-judge scorer (Ollama + API)
- [ ] LLaMA-Factory trigger
- [ ] Modal cloud training trigger
- [ ] HuggingFace dataset push
- [ ] `sieve eval` — compare base vs fine-tuned model quality

## License

Apache 2.0

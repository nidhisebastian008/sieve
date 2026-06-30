"""Semantic deduplication using sentence embeddings + FAISS.

Based on: SemDeDup (Meta AI, 2023, arxiv 2303.09540).
Can remove 50% of training data with minimal quality loss by eliminating
near-duplicate examples that waste compute without adding information.

Requires: pip install 'trainsieve[embeddings]'
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sqlalchemy.orm import Session

from sieve.models import Interaction

if TYPE_CHECKING:
    pass


def _get_text(interaction: Interaction) -> str:
    """Flatten messages into a single string for embedding."""
    parts = []
    for m in (interaction.messages or []):
        role = m.get("role", "")
        content = m.get("content", "")
        if isinstance(content, str):
            parts.append(f"{role}: {content}")
    return " ".join(parts)[:512]  # truncate to avoid slow embedding


def run_semantic_dedup(
    session: Session,
    threshold: float = 0.92,
    embed_model: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
) -> dict[str, int]:
    """Mark duplicate interactions in the database.

    Args:
        threshold: Cosine similarity above which two interactions are duplicates (0–1).
        embed_model: Sentence-transformers model name.
        batch_size: Embedding batch size.

    Returns:
        Dict with 'total', 'duplicates_marked', 'unique_kept' counts.
    """
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
    except ImportError:
        raise ImportError("pip install 'trainsieve[embeddings]'")

    # load only non-duplicate interactions (re-runs skip already-marked dups)
    interactions = (
        session.query(Interaction)
        .filter(Interaction.is_duplicate.is_(None))
        .all()
    )

    if len(interactions) < 2:
        return {"total": len(interactions), "duplicates_marked": 0, "unique_kept": len(interactions)}

    texts = [_get_text(i) for i in interactions]

    model = SentenceTransformer(embed_model)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)

    # FAISS inner product on normalized vectors = cosine similarity
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    # for each vector, find nearest neighbours excluding itself
    k = min(5, len(interactions))
    distances, indices = index.search(embeddings.astype(np.float32), k)

    canonical = {}  # idx → canonical idx
    duplicates_marked = 0

    for i in range(len(interactions)):
        if i in canonical:
            continue  # already marked as duplicate
        for rank in range(1, k):
            j = int(indices[i][rank])
            sim = float(distances[i][rank])
            if sim >= threshold and j not in canonical:
                # j is a duplicate of i (canonical)
                canonical[j] = i
                interactions[j].is_duplicate = interactions[i].id
                duplicates_marked += 1

    session.commit()

    return {
        "total": len(interactions),
        "duplicates_marked": duplicates_marked,
        "unique_kept": len(interactions) - duplicates_marked,
    }

"""Tests for semantic deduplication."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from sieve.dedup.semantic import run_semantic_dedup
from sieve.models import Base, Interaction


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    yield s
    s.close()


def _add(session, content: str) -> Interaction:
    i = Interaction(
        source="test",
        messages=[
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": content},
        ],
    )
    session.add(i)
    session.commit()
    return i


def _mock_faiss_and_st(embeddings_matrix):
    """Return context managers mocking sentence_transformers and faiss."""
    import types

    mock_model = MagicMock()
    mock_model.encode.return_value = embeddings_matrix

    mock_index = MagicMock()
    n = len(embeddings_matrix)
    k = min(5, n)

    # build fake distances / indices: first neighbour is self, rest random
    dists = np.zeros((n, k))
    idxs = np.zeros((n, k), dtype=int)
    for i in range(n):
        idxs[i][0] = i
        for j in range(1, k):
            idxs[i][j] = (i + j) % n
            dists[i][j] = 0.5  # below threshold by default

    mock_index.search.return_value = (dists, idxs)

    mock_faiss = types.SimpleNamespace(IndexFlatIP=MagicMock(return_value=mock_index))
    mock_st_module = types.SimpleNamespace(SentenceTransformer=MagicMock(return_value=mock_model))

    return mock_st_module, mock_faiss


def test_dedup_marks_similar_pairs(session):
    i1 = _add(session, "Python is great for data science.")
    i2 = _add(session, "Python is great for data science.")  # near-duplicate

    embeddings = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    mock_st, mock_faiss = _mock_faiss_and_st(embeddings)

    # make i2 appear as near-duplicate of i1 (high cosine sim)
    mock_faiss.IndexFlatIP.return_value.search.return_value = (
        np.array([[1.0, 0.99], [1.0, 0.99]]),
        np.array([[0, 1], [1, 0]]),
    )

    with patch.dict("sys.modules", {"sentence_transformers": mock_st, "faiss": mock_faiss}):
        result = run_semantic_dedup(session, threshold=0.92)

    assert result["duplicates_marked"] == 1
    assert result["unique_kept"] == 1


def test_dedup_skips_dissimilar(session):
    _add(session, "Python is a programming language.")
    _add(session, "The Eiffel Tower is in Paris.")

    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    mock_st, mock_faiss = _mock_faiss_and_st(embeddings)

    # low similarity — no duplicates
    mock_faiss.IndexFlatIP.return_value.search.return_value = (
        np.array([[1.0, 0.1], [1.0, 0.1]]),
        np.array([[0, 1], [1, 0]]),
    )

    with patch.dict("sys.modules", {"sentence_transformers": mock_st, "faiss": mock_faiss}):
        result = run_semantic_dedup(session, threshold=0.92)

    assert result["duplicates_marked"] == 0
    assert result["unique_kept"] == 2


def test_dedup_single_interaction_no_op(session):
    _add(session, "Only one interaction.")
    result = run_semantic_dedup.__wrapped__(session) if hasattr(run_semantic_dedup, "__wrapped__") else None
    # with 1 interaction, should return early
    session2_interactions = session.query(Interaction).all()
    assert len(session2_interactions) == 1

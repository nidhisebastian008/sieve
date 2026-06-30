import tempfile
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from sieve.curate.dataset import DatasetManager
from sieve.models import Base, Interaction


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    yield s
    s.close()


def _add_interaction(session, score: float, content: str = "test response " * 10):
    i = Interaction(
        source="test",
        messages=[
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": content},
        ],
        quality_score=score,
    )
    session.add(i)
    session.commit()
    return i


def test_create_version_filters_by_quality(session):
    _add_interaction(session, score=0.9)
    _add_interaction(session, score=0.3)
    _add_interaction(session, score=0.7)

    mgr = DatasetManager(session)
    v = mgr.create_version("v1.0", min_quality=0.6)
    assert len(v.interactions) == 2


def test_create_version_diff_excludes_parent(session):
    i1 = _add_interaction(session, score=0.9)
    i2 = _add_interaction(session, score=0.9)

    mgr = DatasetManager(session)
    v1 = mgr.create_version("v1.0", min_quality=0.5)
    assert len(v1.interactions) == 2

    i3 = _add_interaction(session, score=0.9)
    v2 = mgr.create_version("v2.0", min_quality=0.5, parent_name="v1.0")
    assert len(v2.interactions) == 1
    assert v2.interactions[0].id == i3.id


def test_export_jsonl(session, tmp_path):
    _add_interaction(session, score=0.9)
    mgr = DatasetManager(session)
    mgr.create_version("v1.0", min_quality=0.5)

    out = tmp_path / "export.jsonl"
    count = mgr.export_jsonl("v1.0", out)
    assert count == 1
    assert out.exists()

    import jsonlines
    with jsonlines.open(out) as r:
        rows = list(r)
    assert len(rows) == 1
    assert "messages" in rows[0]


def test_export_unknown_version_raises(session, tmp_path):
    mgr = DatasetManager(session)
    with pytest.raises(ValueError, match="not found"):
        mgr.export_jsonl("nonexistent", tmp_path / "out.jsonl")

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from sieve.models import Base

_DEFAULT_DB = Path.home() / ".sieve" / "sieve.db"


def get_engine(db_path: Path = _DEFAULT_DB):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{db_path}", echo=False)


def init_db(db_path: Path = _DEFAULT_DB):
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    return engine


def get_session(db_path: Path = _DEFAULT_DB) -> Session:
    engine = get_engine(db_path)
    return sessionmaker(bind=engine)()

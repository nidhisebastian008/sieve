from datetime import datetime
from uuid import uuid4

from sqlalchemy import Column, DateTime, Float, ForeignKey, JSON, String, Table
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


interaction_dataset = Table(
    "interaction_dataset",
    Base.metadata,
    Column("interaction_id", String, ForeignKey("interactions.id")),
    Column("dataset_version_id", String, ForeignKey("dataset_versions.id")),
)


class Interaction(Base):
    __tablename__ = "interactions"

    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    source = Column(String, nullable=False)
    messages = Column(JSON, nullable=False)  # OpenAI chat format
    metadata_ = Column("metadata", JSON, default=dict)
    ingested_at = Column(DateTime, default=datetime.utcnow)
    quality_score = Column(Float, nullable=True)
    scored_at = Column(DateTime, nullable=True)
    scorer = Column(String, nullable=True)

    dataset_versions = relationship(
        "DatasetVersion", secondary=interaction_dataset, back_populates="interactions"
    )


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"

    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    name = Column(String, nullable=False, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    description = Column(String, nullable=True)
    parent_name = Column(String, nullable=True)
    min_quality_score = Column(Float, nullable=True)

    interactions = relationship(
        "Interaction", secondary=interaction_dataset, back_populates="dataset_versions"
    )


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    dataset_version_name = Column(String, ForeignKey("dataset_versions.name"))
    backend = Column(String, nullable=False)
    config = Column(JSON, default=dict)
    status = Column(String, default="triggered")
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    output_path = Column(String, nullable=True)
    notes = Column(String, nullable=True)

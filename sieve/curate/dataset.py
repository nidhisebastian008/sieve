from pathlib import Path
from typing import Optional

import jsonlines
from sqlalchemy.orm import Session

from sieve.models import DatasetVersion, Interaction


class DatasetManager:
    def __init__(self, session: Session):
        self.session = session

    def create_version(
        self,
        name: str,
        min_quality: float = 0.0,
        description: Optional[str] = None,
        parent_name: Optional[str] = None,
    ) -> DatasetVersion:
        query = self.session.query(Interaction)

        if min_quality > 0:
            query = query.filter(
                Interaction.quality_score >= min_quality,
                Interaction.quality_score.isnot(None),
            )

        if parent_name:
            parent = self.session.query(DatasetVersion).filter_by(name=parent_name).first()
            if parent:
                parent_ids = [i.id for i in parent.interactions]
                if parent_ids:
                    query = query.filter(~Interaction.id.in_(parent_ids))

        interactions = query.all()

        version = DatasetVersion(
            name=name,
            description=description,
            parent_name=parent_name,
            min_quality_score=min_quality,
        )
        version.interactions = interactions
        self.session.add(version)
        self.session.commit()
        return version

    def export_jsonl(self, version_name: str, output_path: Path) -> int:
        version = self.session.query(DatasetVersion).filter_by(name=version_name).first()
        if not version:
            raise ValueError(f"Version {version_name!r} not found")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with jsonlines.open(output_path, mode="w") as writer:
            for interaction in version.interactions:
                writer.write({"messages": interaction.messages})
                count += 1
        return count

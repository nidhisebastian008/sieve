from pathlib import Path
from typing import Optional

import yaml
from sqlalchemy.orm import Session

from sieve.models import TrainingRun


class AxolotlTrigger:
    def __init__(self, session: Session):
        self.session = session

    def generate_config(
        self,
        dataset_path: Path,
        base_model: str,
        output_dir: Path,
        config_path: Optional[Path] = None,
    ) -> Path:
        config = {
            "base_model": base_model,
            "datasets": [{"path": str(dataset_path.resolve()), "type": "sharegpt"}],
            "output_dir": str(output_dir.resolve()),
            "sequence_len": 2048,
            "micro_batch_size": 2,
            "num_epochs": 3,
            "learning_rate": 2e-4,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "lora_target_modules": ["q_proj", "v_proj"],
            "bf16": True,
            "load_in_4bit": True,
            "val_set_size": 0.05,
            "logging_steps": 10,
            "save_steps": 100,
        }

        out = config_path or dataset_path.parent / "axolotl_config.yml"
        out.write_text(yaml.dump(config, default_flow_style=False))
        return out

    def trigger(
        self,
        dataset_version_name: str,
        dataset_path: Path,
        base_model: str,
        output_dir: Path,
    ) -> TrainingRun:
        config_path = self.generate_config(dataset_path, base_model, output_dir)

        run = TrainingRun(
            dataset_version_name=dataset_version_name,
            backend="axolotl",
            config={"config_path": str(config_path), "base_model": base_model},
            status="triggered",
        )
        self.session.add(run)
        self.session.commit()
        return run

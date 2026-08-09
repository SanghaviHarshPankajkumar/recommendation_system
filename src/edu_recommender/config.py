from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    project_root = config_path.parent.parent
    for key, value in config["paths"].items():
        candidate = Path(value)
        config["paths"][key] = str(
            candidate if candidate.is_absolute() else (project_root / candidate).resolve()
        )
    return config, project_root


def write_json(payload: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


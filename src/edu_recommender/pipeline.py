from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import load_config, write_json
from .preprocess_ednet import EdNetPreprocessor
from .preprocess_oulad import OULADPreprocessor
from .profiling import profile_ednet, profile_oulad
from .splitting import TemporalSplitter


def _write_splits(events: pd.DataFrame, dataset: str, output_dir: Path) -> dict[str, Any]:
    dataset_dir = output_dir / "processed" / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for split, frame in events.groupby("split"):
        frame.to_csv(dataset_dir / f"{split}.csv.gz", index=False, compression="gzip")
    manifest = TemporalSplitter.manifest(events)
    write_json(manifest, dataset_dir / "manifest.json")
    return manifest


@dataclass(slots=True)
class Phase13Pipeline:
    """Coordinate profiling, class-based preprocessing, and temporal splitting."""

    config_path: str | Path

    def run(self) -> dict[str, Any]:
        return _run_phase_1_3(self.config_path)


def _run_phase_1_3(config_path: str | Path) -> dict[str, Any]:
    config, _ = load_config(config_path)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    paths = config["paths"]
    profile_config = config["profiling"]
    processing = config["preprocessing"]
    split = config["split"]
    output_dir = Path(paths["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(config, output_dir / "resolved_config.json")

    ednet_profile = profile_ednet(
        paths["ednet_zip"],
        paths["ednet_questions"],
        paths["ednet_lectures"],
        profile_config["ednet_sample_users"],
    )
    oulad_profile = profile_oulad(
        paths["oulad_raw"],
        profile_config["chunk_size"],
        profile_config["oulad_vle_max_rows"],
    )
    write_json(ednet_profile, output_dir / "profiles" / "ednet.json")
    write_json(oulad_profile, output_dir / "profiles" / "oulad.json")

    ednet_events, ednet_processing = EdNetPreprocessor(
        zip_path=paths["ednet_zip"],
        questions_path=paths["ednet_questions"],
        lectures_path=paths["ednet_lectures"],
        max_users=processing["ednet_max_users"],
        session_gap_minutes=processing["session_gap_minutes"],
    ).run()
    oulad_events, oulad_processing = OULADPreprocessor(
        raw_dir=paths["oulad_raw"],
        vle_max_rows=processing["oulad_vle_max_rows"],
        chunk_size=profile_config["chunk_size"],
    ).run()

    splitter = TemporalSplitter(
        train_ratio=split["train"],
        validation_ratio=split["validation"],
        test_ratio=split["test"],
        minimum_events_for_holdout=split["minimum_events_for_holdout"],
    )
    ednet_events = splitter.split(ednet_events)
    oulad_events = splitter.split(oulad_events)
    ednet_manifest = _write_splits(ednet_events, "ednet", output_dir)
    oulad_manifest = _write_splits(oulad_events, "oulad", output_dir)

    summary = {
        "run_name": config["run_name"],
        "ednet_processing": ednet_processing,
        "oulad_processing": oulad_processing,
        "ednet_manifest": ednet_manifest,
        "oulad_manifest": oulad_manifest,
    }
    write_json(summary, output_dir / "summary.json")
    return summary

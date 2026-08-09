from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True, slots=True)
class TemporalSplitter:
    """Assign per-student chronological train, validation, and test splits."""

    train_ratio: float
    validation_ratio: float
    test_ratio: float
    minimum_events_for_holdout: int = 5

    def __post_init__(self) -> None:
        if abs(self.train_ratio + self.validation_ratio + self.test_ratio - 1.0) > 1e-9:
            raise ValueError("Temporal split ratios must sum to 1")
        if self.minimum_events_for_holdout < 1:
            raise ValueError("minimum_events_for_holdout must be positive")

    def split(self, events: pd.DataFrame) -> pd.DataFrame:
        result = events.sort_values(["student_id", "timestamp"], kind="stable").copy()
        position = result.groupby("student_id").cumcount()
        length = result.groupby("student_id")["student_id"].transform("size")
        train_end = (length * self.train_ratio).astype(int).clip(lower=1)
        validation_end = (length * (self.train_ratio + self.validation_ratio)).astype(int)
        validation_end = validation_end.where(validation_end > train_end, train_end + 1)

        result["split"] = "test"
        result.loc[position < validation_end, "split"] = "validation"
        result.loc[position < train_end, "split"] = "train"
        result.loc[length < self.minimum_events_for_holdout, "split"] = "train"
        return result

    @staticmethod
    def manifest(events: pd.DataFrame) -> dict[str, Any]:
        payload: dict[str, Any] = {"total_events": len(events), "splits": {}}
        for name, frame in events.groupby("split"):
            payload["splits"][str(name)] = {
                "events": len(frame),
                "students": int(frame["student_id"].nunique()),
                "timestamp_min": int(frame["timestamp"].min()),
                "timestamp_max": int(frame["timestamp"].max()),
            }
        return payload

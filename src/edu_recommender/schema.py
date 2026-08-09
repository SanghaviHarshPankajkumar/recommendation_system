from __future__ import annotations

import pandas as pd


COMMON_EVENT_COLUMNS = [
    "dataset",
    "student_id",
    "timestamp",
    "relative_day",
    "item_id",
    "item_type",
    "concept_ids",
    "action_type",
    "correctness",
    "score",
    "elapsed_time_ms",
    "engagement",
    "source",
    "session_id",
    "module_id",
    "final_response",
    "is_banked",
    "outcome",
]


def ensure_common_schema(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in COMMON_EVENT_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA
    result = result[COMMON_EVENT_COLUMNS]
    result["student_id"] = result["student_id"].astype("string")
    result["item_id"] = result["item_id"].astype("string")
    result["dataset"] = result["dataset"].astype("string")
    return result


def validate_common_schema(frame: pd.DataFrame) -> None:
    missing = set(COMMON_EVENT_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing common event columns: {sorted(missing)}")
    if frame["student_id"].isna().any():
        raise ValueError("student_id contains missing values")
    if frame["timestamp"].isna().any():
        raise ValueError("timestamp contains missing values")
    if frame["item_id"].isna().any():
        raise ValueError("item_id contains missing values")

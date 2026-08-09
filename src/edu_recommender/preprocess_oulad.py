from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .schema import ensure_common_schema, validate_common_schema


def _presentation_origin(code: str) -> int:
    text = str(code)
    year = int(text[:4])
    term_offset = 31 if text.endswith("B") else 273
    return year * 1000 + term_offset


def _module_id(frame: pd.DataFrame) -> pd.Series:
    return frame["code_module"].astype(str) + ":" + frame["code_presentation"].astype(str)


@dataclass(slots=True)
class OULADPreprocessor:
    """Normalize OULAD tables into the shared event schema."""

    raw_dir: str | Path
    vle_max_rows: int | None
    chunk_size: int = 50_000

    def run(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        return _run_oulad_preprocessing(
            raw_dir=self.raw_dir,
            vle_max_rows=self.vle_max_rows,
            chunk_size=self.chunk_size,
        )


def _run_oulad_preprocessing(
    raw_dir: str | Path,
    vle_max_rows: int | None,
    chunk_size: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if vle_max_rows is None:
        raise ValueError(
            "Full OULAD preprocessing requires the partitioned bounded-memory writer. "
            "Keep preprocessing.oulad_vle_max_rows set for the current smoke pipeline."
        )
    root = Path(raw_dir)
    info = pd.read_csv(root / "studentInfo.csv")
    registrations = pd.read_csv(root / "studentRegistration.csv")
    assessments = pd.read_csv(root / "assessments.csv")
    submissions = pd.read_csv(root / "studentAssessment.csv")
    vle = pd.read_csv(root / "vle.csv")

    outcome_lookup = info.set_index(
        ["code_module", "code_presentation", "id_student"]
    )["final_result"]

    joined_assessments = submissions.merge(assessments, on="id_assessment", how="left")
    joined_assessments["module_id"] = _module_id(joined_assessments)
    joined_assessments["relative_day"] = joined_assessments["date_submitted"]
    joined_assessments["timestamp"] = joined_assessments["code_presentation"].map(
        _presentation_origin
    ) + joined_assessments["relative_day"].fillna(0)
    joined_assessments["outcome"] = [
        outcome_lookup.get((module, presentation, student), pd.NA)
        for module, presentation, student in joined_assessments[
            ["code_module", "code_presentation", "id_student"]
        ].itertuples(index=False, name=None)
    ]
    assessment_events = pd.DataFrame(
        {
            "dataset": "oulad",
            "student_id": "oulad:" + joined_assessments["id_student"].astype(str),
            "timestamp": joined_assessments["timestamp"].astype("int64"),
            "relative_day": joined_assessments["relative_day"],
            "item_id": "oulad:assessment:" + joined_assessments["id_assessment"].astype(str),
            "item_type": "assessment",
            "concept_ids": "module:" + joined_assessments["code_module"].astype(str),
            "action_type": "assessment_submit",
            "correctness": (joined_assessments["score"] >= 40).astype("Float64"),
            "score": joined_assessments["score"],
            "engagement": 1.0,
            "source": joined_assessments["assessment_type"],
            "module_id": joined_assessments["module_id"],
            "is_banked": joined_assessments["is_banked"].astype(bool),
            "outcome": joined_assessments["outcome"],
        }
    )

    activity_lookup = vle.set_index(
        ["code_module", "code_presentation", "id_site"]
    )["activity_type"]
    vle_frames: list[pd.DataFrame] = []
    observed = 0
    for chunk in pd.read_csv(root / "studentVle.csv", chunksize=chunk_size):
        if vle_max_rows is not None:
            remaining = vle_max_rows - observed
            if remaining <= 0:
                break
            chunk = chunk.iloc[:remaining]
        observed += len(chunk)
        chunk["module_id"] = _module_id(chunk)
        chunk["activity_type"] = [
            activity_lookup.get((module, presentation, site), "unknown")
            for module, presentation, site in chunk[
                ["code_module", "code_presentation", "id_site"]
            ].itertuples(index=False, name=None)
        ]
        chunk["timestamp"] = chunk["code_presentation"].map(_presentation_origin) + chunk["date"]
        chunk["outcome"] = [
            outcome_lookup.get((module, presentation, student), pd.NA)
            for module, presentation, student in chunk[
                ["code_module", "code_presentation", "id_student"]
            ].itertuples(index=False, name=None)
        ]
        vle_events = pd.DataFrame(
            {
                "dataset": "oulad",
                "student_id": "oulad:" + chunk["id_student"].astype(str),
                "timestamp": chunk["timestamp"].astype("int64"),
                "relative_day": chunk["date"],
                "item_id": "oulad:vle:" + chunk["id_site"].astype(str),
                "item_type": "vle_activity",
                "concept_ids": "module:" + chunk["code_module"].astype(str),
                "action_type": "vle_interaction",
                "engagement": chunk["sum_click"],
                "source": chunk["activity_type"],
                "module_id": chunk["module_id"],
                "outcome": chunk["outcome"],
            }
        )
        vle_frames.append(vle_events)

    registration_frames: list[pd.DataFrame] = []
    for action, date_column in [
        ("registration", "date_registration"),
        ("withdrawal", "date_unregistration"),
    ]:
        subset = registrations.dropna(subset=[date_column]).copy()
        subset["module_id"] = _module_id(subset)
        subset["timestamp"] = subset["code_presentation"].map(_presentation_origin) + subset[date_column]
        subset["outcome"] = [
            outcome_lookup.get((module, presentation, student), pd.NA)
            for module, presentation, student in subset[
                ["code_module", "code_presentation", "id_student"]
            ].itertuples(index=False, name=None)
        ]
        registration_frames.append(
            pd.DataFrame(
                {
                    "dataset": "oulad",
                    "student_id": "oulad:" + subset["id_student"].astype(str),
                    "timestamp": subset["timestamp"].astype("int64"),
                    "relative_day": subset[date_column],
                    "item_id": "oulad:module:" + subset["module_id"],
                    "item_type": "module",
                    "concept_ids": "module:" + subset["code_module"].astype(str),
                    "action_type": action,
                    "engagement": 0.0,
                    "source": "registration_system",
                    "module_id": subset["module_id"],
                    "outcome": subset["outcome"],
                }
            )
        )

    combined = pd.concat(
        [assessment_events, *vle_frames, *registration_frames], ignore_index=True
    )
    combined = combined.sort_values(["student_id", "timestamp"], kind="stable").reset_index(drop=True)
    combined["session_id"] = (
        combined["student_id"]
        + ":d"
        + combined["relative_day"].fillna(0).astype(int).astype(str)
    )
    events = ensure_common_schema(combined)
    validate_common_schema(events)
    return events, {
        "events": len(events),
        "students": int(events["student_id"].nunique()),
        "vle_rows_processed": observed,
        "assessment_events": len(assessment_events),
        "registration_events": len(registration_frames[0]),
        "withdrawal_events": len(registration_frames[1]),
    }

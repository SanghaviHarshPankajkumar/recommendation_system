from __future__ import annotations

import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


def _missing_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {key: int(value) for key, value in frame.isna().sum().items()}


def profile_ednet(
    zip_path: str | Path,
    questions_path: str | Path,
    lectures_path: str | Path,
    sample_users: int | None,
) -> dict[str, Any]:
    action_counts: Counter[str] = Counter()
    item_type_counts: Counter[str] = Counter()
    platform_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    sequence_lengths: list[int] = []
    timestamps: list[int] = []
    out_of_order_users = 0

    with zipfile.ZipFile(zip_path) as archive:
        members = [
            name
            for name in archive.namelist()
            if name.startswith("KT3/") and name.endswith(".csv")
        ]
        selected = members if sample_users is None else members[:sample_users]
        for member in selected:
            with archive.open(member) as handle:
                frame = pd.read_csv(handle)
            if frame.empty:
                sequence_lengths.append(0)
                continue
            numeric_time = pd.to_numeric(frame["timestamp"], errors="coerce").dropna()
            if not numeric_time.is_monotonic_increasing:
                out_of_order_users += 1
            timestamps.extend([int(numeric_time.min()), int(numeric_time.max())])
            sequence_lengths.append(len(frame))
            action_counts.update(frame["action_type"].dropna().astype(str))
            platform_counts.update(frame["platform"].dropna().astype(str))
            source_counts.update(frame["source"].dropna().astype(str))
            item_type_counts.update(
                frame["item_id"].fillna("").astype(str).str[:1].replace("", "missing")
            )

    questions = pd.read_csv(questions_path)
    lectures = pd.read_csv(lectures_path)
    question_tags = questions["tags"].replace(-1, pd.NA).dropna()
    lecture_tags = lectures["tags"].replace(-1, pd.NA).dropna()
    return {
        "archive_user_files": len(members),
        "sampled_users": len(selected),
        "sampled_interactions": int(sum(sequence_lengths)),
        "sequence_length": {
            "minimum": int(min(sequence_lengths, default=0)),
            "median": float(pd.Series(sequence_lengths).median()) if sequence_lengths else 0,
            "maximum": int(max(sequence_lengths, default=0)),
        },
        "timestamp_min": min(timestamps, default=None),
        "timestamp_max": max(timestamps, default=None),
        "out_of_order_users": out_of_order_users,
        "action_counts": dict(action_counts.most_common()),
        "item_prefix_counts": dict(item_type_counts.most_common()),
        "platform_counts": dict(platform_counts.most_common()),
        "top_sources": dict(source_counts.most_common(20)),
        "metadata": {
            "questions": len(questions),
            "questions_with_tags": int(question_tags.shape[0]),
            "lectures": len(lectures),
            "lectures_with_tags": int(lecture_tags.shape[0]),
            "question_missing": _missing_counts(questions),
            "lecture_missing": _missing_counts(lectures),
        },
    }


def _count_csv_rows(path: Path, chunk_size: int, max_rows: int | None = None) -> int:
    count = 0
    for chunk in pd.read_csv(path, chunksize=chunk_size):
        if max_rows is not None:
            remaining = max_rows - count
            if remaining <= 0:
                break
            chunk = chunk.iloc[:remaining]
        count += len(chunk)
    return count


def profile_oulad(
    raw_dir: str | Path,
    chunk_size: int,
    vle_max_rows: int | None,
) -> dict[str, Any]:
    root = Path(raw_dir)
    small_tables: dict[str, pd.DataFrame] = {}
    row_counts: dict[str, int] = {}
    schemas: dict[str, list[str]] = {}
    missing: dict[str, dict[str, int]] = {}

    for path in sorted(root.glob("*.csv")):
        schemas[path.name] = list(pd.read_csv(path, nrows=0).columns)
        if path.name == "studentVle.csv":
            row_counts[path.name] = _count_csv_rows(path, chunk_size, vle_max_rows)
            continue
        frame = pd.read_csv(path)
        small_tables[path.name] = frame
        row_counts[path.name] = len(frame)
        missing[path.name] = _missing_counts(frame)

    info = small_tables["studentInfo.csv"]
    assessment = small_tables["studentAssessment.csv"]
    registration = small_tables["studentRegistration.csv"]
    return {
        "row_counts": row_counts,
        "schemas": schemas,
        "missing_counts": missing,
        "students": int(info["id_student"].nunique()),
        "modules": int(info["code_module"].nunique()),
        "presentations": int(
            info[["code_module", "code_presentation"]].drop_duplicates().shape[0]
        ),
        "assessment_submissions": len(assessment),
        "withdrawals": int(registration["date_unregistration"].notna().sum()),
        "outcome_counts": {
            str(key): int(value) for key, value in info["final_result"].value_counts().items()
        },
        "vle_profile_is_bounded": vle_max_rows is not None,
    }


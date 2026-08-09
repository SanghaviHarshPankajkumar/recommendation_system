from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .schema import COMMON_EVENT_COLUMNS, validate_common_schema


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
    os.replace(temporary, path)


def _identity_hash(identity: dict[str, Any]) -> str:
    encoded = json.dumps(identity, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(slots=True)
class PartitionedCsvWriter:
    """Atomically write resumable split partitions and maintain a manifest."""

    output_dir: str | Path
    dataset: str
    identity: dict[str, Any]
    resume: bool = True
    _root: Path = field(init=False, repr=False)
    _state_path: Path = field(init=False, repr=False)
    _state: dict[str, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._root = Path(self.output_dir) / self.dataset
        self._root.mkdir(parents=True, exist_ok=True)
        self._state_path = self._root / "manifest.json"
        expected_hash = _identity_hash(self.identity)
        if self._state_path.exists():
            if not self.resume:
                raise FileExistsError(
                    f"Output state already exists at {self._state_path}; enable resume or use a new directory"
                )
            with self._state_path.open("r", encoding="utf-8") as handle:
                self._state = json.load(handle)
            if self._state["identity_hash"] != expected_hash:
                raise ValueError(
                    "Existing output was created with a different input/configuration identity"
                )
        else:
            self._state = {
                "dataset": self.dataset,
                "identity": self.identity,
                "identity_hash": expected_hash,
                "status": "in_progress",
                "next_partition": 0,
                "progress": {},
                "events": 0,
                "units": 0,
                "splits": {},
                "partitions": [],
            }
            _atomic_json(self._state, self._state_path)

    @property
    def progress(self) -> dict[str, Any]:
        return dict(self._state.get("progress", {}))

    @property
    def is_complete(self) -> bool:
        return self._state.get("status") == "complete"

    @property
    def manifest(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._state))

    def write(
        self,
        events: pd.DataFrame,
        progress: dict[str, Any],
        units: int,
    ) -> None:
        if events.empty:
            self._state["progress"] = progress
            self._state["units"] += units
            _atomic_json(self._state, self._state_path)
            return
        validate_common_schema(events)
        if "split" not in events.columns:
            raise ValueError("Partition events must contain a split column")
        partition_index = int(self._state["next_partition"])
        partition_record: dict[str, Any] = {
            "index": partition_index,
            "units": units,
            "events": len(events),
            "files": [],
        }
        for split_name, frame in events.groupby("split", sort=True):
            split = str(split_name)
            split_dir = self._root / split
            split_dir.mkdir(parents=True, exist_ok=True)
            destination = split_dir / f"part-{partition_index:05d}.csv.gz"
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            frame.to_csv(temporary, index=False, compression="gzip")
            os.replace(temporary, destination)
            record = {
                "split": split,
                "path": str(destination.relative_to(self._root)),
                "events": len(frame),
                "students": int(frame["student_id"].nunique()),
                "timestamp_min": int(frame["timestamp"].min()),
                "timestamp_max": int(frame["timestamp"].max()),
                "bytes": destination.stat().st_size,
            }
            partition_record["files"].append(record)
            totals = self._state["splits"].setdefault(
                split,
                {
                    "events": 0,
                    "students": 0,
                    "timestamp_min": None,
                    "timestamp_max": None,
                    "bytes": 0,
                    "files": 0,
                },
            )
            totals["events"] += record["events"]
            totals["students"] += record["students"]
            totals["bytes"] += record["bytes"]
            totals["files"] += 1
            totals["timestamp_min"] = (
                record["timestamp_min"]
                if totals["timestamp_min"] is None
                else min(totals["timestamp_min"], record["timestamp_min"])
            )
            totals["timestamp_max"] = (
                record["timestamp_max"]
                if totals["timestamp_max"] is None
                else max(totals["timestamp_max"], record["timestamp_max"])
            )
        self._state["events"] += len(events)
        self._state["units"] += units
        self._state["partitions"].append(partition_record)
        self._state["next_partition"] = partition_index + 1
        self._state["progress"] = progress
        _atomic_json(self._state, self._state_path)

    def complete(self, validation: dict[str, Any]) -> None:
        self._state["status"] = "complete"
        self._state["validation"] = validation
        _atomic_json(self._state, self._state_path)


@dataclass(slots=True)
class PartitionValidator:
    """Validate all generated compressed partitions with bounded memory."""

    dataset_dir: str | Path
    dataset: str
    chunk_size: int = 100_000

    def run(self) -> dict[str, Any]:
        root = Path(self.dataset_dir) / self.dataset
        files = sorted(root.glob("*/part-*.csv.gz"))
        errors: list[str] = []
        events = 0
        split_counts: dict[str, int] = {}
        required = set(COMMON_EVENT_COLUMNS) | {"split"}
        for path in files:
            expected_split = path.parent.name
            for chunk in pd.read_csv(
                path,
                chunksize=self.chunk_size,
                dtype={
                    "dataset": "string",
                    "student_id": "string",
                    "item_id": "string",
                    "concept_ids": "string",
                    "split": "string",
                },
            ):
                events += len(chunk)
                split_counts[expected_split] = split_counts.get(expected_split, 0) + len(chunk)
                missing_columns = required.difference(chunk.columns)
                if missing_columns:
                    errors.append(f"{path}: missing columns {sorted(missing_columns)}")
                    continue
                if chunk[["student_id", "timestamp", "item_id"]].isna().any().any():
                    errors.append(f"{path}: missing required identifier/time value")
                if not chunk["dataset"].eq(self.dataset).all():
                    errors.append(f"{path}: dataset value mismatch")
                if not chunk["split"].eq(expected_split).all():
                    errors.append(f"{path}: split value mismatch")
                correctness = pd.to_numeric(chunk["correctness"], errors="coerce").dropna()
                if not correctness.between(0, 1).all():
                    errors.append(f"{path}: correctness outside [0,1]")
        return {
            "passed": not errors,
            "files": len(files),
            "events": events,
            "split_counts": split_counts,
            "errors": errors[:100],
        }


def validate_temporal_order(events: pd.DataFrame) -> None:
    rank = events["split"].map({"train": 0, "validation": 1, "test": 2})
    ordered = events.assign(_split_rank=rank).sort_values(
        ["student_id", "timestamp"], kind="stable"
    )
    violations = sum(
        not group["_split_rank"].is_monotonic_increasing
        for _, group in ordered.groupby("student_id", sort=False)
    )
    if violations:
        raise ValueError(f"Temporal split order violated for {violations} students")

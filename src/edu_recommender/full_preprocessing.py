from __future__ import annotations

import json
import shutil
import sqlite3
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .config import load_config, write_json
from .partitioned import (
    PartitionedCsvWriter,
    PartitionValidator,
    validate_temporal_order,
)
from .preprocess_ednet import EdNetPreprocessor
from .schema import ensure_common_schema, validate_common_schema
from .splitting import TemporalSplitter


def _file_identity(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    stat = target.stat()
    return {
        "path": str(target.resolve()),
        "bytes": stat.st_size,
        "modified_ns": stat.st_mtime_ns,
    }


@dataclass(slots=True)
class ResourcePreflight:
    output_dir: str | Path
    minimum_free_gib: float

    def run(self) -> dict[str, Any]:
        root = Path(self.output_dir)
        root.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(root)
        free_gib = usage.free / (1024**3)
        result = {
            "free_gib": round(free_gib, 3),
            "minimum_free_gib": self.minimum_free_gib,
            "passed": free_gib >= self.minimum_free_gib,
        }
        if not result["passed"]:
            raise OSError(
                f"Only {free_gib:.2f} GiB free; {self.minimum_free_gib:.2f} GiB is required"
            )
        return result


@dataclass(slots=True)
class EdNetFullPreprocessor:
    normalizer: EdNetPreprocessor
    splitter: TemporalSplitter
    writer: PartitionedCsvWriter
    batch_users: int = 500
    user_limit: int | None = None

    def run(self) -> dict[str, Any]:
        if self.writer.is_complete:
            return self.writer.manifest
        start = int(self.writer.progress.get("next_member_index", 0))
        with zipfile.ZipFile(self.normalizer.zip_path) as archive:
            members = self.normalizer.list_members(archive)
            total = (
                len(members)
                if self.user_limit is None
                else min(len(members), self.user_limit)
            )
            while start < total:
                end = min(start + self.batch_users, total)
                frames, missing_metadata = self.normalizer.normalize_members(
                    archive, members[start:end]
                )
                events = (
                    pd.concat(frames, ignore_index=True)
                    if frames
                    else ensure_common_schema(pd.DataFrame())
                )
                if not events.empty:
                    events = self.splitter.split(events)
                    validate_temporal_order(events)
                cumulative_missing = int(
                    self.writer.progress.get("missing_concept_metadata", 0)
                ) + missing_metadata
                progress = {
                    "next_member_index": end,
                    "total_member_files": total,
                    "missing_concept_metadata": cumulative_missing,
                }
                self.writer.write(events, progress=progress, units=end - start)
                print(
                    f"EdNet: {end:,}/{total:,} users; {self.writer.manifest['events']:,} events",
                    flush=True,
                )
                start = end
        validation = PartitionValidator(
            Path(self.writer.output_dir), "ednet"
        ).run()
        if not validation["passed"]:
            raise ValueError(f"EdNet partition validation failed: {validation['errors'][:5]}")
        self.writer.complete(validation)
        return self.writer.manifest


def _presentation_origin(code: str) -> int:
    text = str(code)
    year = int(text[:4])
    return year * 1000 + (31 if text.endswith("B") else 273)


def _enrolment_key(module: Any, presentation: Any, student: Any) -> str:
    return f"{module}|{presentation}|{student}"


@dataclass(slots=True)
class OULADFullPreprocessor:
    raw_dir: str | Path
    splitter: TemporalSplitter
    writer: PartitionedCsvWriter
    chunk_size: int = 100_000
    static_batch_enrolments: int = 1_000
    vle_row_limit: int | None = None
    _root: Path = field(init=False, repr=False)
    _outcomes: dict[str, Any] = field(init=False, repr=False)
    _activities: dict[tuple[str, str, int], str] = field(init=False, repr=False)
    _static_events: dict[str, list[dict[str, Any]]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._root = Path(self.raw_dir)
        self._outcomes = {}
        self._activities = {}
        self._static_events = {}
        self._load_reference_data()

    def _load_reference_data(self) -> None:
        info = pd.read_csv(self._root / "studentInfo.csv")
        for row in info.itertuples(index=False):
            key = _enrolment_key(row.code_module, row.code_presentation, row.id_student)
            self._outcomes[key] = row.final_result
        vle = pd.read_csv(self._root / "vle.csv")
        self._activities = {
            (str(row.code_module), str(row.code_presentation), int(row.id_site)): str(
                row.activity_type
            )
            for row in vle.itertuples(index=False)
        }
        self._load_assessments()
        self._load_registration_events()

    def _append_static(self, key: str, event: dict[str, Any]) -> None:
        self._static_events.setdefault(key, []).append(event)

    def _student_id(self, key: str) -> str:
        module, presentation, student = key.split("|", 2)
        return f"oulad:{student}:{module}:{presentation}"

    def _load_assessments(self) -> None:
        definitions = pd.read_csv(self._root / "assessments.csv")
        submissions = pd.read_csv(self._root / "studentAssessment.csv")
        joined = submissions.merge(definitions, on="id_assessment", how="left")
        for row in joined.itertuples(index=False):
            key = _enrolment_key(row.code_module, row.code_presentation, row.id_student)
            relative_day = int(row.date_submitted)
            score = float(row.score) if pd.notna(row.score) else pd.NA
            correctness = float(score >= 40) if pd.notna(score) else pd.NA
            is_banked = bool(row.is_banked)
            module_id = f"{row.code_module}:{row.code_presentation}"
            self._append_static(
                key,
                {
                    "dataset": "oulad",
                    "student_id": self._student_id(key),
                    "timestamp": _presentation_origin(row.code_presentation) + relative_day,
                    "relative_day": relative_day,
                    "item_id": f"oulad:assessment:{row.id_assessment}",
                    "item_type": "assessment",
                    "concept_ids": f"module:{row.code_module}",
                    "action_type": (
                        "assessment_banked" if is_banked else "assessment_submit"
                    ),
                    "correctness": correctness,
                    "score": score,
                    "engagement": 1.0,
                    "source": (
                        f"{row.assessment_type}:banked" if is_banked else row.assessment_type
                    ),
                    "module_id": module_id,
                    "is_banked": is_banked,
                    "outcome": self._outcomes.get(key, pd.NA),
                },
            )

    def _load_registration_events(self) -> None:
        registrations = pd.read_csv(self._root / "studentRegistration.csv")
        for row in registrations.itertuples(index=False):
            key = _enrolment_key(row.code_module, row.code_presentation, row.id_student)
            module_id = f"{row.code_module}:{row.code_presentation}"
            for action, date in (
                ("registration", row.date_registration),
                ("withdrawal", row.date_unregistration),
            ):
                if pd.isna(date):
                    continue
                relative_day = int(date)
                self._append_static(
                    key,
                    {
                        "dataset": "oulad",
                        "student_id": self._student_id(key),
                        "timestamp": _presentation_origin(row.code_presentation) + relative_day,
                        "relative_day": relative_day,
                        "item_id": f"oulad:module:{module_id}",
                        "item_type": "module",
                        "concept_ids": f"module:{row.code_module}",
                        "action_type": action,
                        "engagement": 0.0,
                        "source": "registration_system",
                        "module_id": module_id,
                        "outcome": self._outcomes.get(key, pd.NA),
                    },
                )

    @staticmethod
    def _add_keys(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        result["_key"] = (
            result["code_module"].astype(str)
            + "|"
            + result["code_presentation"].astype(str)
            + "|"
            + result["id_student"].astype(str)
        )
        return result

    def _normalize_enrolment(
        self, key: str, vle_rows: pd.DataFrame | None
    ) -> pd.DataFrame:
        records = list(self._static_events.get(key, []))
        if vle_rows is not None and not vle_rows.empty:
            module, presentation, student = key.split("|", 2)
            module_id = f"{module}:{presentation}"
            student_id = self._student_id(key)
            for row in vle_rows.itertuples(index=False):
                relative_day = int(row.date)
                records.append(
                    {
                        "dataset": "oulad",
                        "student_id": student_id,
                        "timestamp": _presentation_origin(presentation) + relative_day,
                        "relative_day": relative_day,
                        "item_id": f"oulad:vle:{row.id_site}",
                        "item_type": "vle_activity",
                        "concept_ids": f"module:{module}",
                        "action_type": "vle_interaction",
                        "engagement": row.sum_click,
                        "source": self._activities.get(
                            (module, presentation, int(row.id_site)), "unknown"
                        ),
                        "module_id": module_id,
                        "outcome": self._outcomes.get(key, pd.NA),
                    }
                )
        events = ensure_common_schema(pd.DataFrame.from_records(records))
        if events.empty:
            return events
        events = events.sort_values("timestamp", kind="stable").reset_index(drop=True)
        events["session_id"] = (
            events["student_id"]
            + ":d"
            + events["relative_day"].fillna(0).astype(int).astype(str)
        )
        validate_common_schema(events)
        return events

    def _write_groups(
        self,
        groups: Iterable[tuple[str, pd.DataFrame | None]],
        next_enrolment_index: int,
        total_enrolments: int,
    ) -> int:
        frames: list[pd.DataFrame] = []
        for key, vle_rows in groups:
            frame = self._normalize_enrolment(key, vle_rows)
            if not frame.empty:
                frames.append(frame)
        units = next_enrolment_index - int(
            self.writer.progress.get("next_enrolment_index", 0)
        )
        if units <= 0:
            return 0
        events = (
            pd.concat(frames, ignore_index=True)
            if frames
            else ensure_common_schema(pd.DataFrame())
        )
        if not events.empty:
            events = self.splitter.split(events)
            validate_temporal_order(events)
        progress = {
            "staging_complete": True,
            "next_enrolment_index": next_enrolment_index,
            "total_enrolments": total_enrolments,
        }
        self.writer.write(events, progress=progress, units=units)
        print(
            f"OULAD: {next_enrolment_index:,}/{total_enrolments:,} enrolments; "
            f"{self.writer.manifest['events']:,} events",
            flush=True,
        )
        return units

    def _chunks(self, offset: int) -> Iterable[pd.DataFrame]:
        path = self._root / "studentVle.csv"
        if offset == 0:
            return pd.read_csv(path, chunksize=self.chunk_size)
        columns = list(pd.read_csv(path, nrows=0).columns)
        return pd.read_csv(
            path,
            names=columns,
            header=None,
            skiprows=offset + 1,
            chunksize=self.chunk_size,
        )

    @property
    def _staging_path(self) -> Path:
        path = Path(self.writer.output_dir) / "oulad" / "staging.sqlite"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _stage_vle(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._staging_path)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS vle (
                code_module TEXT NOT NULL,
                code_presentation TEXT NOT NULL,
                id_student INTEGER NOT NULL,
                id_site INTEGER NOT NULL,
                event_date INTEGER NOT NULL,
                sum_click INTEGER NOT NULL
            )
            """
        )
        row = connection.execute(
            "SELECT value FROM metadata WHERE key='staged_rows'"
        ).fetchone()
        staged_rows = int(row[0]) if row else 0
        complete = connection.execute(
            "SELECT value FROM metadata WHERE key='staging_complete'"
        ).fetchone()
        if not complete or complete[0] != "1":
            cursor = staged_rows
            for chunk in self._chunks(staged_rows):
                if self.vle_row_limit is not None:
                    remaining = self.vle_row_limit - cursor
                    if remaining <= 0:
                        break
                    chunk = chunk.iloc[:remaining]
                rows = [
                    (
                        str(row.code_module),
                        str(row.code_presentation),
                        int(row.id_student),
                        int(row.id_site),
                        int(row.date),
                        int(row.sum_click),
                    )
                    for row in chunk.itertuples(index=False)
                ]
                with connection:
                    connection.executemany(
                        "INSERT INTO vle VALUES (?, ?, ?, ?, ?, ?)", rows
                    )
                    cursor += len(rows)
                    connection.execute(
                        "INSERT OR REPLACE INTO metadata(key,value) VALUES('staged_rows',?)",
                        (str(cursor),),
                    )
                print(f"OULAD staging: {cursor:,} VLE rows", flush=True)
                if self.vle_row_limit is not None and cursor >= self.vle_row_limit:
                    break
            with connection:
                connection.execute(
                    "INSERT OR REPLACE INTO metadata(key,value) VALUES('staging_complete','1')"
                )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_vle_enrolment "
            "ON vle(code_module, code_presentation, id_student, event_date)"
        )
        connection.commit()
        return connection

    @staticmethod
    def _database_keys(connection: sqlite3.Connection) -> list[str]:
        return [
            _enrolment_key(module, presentation, student)
            for module, presentation, student in connection.execute(
                "SELECT DISTINCT code_module, code_presentation, id_student FROM vle"
            )
        ]

    @staticmethod
    def _fetch_vle(
        connection: sqlite3.Connection, key: str
    ) -> pd.DataFrame | None:
        module, presentation, student = key.split("|", 2)
        rows = connection.execute(
            """
            SELECT event_date, id_site, sum_click
            FROM vle
            WHERE code_module=? AND code_presentation=? AND id_student=?
            ORDER BY event_date
            """,
            (module, presentation, int(student)),
        ).fetchall()
        if not rows:
            return None
        return pd.DataFrame(rows, columns=["date", "id_site", "sum_click"])

    def _remove_staging(self) -> None:
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(str(self._staging_path) + suffix)
            if candidate.exists():
                candidate.unlink()

    def run(self) -> dict[str, Any]:
        if self.writer.is_complete:
            return self.writer.manifest
        connection = self._stage_vle()
        try:
            keys = sorted(set(self._database_keys(connection)) | set(self._static_events))
            total = len(keys)
            start = int(self.writer.progress.get("next_enrolment_index", 0))
            for index in range(start, total, self.static_batch_enrolments):
                batch_keys = keys[index : index + self.static_batch_enrolments]
                groups = [
                    (key, self._fetch_vle(connection, key)) for key in batch_keys
                ]
                self._write_groups(
                    groups,
                    next_enrolment_index=index + len(batch_keys),
                    total_enrolments=total,
                )
        finally:
            connection.close()
        validation = PartitionValidator(
            Path(self.writer.output_dir), "oulad"
        ).run()
        if not validation["passed"]:
            raise ValueError(f"OULAD partition validation failed: {validation['errors'][:5]}")
        self.writer.complete(validation)
        self._remove_staging()
        return self.writer.manifest


@dataclass(slots=True)
class FullScalePreprocessingPipeline:
    config_path: str | Path

    def _splitter(self, config: dict[str, Any]) -> TemporalSplitter:
        split = config["split"]
        return TemporalSplitter(
            split["train"],
            split["validation"],
            split["test"],
            split["minimum_events_for_holdout"],
        )

    def run(self, dataset: str = "all") -> dict[str, Any]:
        config, _ = load_config(self.config_path)
        paths = config["paths"]
        full = config["full_preprocessing"]
        output_dir = Path(paths["output_dir"])
        preflight = ResourcePreflight(
            output_dir, full["minimum_free_gib"]
        ).run()
        write_json(config, output_dir / "resolved_config.json")
        write_json(preflight, output_dir / "resource_preflight.json")
        splitter = self._splitter(config)
        results: dict[str, Any] = {"resource_preflight": preflight}

        if dataset in {"all", "ednet"}:
            identity = {
                "schema_version": 3,
                "input": _file_identity(paths["ednet_zip"]),
                "questions": _file_identity(paths["ednet_questions"]),
                "lectures": _file_identity(paths["ednet_lectures"]),
                "batch_users": full["ednet_batch_users"],
                "user_limit": full.get("ednet_user_limit"),
                "split": config["split"],
            }
            writer = PartitionedCsvWriter(
                output_dir, "ednet", identity, resume=full["resume"]
            )
            normalizer = EdNetPreprocessor(
                paths["ednet_zip"],
                paths["ednet_questions"],
                paths["ednet_lectures"],
                max_users=1,
                session_gap_minutes=full["session_gap_minutes"],
            )
            results["ednet"] = EdNetFullPreprocessor(
                normalizer,
                splitter,
                writer,
                batch_users=full["ednet_batch_users"],
                user_limit=full.get("ednet_user_limit"),
            ).run()

        if dataset in {"all", "oulad"}:
            identity = {
                "schema_version": 3,
                "student_vle": _file_identity(
                    Path(paths["oulad_raw"]) / "studentVle.csv"
                ),
                "chunk_size": full["oulad_chunk_size"],
                "vle_row_limit": full.get("oulad_vle_row_limit"),
                "split": config["split"],
                "student_scope": "module_presentation_enrolment",
            }
            writer = PartitionedCsvWriter(
                output_dir, "oulad", identity, resume=full["resume"]
            )
            results["oulad"] = OULADFullPreprocessor(
                paths["oulad_raw"],
                splitter,
                writer,
                chunk_size=full["oulad_chunk_size"],
                static_batch_enrolments=full["oulad_static_batch_enrolments"],
                vle_row_limit=full.get("oulad_vle_row_limit"),
            ).run()

        summary = {
            key: {
                "status": value.get("status"),
                "events": value.get("events"),
                "units": value.get("units"),
                "splits": value.get("splits"),
            }
            for key, value in results.items()
            if key in {"ednet", "oulad"}
        }
        write_json(summary, output_dir / "summary.json")
        return results

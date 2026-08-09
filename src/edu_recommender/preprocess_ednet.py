from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .schema import ensure_common_schema, validate_common_schema


def _clean_tag(value: Any) -> str | pd.NA:
    if pd.isna(value) or str(value).strip() in {"", "-1"}:
        return pd.NA
    return str(value).strip()


@dataclass(slots=True)
class EdNetPreprocessor:
    """Normalize EdNet-KT3 interactions into the shared event schema."""

    zip_path: str | Path
    questions_path: str | Path
    lectures_path: str | Path
    max_users: int | None
    session_gap_minutes: int = 30
    _question_tags: dict[str, Any] = field(init=False, repr=False)
    _correct_answers: dict[str, str] = field(init=False, repr=False)
    _question_bundle: dict[str, str] = field(init=False, repr=False)
    _lecture_tags: dict[str, Any] = field(init=False, repr=False)
    _bundle_tags: dict[str, str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        questions = pd.read_csv(self.questions_path)
        lectures = pd.read_csv(self.lectures_path)
        questions["question_id"] = questions["question_id"].astype(str)
        lectures["lecture_id"] = lectures["lecture_id"].astype(str)
        self._question_tags = (
            questions.set_index("question_id")["tags"].map(_clean_tag).to_dict()
        )
        self._correct_answers = (
            questions.set_index("question_id")["correct_answer"].astype(str).to_dict()
        )
        self._question_bundle = (
            questions.set_index("question_id")["bundle_id"].astype(str).to_dict()
        )
        self._lecture_tags = (
            lectures.set_index("lecture_id")["tags"].map(_clean_tag).to_dict()
        )
        self._bundle_tags = (
            questions.assign(tags=questions["tags"].map(_clean_tag))
            .dropna(subset=["tags"])
            .groupby("bundle_id")["tags"]
            .apply(lambda values: ";".join(sorted(set(";".join(values).split(";")))))
            .to_dict()
        )

    @staticmethod
    def list_members(archive: zipfile.ZipFile) -> list[str]:
        return [
            name
            for name in archive.namelist()
            if name.startswith("KT3/") and name.endswith(".csv")
        ]

    @staticmethod
    def _item_type(item_id: str) -> str:
        return {
            "q": "question",
            "l": "lecture",
            "e": "explanation",
            "b": "bundle",
        }.get(item_id[:1], "other")

    def _concepts(self, item: str) -> Any:
        if item.startswith("q"):
            return self._question_tags.get(item, pd.NA)
        if item.startswith("l"):
            return self._lecture_tags.get(item, pd.NA)
        if item.startswith("e"):
            return self._bundle_tags.get("b" + item[1:], pd.NA)
        if item.startswith("b"):
            return self._bundle_tags.get(item, pd.NA)
        return pd.NA

    def _mark_final_responses(self, frame: pd.DataFrame) -> pd.Series:
        final = pd.Series(False, index=frame.index, dtype=bool)
        is_submit = frame["action_type"].eq("submit") & frame["item_id"].str.startswith("b")
        next_submit_bundle = frame["item_id"].where(is_submit).bfill()
        submit_sequence = is_submit.cumsum()
        response_bundle = frame["item_id"].map(self._question_bundle)
        candidate = (
            frame["action_type"].eq("respond")
            & frame["item_id"].str.startswith("q")
            & next_submit_bundle.notna()
            & response_bundle.eq(next_submit_bundle)
        )
        if candidate.any():
            candidate_rows = frame.loc[candidate, ["item_id"]].copy()
            candidate_rows["_submit_sequence"] = submit_sequence.loc[candidate]
            final_indices = (
                candidate_rows.groupby(["_submit_sequence", "item_id"], sort=False)
                .tail(1)
                .index
            )
            final.loc[final_indices] = True
        return final

    @staticmethod
    def _elapsed_time(frame: pd.DataFrame) -> pd.Series:
        durations = pd.Series(pd.NA, index=frame.index, dtype="Float64")
        is_enter = frame["action_type"].eq("enter")
        cycle = is_enter.groupby(frame["item_id"]).cumsum()
        entered_at = frame["timestamp"].where(is_enter).groupby(
            [frame["item_id"], cycle]
        ).transform("max")
        quit_candidate = (
            frame["action_type"].eq("quit") & cycle.gt(0) & entered_at.notna()
        )
        is_quit = pd.Series(False, index=frame.index, dtype=bool)
        if quit_candidate.any():
            candidate_rows = frame.loc[quit_candidate, ["item_id"]].copy()
            candidate_rows["_cycle"] = cycle.loc[quit_candidate]
            first_quit = candidate_rows.groupby(["item_id", "_cycle"], sort=False).head(1)
            is_quit.loc[first_quit.index] = True
        durations.loc[is_quit] = (
            frame.loc[is_quit, "timestamp"] - entered_at.loc[is_quit]
        ).clip(lower=0)
        return durations

    def normalize_member(
        self, archive: zipfile.ZipFile, member: str
    ) -> tuple[pd.DataFrame, int]:
        frame = pd.read_csv(io.BytesIO(archive.read(member)))
        if frame.empty:
            return ensure_common_schema(pd.DataFrame()), 0
        student = Path(member).stem.removeprefix("u")
        return self.normalize_frame(frame, student)

    def normalize_members(
        self, archive: zipfile.ZipFile, members: list[str]
    ) -> tuple[list[pd.DataFrame], int]:
        """Parse a batch once, then apply the same per-user normalization."""
        payloads: list[bytes] = []
        students: list[str] = []
        header: bytes | None = None
        for member in members:
            raw = archive.read(member)
            first_line, separator, data = raw.partition(b"\n")
            if not separator:
                continue
            if header is None:
                header = first_line.rstrip(b"\r")
            normalized_data = data.rstrip(b"\r\n")
            if not normalized_data:
                continue
            normalized_data += b"\n"
            row_count = normalized_data.count(b"\n")
            payloads.append(normalized_data)
            students.extend([Path(member).stem.removeprefix("u")] * row_count)
        if header is None or not payloads:
            return [], 0
        frame = pd.read_csv(io.BytesIO(header + b"\n" + b"".join(payloads)))
        if len(frame) != len(students):
            raise ValueError(
                f"EdNet batch row attribution mismatch: {len(frame)} parsed vs {len(students)} attributed"
            )
        frame["_student"] = students
        normalized, missing_metadata = self.normalize_batch_frame(frame)
        return ([normalized] if not normalized.empty else []), missing_metadata

    def normalize_batch_frame(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """Vectorize normalization across a batch while preserving student boundaries."""
        frame = frame.copy()
        frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
        frame = frame.dropna(subset=["_student", "timestamp", "item_id", "action_type"])
        frame["timestamp"] = frame["timestamp"].astype("int64")
        frame["_student"] = frame["_student"].astype(str)
        frame = frame.sort_values(["_student", "timestamp"], kind="stable").reset_index(drop=True)
        frame["item_id"] = frame["item_id"].astype(str)
        frame["item_type"] = frame["item_id"].map(self._item_type)

        is_submit = frame["action_type"].eq("submit") & frame["item_id"].str.startswith("b")
        submit_sequence = is_submit.groupby(frame["_student"]).cumsum()
        next_submit_bundle = frame["item_id"].where(is_submit).groupby(frame["_student"]).bfill()
        response_bundle = frame["item_id"].map(self._question_bundle)
        response_candidate = (
            frame["action_type"].eq("respond")
            & frame["item_id"].str.startswith("q")
            & next_submit_bundle.notna()
            & response_bundle.eq(next_submit_bundle)
        )
        final = pd.Series(False, index=frame.index, dtype=bool)
        if response_candidate.any():
            candidate_rows = frame.loc[response_candidate, ["_student", "item_id"]].copy()
            candidate_rows["_submit_sequence"] = submit_sequence.loc[response_candidate]
            final_indices = candidate_rows.groupby(
                ["_student", "_submit_sequence", "item_id"], sort=False
            ).tail(1).index
            final.loc[final_indices] = True

        is_enter = frame["action_type"].eq("enter")
        cycle = is_enter.groupby([frame["_student"], frame["item_id"]]).cumsum()
        entered_at = frame["timestamp"].where(is_enter).groupby(
            [frame["_student"], frame["item_id"], cycle]
        ).transform("max")
        quit_candidate = (
            frame["action_type"].eq("quit") & cycle.gt(0) & entered_at.notna()
        )
        is_quit = pd.Series(False, index=frame.index, dtype=bool)
        if quit_candidate.any():
            quit_rows = frame.loc[quit_candidate, ["_student", "item_id"]].copy()
            quit_rows["_cycle"] = cycle.loc[quit_candidate]
            first_quit = quit_rows.groupby(
                ["_student", "item_id", "_cycle"], sort=False
            ).head(1)
            is_quit.loc[first_quit.index] = True
        elapsed = pd.Series(pd.NA, index=frame.index, dtype="Float64")
        elapsed.loc[is_quit] = (
            frame.loc[is_quit, "timestamp"] - entered_at.loc[is_quit]
        ).clip(lower=0)

        frame["concept_ids"] = frame["item_id"].map(self._concepts)
        missing_metadata = int(
            frame.loc[frame["item_type"].isin(["question", "lecture"]), "concept_ids"]
            .isna()
            .sum()
        )
        correctness = pd.Series(pd.NA, index=frame.index, dtype="Float64")
        response_mask = final & frame["item_id"].str.startswith("q")
        expected = frame.loc[response_mask, "item_id"].map(self._correct_answers)
        actual = frame.loc[response_mask, "user_answer"].astype("string")
        correctness.loc[response_mask] = (actual == expected.astype("string")).astype(float)
        gaps = frame.groupby("_student", sort=False)["timestamp"].diff().fillna(0)
        session_number = (gaps > self.session_gap_minutes * 60_000).groupby(
            frame["_student"]
        ).cumsum()
        normalized = ensure_common_schema(
            pd.DataFrame(
                {
                    "dataset": "ednet",
                    "student_id": "ednet:" + frame["_student"],
                    "timestamp": frame["timestamp"],
                    "item_id": "ednet:" + frame["item_id"],
                    "item_type": frame["item_type"],
                    "concept_ids": frame["concept_ids"],
                    "action_type": frame["action_type"],
                    "correctness": correctness,
                    "elapsed_time_ms": elapsed,
                    "engagement": 1.0,
                    "source": frame["source"],
                    "session_id": (
                        "ednet:" + frame["_student"] + ":s" + session_number.astype(str)
                    ),
                    "final_response": final,
                }
            )
        )
        validate_common_schema(normalized)
        return normalized, missing_metadata

    def normalize_frame(
        self, frame: pd.DataFrame, student: str
    ) -> tuple[pd.DataFrame, int]:
        frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
        frame = frame.dropna(subset=["timestamp", "item_id", "action_type"])
        frame["timestamp"] = frame["timestamp"].astype("int64")
        frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
        frame["item_id"] = frame["item_id"].astype(str)
        frame["item_type"] = frame["item_id"].map(self._item_type)
        final = self._mark_final_responses(frame)
        elapsed = self._elapsed_time(frame)
        frame["concept_ids"] = frame["item_id"].map(self._concepts)
        missing_metadata = int(
            frame.loc[frame["item_type"].isin(["question", "lecture"]), "concept_ids"]
            .isna()
            .sum()
        )
        correctness = pd.Series(pd.NA, index=frame.index, dtype="Float64")
        response_mask = final & frame["item_id"].str.startswith("q")
        expected = frame.loc[response_mask, "item_id"].map(self._correct_answers)
        actual = frame.loc[response_mask, "user_answer"].astype("string")
        correctness.loc[response_mask] = (actual == expected.astype("string")).astype(float)
        gap = frame["timestamp"].diff().fillna(0)
        session_number = (gap > self.session_gap_minutes * 60_000).cumsum()
        normalized = ensure_common_schema(
            pd.DataFrame(
                {
                    "dataset": "ednet",
                    "student_id": f"ednet:{student}",
                    "timestamp": frame["timestamp"],
                    "item_id": "ednet:" + frame["item_id"],
                    "item_type": frame["item_type"],
                    "concept_ids": frame["concept_ids"],
                    "action_type": frame["action_type"],
                    "correctness": correctness,
                    "elapsed_time_ms": elapsed,
                    "engagement": 1.0,
                    "source": frame["source"],
                    "session_id": [f"ednet:{student}:s{n}" for n in session_number],
                    "final_response": final,
                }
            )
        )
        validate_common_schema(normalized)
        return normalized, missing_metadata

    def run(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        if self.max_users is None:
            raise ValueError(
                "Full EdNet preprocessing requires EdNetFullPreprocessor. "
                "Keep max_users set for this in-memory smoke preprocessor."
            )
        frames: list[pd.DataFrame] = []
        missing_metadata = 0
        with zipfile.ZipFile(self.zip_path) as archive:
            selected = self.list_members(archive)[: self.max_users]
            for member in selected:
                normalized, missing = self.normalize_member(archive, member)
                if not normalized.empty:
                    frames.append(normalized)
                missing_metadata += missing
        events = (
            pd.concat(frames, ignore_index=True)
            if frames
            else ensure_common_schema(pd.DataFrame())
        )
        validate_common_schema(events)
        return events, {
            "selected_user_files": len(selected),
            "events": len(events),
            "students": int(events["student_id"].nunique()),
            "final_responses": int(events["final_response"].fillna(False).sum()),
            "events_missing_concept_metadata": missing_metadata,
        }

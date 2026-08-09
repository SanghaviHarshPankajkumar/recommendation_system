from __future__ import annotations

import gzip
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd


PAD_TOKEN = 0
UNK_TOKEN = 1
SPLIT_IDS = {"train": 0, "validation": 1, "test": 2}
SPLIT_NAMES = {value: key for key, value in SPLIT_IDS.items()}
VOCAB_FIELDS = ("item_id", "action_type", "item_type", "concept_ids", "module_id", "source")
PACKED_EVENT_FIELDS = (
    "item_tokens", "action_tokens", "item_type_tokens", "module_tokens", "source_tokens",
    "timestamps", "relative_days", "time_gaps", "correctness", "scores", "elapsed_log1p",
    "engagement_log1p", "final_response", "is_banked", "split_ids",
)


def _present(value: object) -> bool:
    return not pd.isna(value) and str(value).strip() != ""


def _concepts(value: object) -> tuple[str, ...]:
    if not _present(value):
        return ()
    return tuple(part.strip() for part in str(value).split(";") if part.strip())


class CategoricalVocabulary:
    """Train-fitted categorical vocabulary with fixed PAD and UNK identifiers."""

    def __init__(self, values: Iterable[str] = ()):
        ordered = sorted({str(value) for value in values if str(value)})
        self.token_to_id = {"<PAD>": PAD_TOKEN, "<UNK>": UNK_TOKEN}
        self.token_to_id.update({value: index + 2 for index, value in enumerate(ordered)})

    def encode(self, values: pd.Series, dtype: str = "int32") -> np.ndarray:
        return values.astype("string").map(self.token_to_id).fillna(UNK_TOKEN).to_numpy(dtype=dtype)

    def encode_one(self, value: object) -> int:
        if not _present(value):
            return UNK_TOKEN
        return self.token_to_id.get(str(value), UNK_TOKEN)

    def to_dict(self) -> dict[str, int]:
        return self.token_to_id


class TrainVocabularyFitter:
    """Fit categorical encoders and logged-action support using training events only."""

    usecols = [
        "item_id", "action_type", "item_type", "concept_ids", "module_id", "source",
        "final_response",
    ]

    def __init__(
        self,
        dataset: str,
        train_dir: Path,
        chunksize: int = 250_000,
        official_seeds: dict[str, set[str]] | None = None,
    ):
        self.dataset = dataset
        self.train_dir = Path(train_dir)
        self.chunksize = chunksize
        self.official_seeds = official_seeds or {}

    def fit(self) -> tuple[dict[str, CategoricalVocabulary], pd.DataFrame, dict[str, int]]:
        values: dict[str, set[str]] = {
            field: set(self.official_seeds.get(field, set())) for field in VOCAB_FIELDS
        }
        support: Counter[tuple[str, str, str]] = Counter()
        event_count = 0
        for path in sorted(self.train_dir.glob("part-*.csv.gz")):
            for chunk in pd.read_csv(path, usecols=self.usecols, chunksize=self.chunksize):
                event_count += len(chunk)
                for field in ("item_id", "action_type", "item_type", "module_id", "source"):
                    values[field].update(chunk[field].dropna().astype(str).unique())
                for value in chunk["concept_ids"].dropna().unique():
                    values["concept_ids"].update(_concepts(value))
                candidate_mask = self._candidate_observation_mask(chunk)
                candidates = chunk.loc[candidate_mask, ["item_id", "item_type", "module_id"]].copy()
                candidates["module_id"] = candidates["module_id"].fillna("").astype(str)
                counts = candidates.groupby(["item_id", "item_type", "module_id"], dropna=False).size()
                support.update({tuple(map(str, key)): int(count) for key, count in counts.items()})
        vocabularies = {field: CategoricalVocabulary(items) for field, items in values.items()}
        support_rows = [
            {"item_id": item, "item_type": item_type, "module_id": module, "train_support": count}
            for (item, item_type, module), count in support.items()
        ]
        support_frame = pd.DataFrame(
            support_rows, columns=["item_id", "item_type", "module_id", "train_support"]
        ).sort_values(["item_type", "module_id", "item_id"]).reset_index(drop=True)
        stats = {"train_events_scanned": event_count, **{f"{field}_vocab_size": len(vocab.token_to_id) for field, vocab in vocabularies.items()}}
        return vocabularies, support_frame, stats

    def _candidate_observation_mask(self, frame: pd.DataFrame) -> pd.Series:
        if self.dataset == "ednet":
            final = frame["final_response"].astype(str).str.lower().eq("true")
            return (
                (frame["item_type"].eq("question") & final)
                | (frame["item_type"].isin(["lecture", "explanation"]) & frame["action_type"].eq("enter"))
            )
        if self.dataset == "oulad":
            return frame["item_type"].isin(["vle_activity", "assessment"])
        raise ValueError(f"Unsupported dataset: {self.dataset}")


@dataclass(frozen=True)
class WindowConfig:
    max_length: int = 128
    stride: int = 64

    def __post_init__(self) -> None:
        if self.max_length < 2:
            raise ValueError("max_length must be at least 2")
        if self.stride < 1 or self.stride >= self.max_length:
            raise ValueError("stride must be in [1, max_length)")


class PackedSequenceValidator:
    @staticmethod
    def validate(arrays: dict[str, np.ndarray], window_config: WindowConfig) -> dict[str, object]:
        event_count = len(arrays["item_tokens"])
        for field in PACKED_EVENT_FIELDS:
            if len(arrays[field]) != event_count:
                raise ValueError(f"Packed field {field} has the wrong length")
        offsets = arrays["student_offsets"]
        if offsets[0] != 0 or offsets[-1] != event_count or np.any(np.diff(offsets) <= 0):
            raise ValueError("Invalid student offsets")
        for start, end in zip(offsets[:-1], offsets[1:]):
            if np.any(np.diff(arrays["timestamps"][start:end]) < 0):
                raise ValueError("Events are not temporal within a learner")
        concept_offsets = arrays["concept_offsets"]
        if len(concept_offsets) != event_count + 1 or concept_offsets[0] != 0:
            raise ValueError("Invalid concept offsets")
        if concept_offsets[-1] != len(arrays["concept_values"]) or np.any(np.diff(concept_offsets) < 0):
            raise ValueError("Invalid concept CSR arrays")

        coverage = np.zeros(event_count, dtype=np.uint8)
        for start, end, target_start, split_id in zip(
            arrays["window_starts"], arrays["window_ends"], arrays["window_target_starts"], arrays["window_split_ids"]
        ):
            if not (0 <= start < end <= event_count and end - start <= window_config.max_length):
                raise ValueError("Window bounds are invalid")
            target_global = int(start + target_start)
            if not (start < target_global < end):
                raise ValueError("Window target alignment is invalid")
            if np.any(arrays["split_ids"][target_global:end] != split_id):
                raise ValueError("A window target range crosses split boundaries")
            coverage[target_global:end] += 1
        expected = np.ones(event_count, dtype=np.uint8)
        expected[offsets[:-1]] = 0
        if not np.array_equal(coverage, expected):
            raise ValueError("Every non-initial learner event must be targeted exactly once")
        return {
            "passed": True,
            "event_count": event_count,
            "student_count": len(offsets) - 1,
            "window_count": len(arrays["window_starts"]),
            "target_counts": {
                SPLIT_NAMES[split_id]: int(np.sum((arrays["split_ids"] == split_id) & (expected == 1)))
                for split_id in SPLIT_NAMES
            },
        }


class PackedSequenceWriter:
    """Encode split files into compact per-partition learner arrays and lazy window indices."""

    read_columns = [
        "student_id", "timestamp", "relative_day", "item_id", "item_type", "concept_ids",
        "action_type", "correctness", "score", "elapsed_time_ms", "engagement", "source",
        "module_id", "final_response", "is_banked",
    ]

    def __init__(
        self,
        dataset: str,
        input_root: Path,
        output_dir: Path,
        vocabularies: dict[str, CategoricalVocabulary],
        window_config: WindowConfig,
    ):
        self.dataset = dataset
        self.input_root = Path(input_root)
        self.output_dir = Path(output_dir)
        self.vocabularies = vocabularies
        self.window_config = window_config

    def run(self) -> dict[str, object]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        partition_names = sorted(path.name for path in (self.input_root / "train").glob("part-*.csv.gz"))
        aggregate = {"events": 0, "students": 0, "windows": 0, "targets": {name: 0 for name in SPLIT_IDS}}
        validations = []
        for index, name in enumerate(partition_names):
            frames = []
            for split_name, split_id in SPLIT_IDS.items():
                path = self.input_root / split_name / name
                if path.exists():
                    frame = pd.read_csv(
                        path,
                        usecols=self.read_columns,
                        dtype={
                            "student_id": "string", "item_id": "string", "item_type": "string",
                            "concept_ids": "string", "action_type": "string", "source": "string",
                            "module_id": "string",
                        },
                        low_memory=False,
                    )
                    frame["_split_id"] = split_id
                    frame["_row_order"] = np.arange(len(frame), dtype=np.int64)
                    frames.append(frame)
            combined = pd.concat(frames, ignore_index=True)
            combined = combined.sort_values(
                ["student_id", "timestamp", "_split_id", "_row_order"], kind="stable"
            ).reset_index(drop=True)
            arrays = self._encode(combined)
            validation = PackedSequenceValidator.validate(arrays, self.window_config)
            validations.append(validation)
            np.savez_compressed(self.output_dir / name.replace(".csv.gz", ".npz"), **arrays)
            aggregate["events"] += validation["event_count"]
            aggregate["students"] += validation["student_count"]
            aggregate["windows"] += validation["window_count"]
            for split_name, count in validation["target_counts"].items():
                aggregate["targets"][split_name] += count
        return {"partition_count": len(partition_names), **aggregate, "all_validations_passed": all(v["passed"] for v in validations)}

    def _encode(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        n = len(frame)
        students = frame["student_id"].astype(str).to_numpy()
        changes = np.r_[True, students[1:] != students[:-1]]
        starts = np.flatnonzero(changes).astype(np.int64)
        offsets = np.r_[starts, n].astype(np.int64)
        student_ids = students[starts]
        timestamps = pd.to_numeric(frame["timestamp"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
        time_gaps = np.zeros(n, dtype=np.float32)
        if n:
            differences = np.diff(timestamps, prepend=timestamps[0]).astype(np.float64)
            differences[starts] = 0
            divisor = 1000.0 if self.dataset == "ednet" else 1.0
            time_gaps = np.log1p(np.maximum(differences / divisor, 0)).astype(np.float32)

        concept_offsets = np.zeros(n + 1, dtype=np.int64)
        concept_values: list[int] = []
        concept_vocab = self.vocabularies["concept_ids"]
        for row_index, value in enumerate(frame["concept_ids"]):
            concept_values.extend(concept_vocab.encode_one(concept) for concept in _concepts(value))
            concept_offsets[row_index + 1] = len(concept_values)

        arrays: dict[str, np.ndarray] = {
            "item_tokens": self.vocabularies["item_id"].encode(frame["item_id"]),
            "action_tokens": self.vocabularies["action_type"].encode(frame["action_type"], "int16"),
            "item_type_tokens": self.vocabularies["item_type"].encode(frame["item_type"], "int16"),
            "module_tokens": self.vocabularies["module_id"].encode(frame["module_id"], "int16"),
            "source_tokens": self.vocabularies["source"].encode(frame["source"], "int16"),
            "timestamps": timestamps,
            "relative_days": pd.to_numeric(frame["relative_day"], errors="coerce").to_numpy(dtype=np.float32),
            "time_gaps": time_gaps,
            "correctness": self._nullable_binary(frame["correctness"]),
            "scores": pd.to_numeric(frame["score"], errors="coerce").to_numpy(dtype=np.float32),
            "elapsed_log1p": np.log1p(np.maximum(pd.to_numeric(frame["elapsed_time_ms"], errors="coerce").fillna(0).to_numpy(dtype=np.float64) / 1000.0, 0)).astype(np.float32),
            "engagement_log1p": np.log1p(np.maximum(pd.to_numeric(frame["engagement"], errors="coerce").fillna(0).to_numpy(dtype=np.float64), 0)).astype(np.float32),
            "final_response": frame["final_response"].astype(str).str.lower().eq("true").to_numpy(dtype=np.uint8),
            "is_banked": self._nullable_binary(frame["is_banked"]),
            "split_ids": frame["_split_id"].to_numpy(dtype=np.uint8),
            "concept_values": np.asarray(concept_values, dtype=np.int32),
            "concept_offsets": concept_offsets,
            "student_offsets": offsets,
            "student_ids": student_ids.astype(str),
        }
        arrays.update(self._windows(offsets, arrays["split_ids"]))
        return arrays

    @staticmethod
    def _nullable_binary(series: pd.Series) -> np.ndarray:
        numeric = pd.to_numeric(series, errors="coerce")
        return np.where(numeric.isna(), -1, (numeric >= 0.5).astype(np.int8)).astype(np.int8)

    def _windows(self, offsets: np.ndarray, split_ids: np.ndarray) -> dict[str, np.ndarray]:
        starts: list[int] = []
        ends: list[int] = []
        target_starts: list[int] = []
        window_splits: list[int] = []
        student_indices: list[int] = []
        for student_index, (student_start, student_end) in enumerate(zip(offsets[:-1], offsets[1:])):
            first_target = int(student_start + 1)
            for split_id in SPLIT_NAMES:
                positions = np.flatnonzero(split_ids[first_target:student_end] == split_id) + first_target
                if not len(positions):
                    continue
                block_start, block_end = int(positions[0]), int(positions[-1] + 1)
                if len(positions) != block_end - block_start:
                    raise ValueError("A learner split is not temporally contiguous")
                cursor = block_start
                while cursor < block_end:
                    target_end = min(block_end, cursor + self.window_config.stride)
                    window_start = max(int(student_start), target_end - self.window_config.max_length)
                    starts.append(window_start)
                    ends.append(target_end)
                    target_starts.append(cursor - window_start)
                    window_splits.append(split_id)
                    student_indices.append(student_index)
                    cursor = target_end
        return {
            "window_starts": np.asarray(starts, dtype=np.int64),
            "window_ends": np.asarray(ends, dtype=np.int64),
            "window_target_starts": np.asarray(target_starts, dtype=np.int16),
            "window_split_ids": np.asarray(window_splits, dtype=np.uint8),
            "window_student_indices": np.asarray(student_indices, dtype=np.int32),
        }


class LazySequenceWindows:
    """Read packed partitions and materialize padded fixed-length model examples on demand."""

    def __init__(self, path: Path, max_length: int, max_concepts_per_event: int = 9):
        self.path = Path(path)
        self.max_length = max_length
        self.max_concepts_per_event = max_concepts_per_event
        self.arrays = np.load(self.path, allow_pickle=False)

    def iter_split(self, split: str) -> Iterator[dict[str, np.ndarray]]:
        split_id = SPLIT_IDS[split]
        indices = np.flatnonzero(self.arrays["window_split_ids"] == split_id)
        for index in indices:
            yield self.get(int(index))

    def get(self, index: int) -> dict[str, np.ndarray]:
        start = int(self.arrays["window_starts"][index])
        end = int(self.arrays["window_ends"][index])
        target_start = int(self.arrays["window_target_starts"][index])
        length = end - start
        result: dict[str, np.ndarray] = {}
        for field in PACKED_EVENT_FIELDS:
            source = self.arrays[field][start:end]
            pad_value = 0
            if field in ("correctness", "is_banked"):
                pad_value = -1
            target = np.full(self.max_length, pad_value, dtype=source.dtype)
            target[:length] = source
            result[field] = target
        result["attention_mask"] = np.r_[np.ones(length, dtype=np.uint8), np.zeros(self.max_length - length, dtype=np.uint8)]
        result["input_attention_mask"] = result["attention_mask"][:-1]
        concept_tokens = np.zeros((self.max_length, self.max_concepts_per_event), dtype=np.int32)
        for local_index, global_index in enumerate(range(start, end)):
            concept_start = int(self.arrays["concept_offsets"][global_index])
            concept_end = int(self.arrays["concept_offsets"][global_index + 1])
            values = self.arrays["concept_values"][concept_start:concept_end]
            if len(values) > self.max_concepts_per_event:
                raise ValueError(
                    f"Event has {len(values)} concepts, exceeding max_concepts_per_event={self.max_concepts_per_event}"
                )
            concept_tokens[local_index, :len(values)] = values
        result["concept_tokens"] = concept_tokens
        result["input_concept_tokens"] = concept_tokens[:-1]
        result["target_concept_tokens"] = concept_tokens[1:]
        target_mask = np.zeros(self.max_length - 1, dtype=np.uint8)
        target_mask[max(target_start - 1, 0): length - 1] = 1
        result["target_mask"] = target_mask
        result["input_item_tokens"] = result["item_tokens"][:-1]
        result["target_item_tokens"] = result["item_tokens"][1:]
        result["target_action_tokens"] = result["action_tokens"][1:]
        result["target_correctness"] = result["correctness"][1:]
        return result


class CandidateCatalogBuilder:
    def __init__(
        self,
        dataset: str,
        support: pd.DataFrame,
        vocabularies: dict[str, CategoricalVocabulary],
        graph_root: Path,
    ):
        self.dataset = dataset
        self.support = support.copy()
        self.vocabularies = vocabularies
        self.graph_root = Path(graph_root)

    def build(self) -> pd.DataFrame:
        frame = self.support.copy()
        frame["item_token"] = [self.vocabularies["item_id"].encode_one(value) for value in frame["item_id"]]
        frame["module_token"] = [self.vocabularies["module_id"].encode_one(value) for value in frame["module_id"]]
        if frame.empty:
            frame["graph_node_id"] = pd.Series(dtype="string")
            frame["concept_ids_json"] = pd.Series(dtype="string")
            frame["prerequisite_ids_json"] = pd.Series(dtype="string")
            return frame
        if self.dataset == "ednet":
            skill_map, prerequisite_map = self._ednet_skill_maps()
            frame["graph_node_id"] = frame.apply(lambda row: self._ednet_node(row.item_id, row.item_type), axis=1)
            frame["concept_ids_json"] = frame["graph_node_id"].map(lambda node: json.dumps(sorted(skill_map.get(node, ()))))
            frame["prerequisite_ids_json"] = frame["concept_ids_json"].map(
                lambda text: json.dumps(sorted({pre for skill in json.loads(text) for pre in prerequisite_map.get(skill, ())}))
            )
        else:
            frame["graph_node_id"] = frame.apply(self._oulad_node, axis=1)
            frame["concept_ids_json"] = "[]"
            frame["prerequisite_ids_json"] = "[]"
        return frame.sort_values(["module_id", "item_type", "item_id"]).reset_index(drop=True)

    def _ednet_skill_maps(self) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
        explicit = pd.read_csv(self.graph_root / "ednet" / "edges_explicit.csv.gz")
        dag = pd.read_csv(self.graph_root / "ednet" / "edges_prerequisite_dag.csv.gz")
        skill_map: dict[str, set[str]] = defaultdict(set)
        for source, target, edge_type in explicit[["source_id", "target_id", "edge_type"]].itertuples(index=False):
            if edge_type in ("tests", "teaches"):
                skill_map[str(source)].add(str(target))
        explanation_questions: dict[str, set[str]] = defaultdict(set)
        for source, target, edge_type in explicit[["source_id", "target_id", "edge_type"]].itertuples(index=False):
            if edge_type == "explained_by":
                explanation_questions[str(target)].add(str(source))
        for explanation, questions in explanation_questions.items():
            skill_map[explanation].update(skill for question in questions for skill in skill_map.get(question, ()))
        prerequisites: dict[str, set[str]] = defaultdict(set)
        for source, target in dag[["source_id", "target_id"]].itertuples(index=False):
            prerequisites[str(target)].add(str(source))
        return skill_map, prerequisites

    @staticmethod
    def _ednet_node(item_id: str, item_type: str) -> str:
        raw = str(item_id).removeprefix("ednet:")
        return f"ednet:{item_type}:{raw}"

    @staticmethod
    def _oulad_node(row: pd.Series) -> str:
        if row.item_type == "assessment":
            return str(row.item_id)
        raw = str(row.item_id).removeprefix("oulad:vle:")
        module, presentation = str(row.module_id).split(":", 1)
        return f"oulad:vle_activity:{module}:{presentation}:{raw}"


class GraphAwareCandidateProvider:
    """Apply support, course-context, and optional mastery prerequisite constraints."""

    def __init__(self, catalog: pd.DataFrame, min_train_support: int = 5):
        self.catalog = catalog[catalog["train_support"] >= min_train_support].copy()

    def eligible(
        self,
        module_id: str | None = None,
        mastered_skill_ids: set[str] | None = None,
        enforce_prerequisites: bool = False,
    ) -> np.ndarray:
        candidates = self.catalog
        if module_id is not None and module_id != "":
            candidates = candidates[candidates["module_id"] == module_id]
        if enforce_prerequisites:
            mastered = mastered_skill_ids or set()
            keep = candidates["prerequisite_ids_json"].map(lambda text: set(json.loads(text)).issubset(mastered))
            candidates = candidates[keep]
        return candidates["item_token"].drop_duplicates().to_numpy(dtype=np.int32)


class Phase5SequencePipeline:
    def __init__(self, config: dict[str, object]):
        self.config = config
        self.output_root = Path(str(config["output_root"]))

    @classmethod
    def from_json(cls, path: Path) -> "Phase5SequencePipeline":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def run(self) -> dict[str, object]:
        self.output_root.mkdir(parents=True, exist_ok=True)
        window_config = WindowConfig(**dict(self.config["windowing"]))
        manifest: dict[str, object] = {
            "status": "running",
            "methodology": {
                "identifier_vocabulary": "official content catalogs plus training-observed non-catalog identifiers",
                "behavioural_vocabulary_and_statistics": "training split only",
                "support_fit": "training split only",
                "sequence_context": "chronological earlier events; targets partitioned by target-event split",
                "windowing": window_config.__dict__,
                "storage": "packed per-event arrays with lazy fixed-length windows",
            },
            "datasets": {},
        }
        for dataset in ("ednet", "oulad"):
            input_root = Path(str(self.config["preprocessing_root"])) / dataset
            dataset_output = self.output_root / dataset
            packed_output = dataset_output / "packed"
            dataset_output.mkdir(exist_ok=True)
            official_seeds = self._official_vocabulary_seeds(dataset)
            fitter = TrainVocabularyFitter(
                dataset,
                input_root / "train",
                int(self.config.get("chunksize", 250_000)),
                official_seeds,
            )
            vocabularies, support, fit_stats = fitter.fit()
            vocabulary_payload = {field: vocab.to_dict() for field, vocab in vocabularies.items()}
            (dataset_output / "vocabularies.json").write_text(json.dumps(vocabulary_payload, indent=2), encoding="utf-8")
            writer = PackedSequenceWriter(dataset, input_root, packed_output, vocabularies, window_config)
            sequence_stats = writer.run()
            catalog = CandidateCatalogBuilder(
                dataset, support, vocabularies, Path(str(self.config["graph_root"]))
            ).build()
            with gzip.open(dataset_output / "candidate_catalog.csv.gz", "wt", encoding="utf-8", newline="") as handle:
                catalog.to_csv(handle, index=False)
            unknown_stats = self._unknown_statistics(packed_output)
            manifest["datasets"][dataset] = {
                "vocabulary_fit": fit_stats,
                "sequences": sequence_stats,
                "candidate_count": int(len(catalog)),
                "candidates_meeting_default_support": int((catalog["train_support"] >= int(self.config["min_train_support"])).sum()),
                "unknown_item_rates": unknown_stats,
            }
            (self.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        manifest["status"] = "complete"
        (self.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    def _official_vocabulary_seeds(self, dataset: str) -> dict[str, set[str]]:
        """Seed stable identifiers from official catalogs, never from held-out interactions."""
        seeds: dict[str, set[str]] = defaultdict(set)
        if dataset == "ednet":
            root = Path(str(self.config["ednet_metadata_dir"]))
            questions = pd.read_csv(root / "questions.csv", dtype=str)
            lectures = pd.read_csv(root / "lectures.csv", dtype=str)
            payments = pd.read_csv(root / "payments.csv", dtype=str)
            coupons = pd.read_csv(root / "coupons.csv", dtype=str)
            seeds["item_id"].update("ednet:" + value for value in questions["question_id"].dropna())
            seeds["item_id"].update("ednet:" + value for value in questions["bundle_id"].dropna())
            seeds["item_id"].update("ednet:" + value for value in questions["explanation_id"].dropna())
            seeds["item_id"].update("ednet:" + value for value in lectures["lecture_id"].dropna())
            seeds["item_id"].update("ednet:" + value for value in payments["payment_item_id"].dropna())
            seeds["item_id"].update("ednet:" + value for value in coupons["coupon_id"].dropna())
            for value in pd.concat([questions["tags"], lectures["tags"]]).dropna().unique():
                seeds["concept_ids"].update(_concepts(value))
            seeds["item_type"].update(["question", "bundle", "explanation", "lecture", "payment", "coupon"])
        elif dataset == "oulad":
            root = Path(str(self.config["oulad_raw_dir"]))
            courses = pd.read_csv(root / "courses.csv", dtype=str)
            assessments = pd.read_csv(root / "assessments.csv", dtype=str)
            vle = pd.read_csv(root / "vle.csv", dtype=str)
            seeds["item_id"].update("oulad:assessment:" + value for value in assessments["id_assessment"].dropna())
            seeds["item_id"].update("oulad:vle:" + value for value in vle["id_site"].dropna())
            module_ids = courses["code_module"] + ":" + courses["code_presentation"]
            seeds["item_id"].update("oulad:module:" + value for value in module_ids.dropna())
            seeds["module_id"].update(module_ids.dropna())
            seeds["concept_ids"].update("module:" + value for value in courses["code_module"].dropna().unique())
            seeds["source"].update(vle["activity_type"].dropna())
            seeds["source"].update(assessments["assessment_type"].dropna())
            seeds["item_type"].update(["assessment", "vle_activity", "module"])
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
        return seeds

    @staticmethod
    def _unknown_statistics(packed_output: Path) -> dict[str, float]:
        totals = Counter()
        unknown = Counter()
        for path in packed_output.glob("part-*.npz"):
            with np.load(path, allow_pickle=False) as arrays:
                for split_id, split_name in SPLIT_NAMES.items():
                    mask = arrays["split_ids"] == split_id
                    totals[split_name] += int(mask.sum())
                    unknown[split_name] += int(np.sum(mask & (arrays["item_tokens"] == UNK_TOKEN)))
        return {split: (unknown[split] / totals[split] if totals[split] else 0.0) for split in SPLIT_IDS}

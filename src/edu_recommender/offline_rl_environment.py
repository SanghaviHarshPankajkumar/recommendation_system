from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Protocol, Sequence

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from gymnasium import spaces

from .student_state_model import (
    GraphTensorBuilder,
    KnowledgeAwareStudentStateModel,
    StudentStateModelConfig,
)


SPLIT_IDS = {"train": 0, "validation": 1, "test": 2}


class OfflineEnvironmentError(RuntimeError):
    """Base exception for invalid offline-environment operations."""


class CounterfactualActionError(OfflineEnvironmentError):
    """Raised when replay is asked to fabricate an unobserved transition."""


@dataclass(frozen=True)
class RewardWeights:
    mastery_progression: float = 1.0
    learning_opportunity: float = 0.10
    correctness: float = 0.05
    score: float = 0.0
    engagement: float = 0.05
    time_cost: float = 0.05
    prerequisite_violation: float = 0.25
    repetition: float = 0.10
    dropout: float = 1.0


@dataclass(frozen=True)
class EnvironmentSettings:
    dataset: str
    split: str = "train"
    state_dim: int = 64
    max_history: int = 127
    max_concepts_per_event: int = 9
    min_train_support: int = 5
    mastery_threshold: float = 0.7
    enforce_prerequisites: bool = False
    avoid_immediate_repeat: bool = False
    max_episode_steps: int = 128
    preserve_trajectory_continuity: bool = True
    ednet_session_gap_hours: float = 8.0
    reward_clip: float = 1.0
    mastery_progression_scale: float = 10.0
    reward_weights: RewardWeights = field(default_factory=RewardWeights)

    def __post_init__(self) -> None:
        if self.dataset not in {"ednet", "oulad"}:
            raise ValueError("dataset must be 'ednet' or 'oulad'")
        if self.split not in SPLIT_IDS:
            raise ValueError(f"split must be one of {sorted(SPLIT_IDS)}")
        if self.state_dim <= 0 or self.max_history <= 0 or self.max_episode_steps <= 0:
            raise ValueError("state_dim, max_history, and max_episode_steps must be positive")
        if not 0 <= self.mastery_threshold <= 1:
            raise ValueError("mastery_threshold must be in [0, 1]")
        if self.reward_clip <= 0:
            raise ValueError("reward_clip must be positive")
        if self.mastery_progression_scale <= 0:
            raise ValueError("mastery_progression_scale must be positive")


@dataclass(frozen=True)
class EncodedObservation:
    student_state: np.ndarray
    mastery: np.ndarray
    recent_features: np.ndarray
    module: int
    eligible_actions: np.ndarray


@dataclass(frozen=True)
class OfflineStep:
    observation: EncodedObservation
    logged_action: int
    reward: float
    next_observation: EncodedObservation
    terminated: bool
    truncated: bool
    info: Mapping[str, Any]


@dataclass(frozen=True)
class OfflineEpisode:
    episode_id: str
    steps: tuple[OfflineStep, ...]

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError("OfflineEpisode must contain at least one step")


class PackedStateEncoder(Protocol):
    state_dim: int
    mastery_dim: int
    name: str

    def encode(
        self,
        arrays: Mapping[str, np.ndarray],
        start: int,
        end: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...

    def encode_many(
        self,
        arrays: Mapping[str, np.ndarray],
        ranges: Sequence[tuple[int, int]],
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]: ...


def _ordered_vocabulary(path: Path, field: str) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    mapping = payload[field]
    return [str(key) for key, _ in sorted(mapping.items(), key=lambda pair: pair[1])]


class ProvisionalStateEncoder:
    """Deterministic, future-safe encoder for environment development only.

    It must be replaced by a validation-selected Phase 7 checkpoint before
    behavior-policy or offline-policy training.
    """

    name = "provisional_history_statistics"

    def __init__(self, state_dim: int, mastery_dim: int, max_history: int = 127):
        self.state_dim = int(state_dim)
        self.mastery_dim = int(mastery_dim)
        self.max_history = int(max_history)

    def encode(
        self,
        arrays: Mapping[str, np.ndarray],
        start: int,
        end: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        start = max(int(start), int(end) - self.max_history)
        end = int(end)
        if end <= start:
            raise ValueError("State encoding requires at least one historical event")

        correctness = arrays["correctness"][start:end].astype(np.float32)
        valid_correctness = correctness[correctness >= 0]
        scores = arrays["scores"][start:end].astype(np.float32)
        valid_scores = scores[np.isfinite(scores)] / 100.0
        engagement = arrays["engagement_log1p"][start:end].astype(np.float32)
        elapsed = arrays["elapsed_log1p"][start:end].astype(np.float32)
        time_gaps = arrays["time_gaps"][start:end].astype(np.float32)
        relative_days = arrays["relative_days"][start:end].astype(np.float32)

        def mean_or_zero(values: np.ndarray) -> float:
            return float(values.mean()) if values.size else 0.0

        recent_correctness = valid_correctness[-20:]
        last_correctness = float(valid_correctness[-1]) if valid_correctness.size else -1.0
        latest_score = float(valid_scores[-1]) if valid_scores.size else -1.0
        latest_day_values = relative_days[np.isfinite(relative_days)]
        latest_day = float(latest_day_values[-1] / 365.0) if latest_day_values.size else 0.0
        recent = np.asarray(
            [
                last_correctness,
                mean_or_zero(recent_correctness[-5:]),
                mean_or_zero(recent_correctness),
                latest_score,
                float(np.tanh(mean_or_zero(engagement[-20:]) / 5.0)),
                float(np.tanh(mean_or_zero(elapsed[-20:]) / 10.0)),
                float(np.tanh(float(time_gaps[-1]) / 10.0)) if time_gaps.size else 0.0,
                float(np.clip(latest_day, -2.0, 2.0)),
            ],
            dtype=np.float32,
        )

        success = np.ones(self.mastery_dim, dtype=np.float32)
        failure = np.ones(self.mastery_dim, dtype=np.float32)
        concept_offsets = arrays["concept_offsets"]
        concept_values = arrays["concept_values"]
        for event_index in range(start, end):
            event_correctness = int(arrays["correctness"][event_index])
            event_score = float(arrays["scores"][event_index])
            if event_correctness >= 0:
                evidence = float(event_correctness)
            elif np.isfinite(event_score):
                evidence = float(np.clip(event_score / 100.0, 0.0, 1.0))
            else:
                continue
            concept_start = int(concept_offsets[event_index])
            concept_end = int(concept_offsets[event_index + 1])
            for token in np.unique(concept_values[concept_start:concept_end]):
                token = int(token)
                if 2 <= token < self.mastery_dim:
                    success[token] += evidence
                    failure[token] += 1.0 - evidence
        mastery = success / (success + failure)
        if self.mastery_dim >= 2:
            mastery[:2] = 0.0

        valid_mastery = mastery[2:] if self.mastery_dim > 2 else mastery
        mastery_summary = np.asarray(
            [
                mean_or_zero(valid_mastery),
                float(valid_mastery.min()) if valid_mastery.size else 0.0,
                float(valid_mastery.max()) if valid_mastery.size else 0.0,
                float(valid_mastery.std()) if valid_mastery.size else 0.0,
                float(np.mean(valid_mastery >= 0.7)) if valid_mastery.size else 0.0,
            ],
            dtype=np.float32,
        )
        history_features = np.asarray(
            [
                min((end - start) / self.max_history, 1.0),
                mean_or_zero(valid_correctness),
                mean_or_zero(valid_scores),
                float(np.tanh(mean_or_zero(engagement) / 5.0)),
                float(np.tanh(mean_or_zero(elapsed) / 10.0)),
                float(np.tanh(mean_or_zero(time_gaps) / 10.0)),
                float(np.mean(arrays["final_response"][start:end])) if end > start else 0.0,
            ],
            dtype=np.float32,
        )
        feature_bank = np.concatenate([recent, mastery_summary, history_features])
        student_state = np.zeros(self.state_dim, dtype=np.float32)
        student_state[: min(self.state_dim, feature_bank.size)] = feature_bank[: self.state_dim]
        return student_state, mastery.astype(np.float32), recent

    def encode_many(
        self,
        arrays: Mapping[str, np.ndarray],
        ranges: Sequence[tuple[int, int]],
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        return [self.encode(arrays, start, end) for start, end in ranges]


class TorchStudentStateEncoder:
    """CPU-compatible adapter for a Phase 7 student-state checkpoint."""

    name = "phase7_student_state_checkpoint"

    def __init__(
        self,
        dataset: str,
        checkpoint_path: Path,
        graph_root: Path,
        sequence_root: Path,
        max_history: int = 127,
        max_concepts_per_event: int = 9,
        encode_batch_size: int = 128,
        device: str = "cpu",
    ):
        self.dataset = dataset
        self.device = torch.device(device)
        self.max_history = int(max_history)
        self.max_concepts_per_event = int(max_concepts_per_event)
        self.encode_batch_size = int(encode_batch_size)
        if self.encode_batch_size <= 0:
            raise ValueError("encode_batch_size must be positive")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        config = StudentStateModelConfig(**checkpoint["model_config"])
        if self.max_history > config.max_sequence_length:
            raise ValueError(
                f"max_history={self.max_history} exceeds checkpoint maximum "
                f"sequence length {config.max_sequence_length}"
            )
        self.state_dim = config.state_dim
        self.mastery_dim = config.concept_vocab_size
        self.graph = GraphTensorBuilder(dataset, Path(graph_root), Path(sequence_root)).build().to(self.device)
        self.model = KnowledgeAwareStudentStateModel(config, self.graph).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.eval()

    def encode(
        self,
        arrays: Mapping[str, np.ndarray],
        start: int,
        end: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        start = max(int(start), int(end) - self.max_history)
        end = int(end)
        length = end - start
        if length <= 0:
            raise ValueError("State encoding requires at least one historical event")
        return self.encode_many(arrays, [(start, end)])[0]

    def encode_many(
        self,
        arrays: Mapping[str, np.ndarray],
        ranges: Sequence[tuple[int, int]],
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        normalized = [
            (max(int(start), int(end) - self.max_history), int(end))
            for start, end in ranges
        ]
        if any(end <= start for start, end in normalized):
            raise ValueError("State encoding requires at least one historical event")
        provisional = ProvisionalStateEncoder(self.state_dim, self.mastery_dim, self.max_history)
        results: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for offset in range(0, len(normalized), self.encode_batch_size):
            chunk = normalized[offset : offset + self.encode_batch_size]
            batch, lengths = self._batch_many(arrays, chunk)
            with torch.inference_mode():
                outputs = self.model(batch, self.graph)
            states = outputs["student_states"]
            mastery_probabilities = outputs["mastery_probabilities"]
            for batch_index, ((start, end), length) in enumerate(zip(chunk, lengths)):
                state = states[batch_index, length - 1].detach().cpu().numpy().astype(np.float32)
                mastery = (
                    mastery_probabilities[batch_index, length - 1]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                _, _, recent = provisional.encode(arrays, start, end)
                results.append((state, mastery, recent))
        return results

    def _batch(self, arrays: Mapping[str, np.ndarray], start: int, end: int) -> dict[str, torch.Tensor]:
        batch, _ = self._batch_many(arrays, [(start, end)])
        return batch

    def _batch_many(
        self,
        arrays: Mapping[str, np.ndarray],
        ranges: Sequence[tuple[int, int]],
    ) -> tuple[dict[str, torch.Tensor], list[int]]:
        lengths = [end - start for start, end in ranges]
        max_length = max(lengths)
        full_length = max_length + 1
        batch: dict[str, torch.Tensor] = {}
        integer_fields = {"item_tokens", "action_tokens", "item_type_tokens", "module_tokens", "source_tokens"}
        for field in (
            "item_tokens", "action_tokens", "item_type_tokens", "module_tokens", "source_tokens",
            "time_gaps", "elapsed_log1p", "engagement_log1p", "scores", "relative_days", "correctness",
        ):
            pad_value = -1 if field == "correctness" else 0
            padded = np.full((len(ranges), full_length), pad_value, dtype=arrays[field].dtype)
            for row, (start, end) in enumerate(ranges):
                padded[row, : end - start] = arrays[field][start:end]
            tensor = torch.from_numpy(padded).to(self.device)
            batch[field] = tensor.long() if field in integer_fields else tensor
        concepts = np.zeros(
            (len(ranges), max_length, self.max_concepts_per_event), dtype=np.int64
        )
        attention = np.zeros((len(ranges), max_length), dtype=np.bool_)
        for row, (start, end) in enumerate(ranges):
            attention[row, : end - start] = True
            for local, event_index in enumerate(range(start, end)):
                concept_start = int(arrays["concept_offsets"][event_index])
                concept_end = int(arrays["concept_offsets"][event_index + 1])
                values = arrays["concept_values"][concept_start:concept_end][
                    : self.max_concepts_per_event
                ]
                concepts[row, local, : len(values)] = values
        batch["input_concept_tokens"] = torch.from_numpy(concepts).to(self.device)
        batch["input_item_tokens"] = batch["item_tokens"][:, :max_length]
        batch["input_attention_mask"] = torch.from_numpy(attention).to(self.device)
        return batch, lengths


class ActionCatalog:
    def __init__(
        self,
        dataset: str,
        frame: pd.DataFrame,
        concept_vocabulary: Sequence[str],
    ):
        required = {
            "item_id", "item_type", "module_id", "train_support", "item_token", "module_token",
            "prerequisite_ids_json",
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Candidate catalog is missing columns: {sorted(missing)}")
        if frame["item_token"].duplicated().any():
            raise ValueError("Supported candidate item tokens must be unique")
        self.dataset = dataset
        self.frame = frame.reset_index(drop=True).copy()
        self.frame["action_index"] = np.arange(len(self.frame), dtype=np.int32)
        self.action_count = int(len(self.frame))
        self.item_token_to_action = {
            int(row.item_token): int(row.action_index) for row in self.frame.itertuples(index=False)
        }
        self.action_to_item_token = self.frame["item_token"].to_numpy(dtype=np.int32)
        self.action_to_item_id = self.frame["item_id"].astype(str).to_numpy()
        self.action_to_item_type = self.frame["item_type"].astype(str).to_numpy()
        self.action_to_module = self.frame["module_token"].to_numpy(dtype=np.int32)
        concept_to_token = {value: index for index, value in enumerate(concept_vocabulary)}
        prerequisites: list[np.ndarray] = []
        for text in self.frame["prerequisite_ids_json"].fillna("[]"):
            tokens = []
            for value in json.loads(str(text)):
                raw = str(value).removeprefix("ednet:skill:")
                if raw in concept_to_token:
                    tokens.append(concept_to_token[raw])
            prerequisites.append(np.asarray(sorted(set(tokens)), dtype=np.int32))
        self.prerequisite_tokens = tuple(prerequisites)

    @classmethod
    def from_phase5(
        cls,
        dataset: str,
        sequence_root: Path,
        min_train_support: int,
    ) -> "ActionCatalog":
        dataset_root = Path(sequence_root) / dataset
        frame = pd.read_csv(dataset_root / "candidate_catalog.csv.gz")
        frame = frame[frame["train_support"] >= int(min_train_support)].copy()
        concept_vocabulary = _ordered_vocabulary(dataset_root / "vocabularies.json", "concept_ids")
        return cls(dataset, frame, concept_vocabulary)

    def eligible_actions(
        self,
        module_token: int,
        mastery: np.ndarray,
        mastery_threshold: float,
        enforce_prerequisites: bool,
        previous_item_token: int | None,
        avoid_immediate_repeat: bool,
    ) -> np.ndarray:
        actions = np.arange(self.action_count, dtype=np.int32)
        if self.dataset == "oulad":
            actions = actions[self.action_to_module == int(module_token)]
        if enforce_prerequisites:
            actions = np.asarray(
                [
                    action
                    for action in actions
                    if all(mastery[token] >= mastery_threshold for token in self.prerequisite_tokens[int(action)])
                ],
                dtype=np.int32,
            )
        if avoid_immediate_repeat and previous_item_token is not None:
            actions = actions[self.action_to_item_token[actions] != int(previous_item_token)]
        return actions


class MasteryOrientedReward:
    def __init__(self, settings: EnvironmentSettings, action_catalog: ActionCatalog):
        self.settings = settings
        self.action_catalog = action_catalog

    def calculate(
        self,
        arrays: Mapping[str, np.ndarray],
        event_index: int,
        observation: EncodedObservation,
        next_observation: EncodedObservation,
        action: int,
        dropout: bool,
        repeated: bool,
    ) -> tuple[float, dict[str, float]]:
        weights = self.settings.reward_weights
        concept_start = int(arrays["concept_offsets"][event_index])
        concept_end = int(arrays["concept_offsets"][event_index + 1])
        concepts = np.unique(arrays["concept_values"][concept_start:concept_end]).astype(np.int64)
        concepts = concepts[(concepts >= 2) & (concepts < observation.mastery.size)]
        raw_mastery_delta = (
            float(np.mean(next_observation.mastery[concepts] - observation.mastery[concepts]))
            if concepts.size
            else 0.0
        )
        mastery_delta = float(
            np.clip(
                raw_mastery_delta * self.settings.mastery_progression_scale,
                -1.0,
                1.0,
            )
        )
        # Peak reward at mastery 0.5 encourages the policy to select material
        # in the learner's zone of proximal development instead of repeatedly
        # serving already-mastered easy items.
        learning_opportunity = (
            float(np.mean(4.0 * observation.mastery[concepts] * (1.0 - observation.mastery[concepts])))
            if concepts.size
            else 0.0
        )
        correctness_raw = int(arrays["correctness"][event_index])
        correctness = float(correctness_raw) if correctness_raw >= 0 else 0.0
        score_raw = float(arrays["scores"][event_index])
        score = float(np.clip(score_raw / 100.0, 0.0, 1.0)) if np.isfinite(score_raw) else 0.0
        engagement = float(np.tanh(float(arrays["engagement_log1p"][event_index]) / 5.0))
        time_cost = float(np.tanh(float(arrays["elapsed_log1p"][event_index]) / 10.0))
        prerequisite_tokens = self.action_catalog.prerequisite_tokens[int(action)]
        prerequisite_violation = float(
            any(observation.mastery[token] < self.settings.mastery_threshold for token in prerequisite_tokens)
        )
        components = {
            "mastery_progression": mastery_delta,
            "mastery_progression_raw": raw_mastery_delta,
            "learning_opportunity": learning_opportunity,
            "correctness": correctness,
            "score": score,
            "engagement": engagement,
            "time_cost": -time_cost,
            "prerequisite_violation": -prerequisite_violation,
            "repetition": -float(repeated),
            "dropout": -float(dropout),
        }
        reward = (
            weights.mastery_progression * components["mastery_progression"]
            + weights.learning_opportunity * components["learning_opportunity"]
            + weights.correctness * components["correctness"]
            + weights.score * components["score"]
            + weights.engagement * components["engagement"]
            + weights.time_cost * components["time_cost"]
            + weights.prerequisite_violation * components["prerequisite_violation"]
            + weights.repetition * components["repetition"]
            + weights.dropout * components["dropout"]
        )
        return float(np.clip(reward, -self.settings.reward_clip, self.settings.reward_clip)), components


class PackedEpisodeBuilder:
    """Create future-safe logged episodes from Phase 5 packed arrays."""

    def __init__(
        self,
        settings: EnvironmentSettings,
        sequence_root: Path,
        action_catalog: ActionCatalog,
        encoder: PackedStateEncoder,
    ):
        self.settings = settings
        self.sequence_root = Path(sequence_root)
        self.action_catalog = action_catalog
        self.encoder = encoder
        self.dataset_root = self.sequence_root / settings.dataset
        self.vocabulary_path = self.dataset_root / "vocabularies.json"
        self.item_types = _ordered_vocabulary(self.vocabulary_path, "item_type")
        self.actions = _ordered_vocabulary(self.vocabulary_path, "action_type")
        self.reward = MasteryOrientedReward(settings, action_catalog)
        if encoder.state_dim != settings.state_dim:
            raise ValueError("Encoder state dimension does not match environment settings")

    def build(self, max_episodes: int | None = None) -> list[OfflineEpisode]:
        episodes: list[OfflineEpisode] = []
        for partition in sorted((self.dataset_root / "packed").glob("*.npz")):
            with np.load(partition, allow_pickle=False) as loaded:
                arrays = {key: loaded[key] for key in loaded.files}
            for student_index, student_id in enumerate(arrays["student_ids"]):
                student_start = int(arrays["student_offsets"][student_index])
                student_end = int(arrays["student_offsets"][student_index + 1])
                for segment_number, (segment_start, segment_end) in enumerate(
                    self._segments(arrays, student_start, student_end)
                ):
                    split_events = [
                        event
                        for event in range(segment_start, segment_end)
                        if int(arrays["split_ids"][event]) == SPLIT_IDS[self.settings.split]
                    ]
                    if not split_events:
                        continue
                    split_end = split_events[-1] + 1
                    candidate_events = [
                        event
                        for event in range(max(segment_start + 1, student_start + 1), segment_end)
                        if int(arrays["split_ids"][event]) == SPLIT_IDS[self.settings.split]
                        and self._candidate_action(arrays, event) is not None
                        and self._decision_index(arrays, event, segment_start) > segment_start
                    ]
                    encoding_cache = self._encoding_cache(
                        arrays, segment_start, candidate_events
                    )
                    for chunk_number, (chunk, chunk_truncated) in enumerate(
                        self._candidate_chunks(candidate_events)
                    ):
                        episode = self._episode(
                            arrays,
                            str(student_id),
                            segment_start,
                            split_end,
                            chunk,
                            segment_number,
                            chunk_number,
                            chunk_truncated,
                            encoding_cache,
                        )
                        episodes.append(episode)
                        if max_episodes is not None and len(episodes) >= max_episodes:
                            return episodes
        return episodes

    def _candidate_chunks(
        self, candidate_events: Sequence[int]
    ) -> list[tuple[list[int], bool]]:
        events = list(candidate_events)
        if not events:
            return []
        if self.settings.preserve_trajectory_continuity:
            return [(events, False)]
        result: list[tuple[list[int], bool]] = []
        for start in range(0, len(events), self.settings.max_episode_steps):
            chunk = events[start : start + self.settings.max_episode_steps]
            result.append((chunk, start + len(chunk) < len(events)))
        return result

    def _encoding_cache(
        self,
        arrays: Mapping[str, np.ndarray],
        segment_start: int,
        candidate_events: Sequence[int],
    ) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]]:
        ranges: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for event_index in candidate_events:
            decision_index = self._decision_index(arrays, event_index, segment_start)
            requested = (
                (
                    max(segment_start, decision_index - self.settings.max_history),
                    decision_index,
                ),
                (
                    max(segment_start, event_index + 1 - self.settings.max_history),
                    event_index + 1,
                ),
            )
            for value in requested:
                if value not in seen:
                    seen.add(value)
                    ranges.append(value)
        if not ranges:
            return {}
        values = self.encoder.encode_many(arrays, ranges)
        if len(values) != len(ranges):
            raise ValueError("State encoder returned the wrong number of batched encodings")
        return dict(zip(ranges, values))

    def _segments(
        self,
        arrays: Mapping[str, np.ndarray],
        start: int,
        end: int,
    ) -> list[tuple[int, int]]:
        if self.settings.dataset != "ednet" or end - start <= 1:
            return [(start, end)]
        timestamps = arrays["timestamps"][start:end].astype(np.int64)
        threshold = int(self.settings.ednet_session_gap_hours * 60 * 60 * 1000)
        breaks = np.flatnonzero(np.diff(timestamps) > threshold) + start + 1
        boundaries = [start, *breaks.tolist(), end]
        return [
            (int(left), int(right))
            for left, right in zip(boundaries[:-1], boundaries[1:])
            if right - left >= 2
        ]

    def _candidate_action(self, arrays: Mapping[str, np.ndarray], event: int) -> int | None:
        item_token = int(arrays["item_tokens"][event])
        action = self.action_catalog.item_token_to_action.get(item_token)
        if action is None:
            return None
        item_type = self.item_types[int(arrays["item_type_tokens"][event])]
        action_type = self.actions[int(arrays["action_tokens"][event])]
        if self.settings.dataset == "ednet":
            if item_type == "question":
                return action if int(arrays["final_response"][event]) == 1 else None
            return action if item_type in {"lecture", "explanation"} and action_type == "enter" else None
        if item_type == "assessment" and action_type == "assessment_banked":
            return None
        return action if item_type in {"assessment", "vle_activity"} else None

    def _decision_index(
        self,
        arrays: Mapping[str, np.ndarray],
        event_index: int,
        segment_start: int,
    ) -> int:
        """Return the last history index available when the logged item was chosen.

        EdNet records one or more response events for a question before its final
        response. The recommendation precedes that contiguous response block, so
        those responses must not appear in the action's observation.
        """
        if self.settings.dataset != "ednet":
            return int(event_index)
        item_type = self.item_types[int(arrays["item_type_tokens"][event_index])]
        if item_type != "question":
            return int(event_index)
        item_token = int(arrays["item_tokens"][event_index])
        decision_index = int(event_index)
        while (
            decision_index > segment_start
            and int(arrays["item_tokens"][decision_index - 1]) == item_token
            and self.item_types[int(arrays["item_type_tokens"][decision_index - 1])] == "question"
        ):
            decision_index -= 1
        return decision_index

    def _episode(
        self,
        arrays: Mapping[str, np.ndarray],
        student_id: str,
        segment_start: int,
        split_end: int,
        candidate_events: Sequence[int],
        segment_number: int,
        chunk_number: int,
        chunk_truncated: bool,
        encoding_cache: Mapping[
            tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]
        ],
    ) -> OfflineEpisode:
        steps: list[OfflineStep] = []
        previous_candidate_item: int | None = None
        withdrawal_token = self.actions.index("withdrawal") if "withdrawal" in self.actions else -1
        dropout = bool(
            withdrawal_token >= 0
            and np.any(arrays["action_tokens"][candidate_events[-1] + 1 : split_end] == withdrawal_token)
        )
        for local_index, event_index in enumerate(candidate_events):
            action = self._candidate_action(arrays, event_index)
            if action is None:
                raise AssertionError("Candidate event lost its catalog action")
            decision_index = self._decision_index(arrays, event_index, segment_start)
            history_start = max(segment_start, decision_index - self.settings.max_history)
            state, mastery, recent = encoding_cache[(history_start, decision_index)]
            next_start = max(segment_start, event_index + 1 - self.settings.max_history)
            next_state, next_mastery, next_recent = encoding_cache[
                (next_start, event_index + 1)
            ]
            module_token = int(arrays["module_tokens"][event_index])
            eligible = self.action_catalog.eligible_actions(
                module_token,
                mastery,
                self.settings.mastery_threshold,
                self.settings.enforce_prerequisites,
                previous_candidate_item,
                self.settings.avoid_immediate_repeat,
            )
            next_eligible = self.action_catalog.eligible_actions(
                module_token,
                next_mastery,
                self.settings.mastery_threshold,
                self.settings.enforce_prerequisites,
                int(arrays["item_tokens"][event_index]),
                self.settings.avoid_immediate_repeat,
            )
            observation = EncodedObservation(state, mastery, recent, module_token, eligible)
            next_observation = EncodedObservation(
                next_state, next_mastery, next_recent, module_token, next_eligible
            )
            repeated = previous_candidate_item == int(arrays["item_tokens"][event_index])
            is_last = local_index == len(candidate_events) - 1
            reward, components = self.reward.calculate(
                arrays,
                event_index,
                observation,
                next_observation,
                action,
                dropout=dropout and is_last and not chunk_truncated,
                repeated=repeated,
            )
            forced_logged_action = int(action) not in set(eligible.tolist())
            if forced_logged_action:
                eligible = np.unique(np.append(eligible, int(action))).astype(np.int32)
                observation = EncodedObservation(state, mastery, recent, module_token, eligible)
            steps.append(
                OfflineStep(
                    observation=observation,
                    logged_action=int(action),
                    reward=reward,
                    next_observation=next_observation,
                    terminated=bool(is_last and not chunk_truncated),
                    truncated=bool(is_last and chunk_truncated),
                    info={
                        "student_id": student_id,
                        "event_index": int(event_index),
                        "decision_index": int(decision_index),
                        "item_id": str(self.action_catalog.action_to_item_id[int(action)]),
                        "item_type": str(self.action_catalog.action_to_item_type[int(action)]),
                        "item_token": int(arrays["item_tokens"][event_index]),
                        "module_token": module_token,
                        "reward_components": components,
                        "dropout_terminal": bool(dropout and is_last and not chunk_truncated),
                        "logged_action_forced_eligible": forced_logged_action,
                        "encoder": self.encoder.name,
                    },
                )
            )
            previous_candidate_item = int(arrays["item_tokens"][event_index])
        episode_id = f"{student_id}:segment-{segment_number}:chunk-{chunk_number}"
        return OfflineEpisode(episode_id, tuple(steps))


class BaseOfflineEducationEnv(gym.Env[dict[str, np.ndarray | int], int]):
    """Dataset/replay adapter, not an interactive student simulator.

    ``step`` accepts only the logged action. Policy training must consume
    ``iter_transitions`` (or the converted offline dataset); online Gym rollouts
    would require a separately validated counterfactual student simulator.
    """

    metadata = {"render_modes": []}
    offline_replay_only = True

    def __init__(
        self,
        episodes: Sequence[OfflineEpisode],
        action_catalog: ActionCatalog,
        state_dim: int,
        mastery_dim: int,
        module_count: int,
    ):
        super().__init__()
        if not episodes:
            raise ValueError("At least one offline episode is required")
        self.episodes = tuple(episodes)
        self.action_catalog = action_catalog
        self.action_space = spaces.Discrete(action_catalog.action_count)
        self.observation_space = spaces.Dict(
            {
                "student_state": spaces.Box(-np.inf, np.inf, shape=(state_dim,), dtype=np.float32),
                "mastery": spaces.Box(0.0, 1.0, shape=(mastery_dim,), dtype=np.float32),
                "recent_features": spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32),
                "module": spaces.Discrete(max(int(module_count), 1)),
                "action_mask": spaces.MultiBinary(action_catalog.action_count),
            }
        )
        self._episode_index = 0
        self._step_index = 0
        self._active_episode: OfflineEpisode | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray | int], dict[str, Any]]:
        super().reset(seed=seed)
        if options and "episode_index" in options:
            episode_index = int(options["episode_index"])
        elif options and options.get("random_episode"):
            episode_index = int(self.np_random.integers(len(self.episodes)))
        else:
            episode_index = self._episode_index % len(self.episodes)
            self._episode_index += 1
        if not 0 <= episode_index < len(self.episodes):
            raise IndexError("episode_index is outside the loaded offline dataset")
        self._active_episode = self.episodes[episode_index]
        self._step_index = 0
        step = self._active_episode.steps[0]
        return self._materialize(step.observation), {
            "episode_id": self._active_episode.episode_id,
            "logged_action": step.logged_action,
            **dict(step.info),
        }

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, np.ndarray | int], float, bool, bool, dict[str, Any]]:
        if self._active_episode is None:
            raise OfflineEnvironmentError("reset() must be called before step()")
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action} is outside the action space")
        step = self._active_episode.steps[self._step_index]
        if int(action) != step.logged_action:
            proposed = str(self.action_catalog.action_to_item_id[int(action)])
            logged = str(self.action_catalog.action_to_item_id[step.logged_action])
            raise CounterfactualActionError(
                f"Offline replay observed {logged!r}, not proposed action {proposed!r}; "
                "no student simulator exists to generate a counterfactual next state"
            )
        self._step_index += 1
        info = {
            "episode_id": self._active_episode.episode_id,
            "logged_action": step.logged_action,
            **dict(step.info),
        }
        if not (step.terminated or step.truncated) and self._step_index < len(self._active_episode.steps):
            info["next_logged_action"] = self._active_episode.steps[self._step_index].logged_action
        return (
            self._materialize(step.next_observation),
            float(step.reward),
            bool(step.terminated),
            bool(step.truncated),
            info,
        )

    def replay_logged_step(
        self,
    ) -> tuple[dict[str, np.ndarray | int], float, bool, bool, dict[str, Any]]:
        """Advance one observed transition without accepting a policy action."""
        if self._active_episode is None:
            raise OfflineEnvironmentError("reset() must be called before replay_logged_step()")
        logged_action = self._active_episode.steps[self._step_index].logged_action
        return self.step(logged_action)

    def iter_transitions(
        self,
    ) -> Iterator[tuple[dict[str, np.ndarray | int], int, float, dict[str, np.ndarray | int], bool, bool, Mapping[str, Any]]]:
        for episode in self.episodes:
            for step in episode.steps:
                yield (
                    self._materialize(step.observation),
                    step.logged_action,
                    step.reward,
                    self._materialize(step.next_observation),
                    step.terminated,
                    step.truncated,
                    step.info,
                )

    def validate_dataset(self) -> dict[str, Any]:
        transitions = 0
        rewards: list[float] = []
        forced = 0
        for observation, action, reward, next_observation, terminated, truncated, info in self.iter_transitions():
            if not self.observation_space.contains(observation):
                raise ValueError("Observation does not satisfy observation_space")
            if not self.observation_space.contains(next_observation):
                raise ValueError("Next observation does not satisfy observation_space")
            if not self.action_space.contains(action):
                raise ValueError("Logged action does not satisfy action_space")
            if observation["action_mask"][action] != 1:
                raise ValueError("Logged action is absent from its action mask")
            if terminated and truncated:
                raise ValueError("A transition cannot be both terminated and truncated")
            transitions += 1
            rewards.append(float(reward))
            forced += int(bool(info.get("logged_action_forced_eligible")))
        return {
            "passed": True,
            "episodes": len(self.episodes),
            "transitions": transitions,
            "action_count": int(self.action_space.n),
            "reward_min": min(rewards),
            "reward_max": max(rewards),
            "reward_mean": float(np.mean(rewards)),
            "logged_actions_forced_eligible": forced,
        }

    def _materialize(self, observation: EncodedObservation) -> dict[str, np.ndarray | int]:
        mask = np.zeros(self.action_catalog.action_count, dtype=np.int8)
        mask[observation.eligible_actions] = 1
        return {
            "student_state": np.asarray(observation.student_state, dtype=np.float32),
            "mastery": np.asarray(observation.mastery, dtype=np.float32),
            "recent_features": np.asarray(observation.recent_features, dtype=np.float32),
            "module": int(observation.module),
            "action_mask": mask,
        }


class EdNetOfflineEnv(BaseOfflineEducationEnv):
    def __init__(self, *args: Any, **kwargs: Any):
        action_catalog = kwargs.get("action_catalog") or (args[1] if len(args) > 1 else None)
        if action_catalog is None or action_catalog.dataset != "ednet":
            raise ValueError("EdNetOfflineEnv requires an EdNet action catalog")
        super().__init__(*args, **kwargs)


class OULADOfflineEnv(BaseOfflineEducationEnv):
    def __init__(self, *args: Any, **kwargs: Any):
        action_catalog = kwargs.get("action_catalog") or (args[1] if len(args) > 1 else None)
        if action_catalog is None or action_catalog.dataset != "oulad":
            raise ValueError("OULADOfflineEnv requires an OULAD action catalog")
        super().__init__(*args, **kwargs)


def build_environment(
    settings: EnvironmentSettings,
    sequence_root: Path,
    encoder: PackedStateEncoder,
    max_episodes: int | None = None,
) -> BaseOfflineEducationEnv:
    sequence_root = Path(sequence_root)
    catalog = ActionCatalog.from_phase5(
        settings.dataset,
        sequence_root,
        settings.min_train_support,
    )
    episodes = PackedEpisodeBuilder(settings, sequence_root, catalog, encoder).build(max_episodes=max_episodes)
    module_count = len(_ordered_vocabulary(sequence_root / settings.dataset / "vocabularies.json", "module_id"))
    environment_type = EdNetOfflineEnv if settings.dataset == "ednet" else OULADOfflineEnv
    return environment_type(
        episodes=episodes,
        action_catalog=catalog,
        state_dim=settings.state_dim,
        mastery_dim=encoder.mastery_dim,
        module_count=module_count,
    )

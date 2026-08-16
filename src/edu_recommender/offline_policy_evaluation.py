from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from .d3rlpy_adapter import MaskedD3RLPYActionSelector


class EvaluationError(RuntimeError):
    """Raised when policy evaluation inputs are incomplete or unsafe."""


@dataclass(frozen=True)
class OfflineEvaluationDataset:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminals: np.ndarray
    timeouts: np.ndarray
    eligible_actions: tuple[np.ndarray, ...]
    episode_ids: np.ndarray
    action_size: int
    dataset_name: str
    provisional_states: bool

    def __post_init__(self) -> None:
        rows = int(self.observations.shape[0])
        if self.observations.ndim != 2 or self.observations.dtype != np.float32:
            raise ValueError("observations must be a two-dimensional float32 array")
        for values in (self.actions, self.rewards, self.terminals, self.timeouts, self.episode_ids):
            if len(values) != rows:
                raise ValueError("Evaluation arrays must have equal row counts")
        if len(self.eligible_actions) != rows:
            raise ValueError("Sparse eligibility lists must align with observations")
        if np.any(self.actions < 0) or np.any(self.actions >= self.action_size):
            raise ValueError("A logged action is outside the action catalog")
        if np.any((self.terminals + self.timeouts) > 1):
            raise ValueError("A row cannot be terminal and timeout simultaneously")
        for row, eligible in enumerate(self.eligible_actions):
            if eligible.size == 0:
                raise ValueError(f"Row {row} has no eligible actions")
            if np.any(eligible < 0) or np.any(eligible >= self.action_size):
                raise ValueError(f"Row {row} contains an out-of-catalog eligible action")
            if int(self.actions[row]) not in eligible:
                raise ValueError(f"Logged action is not eligible at row {row}")

    @classmethod
    def load(cls, dataset_path: Path, metadata_path: Path) -> "OfflineEvaluationDataset":
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        with np.load(dataset_path, allow_pickle=False) as loaded:
            required = {
                "observations",
                "actions",
                "rewards",
                "terminals",
                "timeouts",
                "eligibility_offsets",
                "eligibility_values",
                "episode_ids",
            }
            missing = required.difference(loaded.files)
            if missing:
                raise EvaluationError(
                    f"Dataset is missing evaluation metadata {sorted(missing)}; "
                    "rerun scripts/prepare_phase9_d3rlpy.py"
                )
            offsets = loaded["eligibility_offsets"].astype(np.int64)
            values = loaded["eligibility_values"].astype(np.int32)
            rows = int(loaded["observations"].shape[0])
            if offsets.shape != (rows + 1,) or offsets[0] != 0 or offsets[-1] != len(values):
                raise ValueError("Invalid sparse eligibility offsets")
            eligible = tuple(values[offsets[index] : offsets[index + 1]] for index in range(rows))
            return cls(
                observations=loaded["observations"].astype(np.float32),
                actions=loaded["actions"].astype(np.int64),
                rewards=loaded["rewards"].astype(np.float32),
                terminals=loaded["terminals"].astype(np.float32),
                timeouts=loaded["timeouts"].astype(np.float32),
                eligible_actions=eligible,
                episode_ids=loaded["episode_ids"].astype(np.str_),
                action_size=int(metadata["action_size"]),
                dataset_name=str(metadata["dataset"]),
                provisional_states=bool(metadata.get("provisional_states", False)),
            )


class PolicyScorer(Protocol):
    name: str

    def score_eligible(
        self, observation: np.ndarray, eligible_actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def raw_action(self, observation: np.ndarray) -> int: ...


class PopularityPolicyScorer:
    """Deterministic logged-frequency baseline for evaluation-pipeline checks."""

    name = "popularity"

    def __init__(self, logged_actions: np.ndarray, action_size: int):
        self.counts = np.bincount(logged_actions.astype(np.int64), minlength=action_size).astype(
            np.float64
        )

    def score_eligible(
        self, observation: np.ndarray, eligible_actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        del observation
        eligible = np.unique(np.asarray(eligible_actions, dtype=np.int64))
        return eligible, self.counts[eligible]

    def raw_action(self, observation: np.ndarray) -> int:
        del observation
        return int(np.argmax(self.counts))


class D3RLPYPolicyScorer:
    def __init__(self, algorithm: Any, algorithm_name: str):
        self.algorithm = algorithm
        self.name = algorithm_name
        self.selector = MaskedD3RLPYActionSelector(algorithm, algorithm_name)

    def score_eligible(
        self, observation: np.ndarray, eligible_actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.selector.score_eligible(observation, eligible_actions)

    def raw_action(self, observation: np.ndarray) -> int:
        return int(self.algorithm.predict(observation[None, :])[0])


def _rank(scores: np.ndarray, target_index: int) -> float:
    target_score = float(scores[target_index])
    return (
        1.0
        + float(np.sum(scores > target_score))
        + 0.5 * float(np.sum(scores == target_score) - 1)
    )


def _cluster_bootstrap(
    values: np.ndarray,
    episode_ids: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    point = float(np.mean(values))
    if replicates <= 0:
        return {"estimate": point, "ci95_low": point, "ci95_high": point}
    episodes = np.unique(episode_ids)
    row_groups = [np.flatnonzero(episode_ids == episode) for episode in episodes]
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled_groups = rng.integers(0, len(row_groups), size=len(row_groups))
        rows = np.concatenate([row_groups[index] for index in sampled_groups])
        estimates[replicate] = float(np.mean(values[rows]))
    low, high = np.percentile(estimates, [2.5, 97.5])
    return {"estimate": point, "ci95_low": float(low), "ci95_high": float(high)}


def _episode_returns(rewards: np.ndarray, episode_ids: np.ndarray) -> np.ndarray:
    return np.asarray(
        [float(rewards[episode_ids == episode].sum()) for episode in np.unique(episode_ids)],
        dtype=np.float64,
    )


class OfflinePolicyEvaluator:
    def __init__(
        self,
        ks: Sequence[int] = (5, 10, 20),
        bootstrap_replicates: int = 1000,
        seed: int = 42,
    ):
        if not ks or any(int(k) <= 0 for k in ks):
            raise ValueError("ks must contain positive cutoffs")
        if bootstrap_replicates < 0:
            raise ValueError("bootstrap_replicates cannot be negative")
        self.ks = tuple(sorted(set(int(k) for k in ks)))
        self.bootstrap_replicates = int(bootstrap_replicates)
        self.seed = int(seed)

    def evaluate(
        self,
        dataset: OfflineEvaluationDataset,
        scorer: PolicyScorer,
        max_transitions: int | None = None,
    ) -> dict[str, Any]:
        rows = len(dataset.actions)
        if max_transitions is not None:
            if max_transitions <= 0:
                raise ValueError("max_transitions must be positive")
            rows = min(rows, int(max_transitions))
        ranks = np.empty(rows, dtype=np.float64)
        logged_scores = np.empty(rows, dtype=np.float64)
        selected_scores = np.empty(rows, dtype=np.float64)
        selected_actions = np.empty(rows, dtype=np.int64)
        raw_eligible = np.empty(rows, dtype=np.float64)

        for row in range(rows):
            observation = dataset.observations[row]
            eligible = dataset.eligible_actions[row]
            actions, scores = scorer.score_eligible(observation, eligible)
            if actions.shape != scores.shape or actions.size == 0:
                raise EvaluationError("Policy scorer returned invalid action-score arrays")
            matches = np.flatnonzero(actions == int(dataset.actions[row]))
            if matches.size != 1:
                raise EvaluationError("Logged action must appear exactly once in scored candidates")
            target_index = int(matches[0])
            selected_index = int(np.argmax(scores))
            ranks[row] = _rank(scores, target_index)
            logged_scores[row] = float(scores[target_index])
            selected_scores[row] = float(scores[selected_index])
            selected_actions[row] = int(actions[selected_index])
            raw_eligible[row] = float(int(scorer.raw_action(observation)) in eligible)

        episode_ids = dataset.episode_ids[:rows]
        top1 = (selected_actions == dataset.actions[:rows]).astype(np.float64)
        reciprocal_rank = 1.0 / ranks
        metric_values: dict[str, np.ndarray] = {
            "top1_agreement": top1,
            "mrr": reciprocal_rank,
        }
        for k in self.ks:
            metric_values[f"hit_rate@{k}"] = (ranks <= k).astype(np.float64)
            metric_values[f"ndcg@{k}"] = np.where(
                ranks <= k, 1.0 / np.log2(ranks + 1.0), 0.0
            )
        estimates = {
            name: _cluster_bootstrap(
                values,
                episode_ids,
                self.bootstrap_replicates,
                self.seed + index,
            )
            for index, (name, values) in enumerate(metric_values.items())
        }
        returns = _episode_returns(dataset.rewards[:rows], episode_ids)
        unique_selected = np.unique(selected_actions)
        return {
            "status": "complete",
            "dataset": dataset.dataset_name,
            "policy": scorer.name,
            "provisional_states": dataset.provisional_states,
            "evaluated_transitions": rows,
            "evaluated_episodes": int(np.unique(episode_ids).size),
            "metrics": estimates,
            "scores": {
                "mean_logged_action_score": float(logged_scores.mean()),
                "mean_selected_action_score": float(selected_scores.mean()),
            },
            "safety": {
                "masked_eligible_action_rate": 1.0,
                "raw_eligible_action_rate": float(raw_eligible.mean()),
                "catalog_unsupported_action_rate": float(
                    np.mean((selected_actions < 0) | (selected_actions >= dataset.action_size))
                ),
            },
            "coverage": {
                "unique_recommended_actions": int(unique_selected.size),
                "catalog_action_coverage": float(unique_selected.size / dataset.action_size),
            },
            "logged_data_diagnostics": {
                "note": "These are historical-data statistics, not counterfactual policy returns.",
                "reward_mean": float(dataset.rewards[:rows].mean()),
                "reward_std": float(dataset.rewards[:rows].std()),
                "episode_return_mean": float(returns.mean()),
                "episode_return_std": float(returns.std()),
                "episode_return_min": float(returns.min()),
                "episode_return_max": float(returns.max()),
                "terminal_rows": int(dataset.terminals[:rows].sum()),
                "timeout_rows": int(dataset.timeouts[:rows].sum()),
                "distinct_logged_actions": int(np.unique(dataset.actions[:rows]).size),
            },
            "limitations": {
                "fqe_included": False,
                "importance_sampling_included": False,
                "causal_improvement_established": False,
            },
        }


from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from d3rlpy.algos import DiscreteBCConfig, DiscreteCQLConfig
from d3rlpy.dataset import MDPDataset

from .offline_rl_environment import BaseOfflineEducationEnv, EncodedObservation, OfflineEpisode


SUPPORTED_ALGORITHMS = ("discrete_bc", "discrete_cql")


@dataclass(frozen=True)
class D3RLPYDatasetBundle:
    """d3rlpy arrays plus recommendation metadata that d3rlpy does not store.

    Dynamic eligible-action lists stay sparse and outside the feature vector. A
    dense EdNet mask would waste roughly 21 KB per transition. This metadata is
    sufficient for BC selection, but stock d3rlpy CQL is deliberately rejected
    because it cannot consume the masks during training.
    """

    dataset_name: str
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminals: np.ndarray
    timeouts: np.ndarray
    eligible_actions: tuple[np.ndarray, ...]
    episode_ids: tuple[str, ...]
    action_size: int
    state_encoder_name: str
    provisional_states: bool

    def __post_init__(self) -> None:
        size = self.observations.shape[0]
        arrays = (self.actions, self.rewards, self.terminals, self.timeouts)
        if any(array.shape[0] != size for array in arrays):
            raise ValueError("All d3rlpy arrays must contain the same number of rows")
        if len(self.eligible_actions) != size or len(self.episode_ids) != size:
            raise ValueError("Recommendation metadata must align with transition rows")
        if self.observations.ndim != 2 or self.observations.dtype != np.float32:
            raise ValueError("observations must be a two-dimensional float32 array")
        if np.any(self.actions < 0) or np.any(self.actions >= self.action_size):
            raise ValueError("An action is outside the declared discrete action space")
        if np.any((self.terminals + self.timeouts) > 1):
            raise ValueError("A row cannot be both terminal and timeout")
        for index, eligible in enumerate(self.eligible_actions):
            if int(self.actions[index]) not in eligible:
                raise ValueError("Every logged action must be eligible in its observation")

    @property
    def transition_count(self) -> int:
        return int(self.observations.shape[0])

    @property
    def observation_size(self) -> int:
        return int(self.observations.shape[1])

    def to_mdp_dataset(self) -> MDPDataset:
        """Create the native fixed-demonstration dataset consumed by d3rlpy."""

        return MDPDataset(
            observations=self.observations,
            actions=self.actions,
            rewards=self.rewards,
            terminals=self.terminals,
            timeouts=self.timeouts,
            action_size=self.action_size,
        )

    def summary(self) -> dict[str, Any]:
        native = self.to_mdp_dataset()
        return {
            "dataset": self.dataset_name,
            "transitions": self.transition_count,
            "d3rlpy_transitions": int(native.transition_count),
            "episodes": len(native.episodes),
            "observation_size": self.observation_size,
            "action_size": self.action_size,
            "terminal_rows": int(self.terminals.sum()),
            "timeout_rows": int(self.timeouts.sum()),
            "reward_min": float(self.rewards.min()),
            "reward_max": float(self.rewards.max()),
            "reward_mean": float(self.rewards.mean()),
            "state_encoder": self.state_encoder_name,
            "provisional_states": self.provisional_states,
        }


class D3RLPYTransitionAdapter:
    """Convert Phase 8 episodes into fixed-size d3rlpy observations."""

    def __init__(self, module_count: int):
        if module_count <= 0:
            raise ValueError("module_count must be positive")
        self.module_count = int(module_count)

    @property
    def module_feature_size(self) -> int:
        return self.module_count

    def flatten_observation(self, observation: EncodedObservation) -> np.ndarray:
        module = int(observation.module)
        if not 0 <= module < self.module_count:
            raise ValueError(f"module token {module} is outside [0, {self.module_count})")
        module_one_hot = np.zeros(self.module_count, dtype=np.float32)
        module_one_hot[module] = 1.0
        return np.concatenate(
            (
                np.asarray(observation.student_state, dtype=np.float32),
                np.asarray(observation.mastery, dtype=np.float32),
                np.asarray(observation.recent_features, dtype=np.float32),
                module_one_hot,
            )
        ).astype(np.float32, copy=False)

    def convert(
        self,
        dataset_name: str,
        episodes: Sequence[OfflineEpisode],
        action_size: int,
    ) -> D3RLPYDatasetBundle:
        if dataset_name not in {"ednet", "oulad"}:
            raise ValueError("dataset_name must be 'ednet' or 'oulad'")
        if not episodes:
            raise ValueError("At least one episode is required")

        observations: list[np.ndarray] = []
        actions: list[int] = []
        rewards: list[float] = []
        terminals: list[float] = []
        timeouts: list[float] = []
        eligible_actions: list[np.ndarray] = []
        episode_ids: list[str] = []
        encoder_names: set[str] = set()

        for episode in episodes:
            for step in episode.steps:
                observations.append(self.flatten_observation(step.observation))
                actions.append(int(step.logged_action))
                rewards.append(float(step.reward))
                terminals.append(float(step.terminated))
                timeouts.append(float(step.truncated))
                eligible_actions.append(
                    np.asarray(step.observation.eligible_actions, dtype=np.int32)
                )
                episode_ids.append(episode.episode_id)
                encoder_names.add(str(step.info.get("encoder", "unknown")))

        if len(encoder_names) != 1:
            raise ValueError(f"Expected one state encoder, found {sorted(encoder_names)}")
        encoder_name = next(iter(encoder_names))
        return D3RLPYDatasetBundle(
            dataset_name=dataset_name,
            observations=np.stack(observations).astype(np.float32, copy=False),
            actions=np.asarray(actions, dtype=np.int64),
            rewards=np.asarray(rewards, dtype=np.float32),
            terminals=np.asarray(terminals, dtype=np.float32),
            timeouts=np.asarray(timeouts, dtype=np.float32),
            eligible_actions=tuple(eligible_actions),
            episode_ids=tuple(episode_ids),
            action_size=int(action_size),
            state_encoder_name=encoder_name,
            provisional_states=encoder_name == "provisional_history_statistics",
        )

    def from_environment(self, environment: BaseOfflineEducationEnv) -> D3RLPYDatasetBundle:
        return self.convert(
            dataset_name=environment.action_catalog.dataset,
            episodes=environment.episodes,
            action_size=int(environment.action_space.n),
        )


def create_d3rlpy_algorithm(
    algorithm: str,
    *,
    batch_size: int,
    learning_rate: float,
    gamma: float,
    device: bool | int | str | None = False,
    cql_alpha: float = 1.0,
    bc_beta: float = 0.5,
    dynamic_action_masks: bool = True,
) -> Any:
    """Construct an untrained d3rlpy algorithm; this function never calls fit."""

    if algorithm == "discrete_bc":
        config = DiscreteBCConfig(
            batch_size=int(batch_size),
            learning_rate=float(learning_rate),
            gamma=float(gamma),
            beta=float(bc_beta),
        )
    elif algorithm == "discrete_cql":
        if dynamic_action_masks:
            raise ValueError(
                "Stock d3rlpy DiscreteCQL does not apply dynamic eligibility masks "
                "inside Bellman targets or the conservative loss. Use a mask-aware "
                "parametric Q(s, action_features) implementation instead."
            )
        config = DiscreteCQLConfig(
            batch_size=int(batch_size),
            learning_rate=float(learning_rate),
            gamma=float(gamma),
            alpha=float(cql_alpha),
        )
    else:
        raise ValueError(f"algorithm must be one of {SUPPORTED_ALGORITHMS}")
    return config.create(device=device)


class MaskedD3RLPYActionSelector:
    """Choose only eligible recommendations from a fitted d3rlpy policy.

    CQL actions are ranked with public ``predict_value``. Discrete BC requires
    its categorical logits; this pinned d3rlpy 2.8.1 adapter accesses the
    fitted implementation because d3rlpy has no public masked-BC prediction API.
    """

    def __init__(self, algorithm: Any, algorithm_name: str, score_batch_size: int = 2048):
        if algorithm_name not in SUPPORTED_ALGORITHMS:
            raise ValueError(f"algorithm_name must be one of {SUPPORTED_ALGORITHMS}")
        if score_batch_size <= 0:
            raise ValueError("score_batch_size must be positive")
        self.algorithm = algorithm
        self.algorithm_name = algorithm_name
        self.score_batch_size = int(score_batch_size)

    def select(self, observation: np.ndarray, eligible_actions: np.ndarray) -> int:
        eligible, scores = self.score_eligible(observation, eligible_actions)
        return int(eligible[int(np.argmax(scores))])

    def score_eligible(
        self, observation: np.ndarray, eligible_actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return eligible action ids and their BC logits or CQL Q-values."""

        observation = np.asarray(observation, dtype=np.float32)
        eligible = np.unique(np.asarray(eligible_actions, dtype=np.int64))
        if observation.ndim != 1:
            raise ValueError("observation must be a one-dimensional flattened state")
        if eligible.size == 0:
            raise ValueError("eligible_actions cannot be empty")
        if self.algorithm.impl is None:
            raise RuntimeError("The d3rlpy algorithm must be fitted or built before prediction")

        if self.algorithm_name == "discrete_cql":
            scores: list[np.ndarray] = []
            for start in range(0, eligible.size, self.score_batch_size):
                candidates = eligible[start : start + self.score_batch_size]
                batch = np.repeat(observation[None, :], len(candidates), axis=0)
                scores.append(
                    np.asarray(self.algorithm.predict_value(batch, candidates), dtype=np.float32)
                )
            return eligible, np.concatenate(scores)

        # d3rlpy 2.8.1 DiscreteBC: use the categorical policy logits so the
        # eligibility mask is applied before argmax.
        device = self.algorithm._device  # pinned-version integration point
        torch_observation = torch.as_tensor(observation[None, :], device=device)
        scaler = self.algorithm.config.observation_scaler
        if scaler is not None:
            torch_observation = scaler.transform(torch_observation)
        with torch.no_grad():
            logits = self.algorithm.impl.modules.imitator(torch_observation).logits[0]
        eligible_tensor = torch.as_tensor(eligible, dtype=torch.long, device=logits.device)
        scores = logits[eligible_tensor].detach().float().cpu().numpy()
        return eligible, scores

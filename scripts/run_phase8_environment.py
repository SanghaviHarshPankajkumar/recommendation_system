from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from edu_recommender.config import load_config, write_json
from edu_recommender.offline_rl_environment import (
    EnvironmentSettings,
    ProvisionalStateEncoder,
    RewardWeights,
    build_environment,
)


def settings_for(dataset: str, config: dict[str, object]) -> EnvironmentSettings:
    return EnvironmentSettings(
        dataset=dataset,
        split=str(config["split"]),
        state_dim=int(config["state_dim"]),
        max_history=int(config["max_history"]),
        max_concepts_per_event=int(config["max_concepts_per_event"]),
        min_train_support=int(config["min_train_support"]),
        mastery_threshold=float(config["mastery_threshold"]),
        enforce_prerequisites=bool(config["enforce_prerequisites"]),
        avoid_immediate_repeat=bool(config["avoid_immediate_repeat"]),
        max_episode_steps=int(config["max_episode_steps"]),
        preserve_trajectory_continuity=bool(
            config.get("preserve_trajectory_continuity", True)
        ),
        ednet_session_gap_hours=float(config["ednet_session_gap_hours"]),
        reward_clip=float(config["reward_clip"]),
        mastery_progression_scale=float(config.get("mastery_progression_scale", 10.0)),
        reward_weights=RewardWeights(**dict(config["reward_weights"])),
    )


def vocabulary_size(path: Path, field: str) -> int:
    import json

    return len(json.loads(path.read_text(encoding="utf-8"))[field])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and validate EdNet/OULAD offline replay environments")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "phase8_offline_environment.json")
    parser.add_argument("--dataset", choices=("both", "ednet", "oulad"), default="both")
    parser.add_argument("--max-episodes", type=int, default=None)
    args = parser.parse_args()

    config, _ = load_config(args.config)
    sequence_root = Path(str(config["paths"]["sequence_root"]))
    output_root = Path(str(config["paths"]["output_root"]))
    output_root.mkdir(parents=True, exist_ok=True)
    datasets = ("ednet", "oulad") if args.dataset == "both" else (args.dataset,)
    max_episodes = args.max_episodes
    if max_episodes is None:
        max_episodes = int(config["development_max_episodes"])
    if max_episodes <= 0:
        raise ValueError("max_episodes must be positive")

    manifest: dict[str, object] = {
        "status": "running",
        "mode": "real-data environment validation",
        "warning": (
            "The provisional history-statistics encoder validates Phase 8 mechanics only. "
            "Behavior-policy and offline-policy training must regenerate states with a validation-selected Phase 7 checkpoint."
        ),
        "datasets": {},
    }
    for dataset in datasets:
        started = time.perf_counter()
        settings = settings_for(dataset, config)
        vocabulary_path = sequence_root / dataset / "vocabularies.json"
        mastery_dim = vocabulary_size(vocabulary_path, "concept_ids")
        encoder = ProvisionalStateEncoder(settings.state_dim, mastery_dim, settings.max_history)
        environment = build_environment(settings, sequence_root, encoder, max_episodes=max_episodes)
        validation = environment.validate_dataset()

        observation, reset_info = environment.reset(options={"episode_index": 0})
        replay_steps = 0
        terminated = truncated = False
        while not (terminated or truncated):
            observation, _, terminated, truncated, step_info = environment.replay_logged_step()
            replay_steps += 1

        catalog = environment.action_catalog.frame
        dataset_result = {
            **validation,
            "dataset": dataset,
            "split": settings.split,
            "encoder": encoder.name,
            "provisional_states": True,
            "observation_space": {
                "student_state": [settings.state_dim],
                "mastery": [mastery_dim],
                "recent_features": [8],
                "module": "Discrete",
            "action_mask": [int(environment.action_space.n)],
            },
            "action_space": f"Discrete({int(environment.action_space.n)})",
            "candidate_type_counts": {
                str(key): int(value) for key, value in catalog.groupby("item_type").size().items()
            },
            "first_episode_replay_steps": replay_steps,
            "first_episode_id": str(reset_info["episode_id"]),
            "runtime_seconds": time.perf_counter() - started,
        }
        write_json(dataset_result, output_root / dataset / "validation.json")
        manifest["datasets"][dataset] = dataset_result

    manifest["status"] = "complete"
    write_json(manifest, output_root / "manifest.json")
    print(f"Phase 8 environment validation complete: {output_root / 'manifest.json'}")


if __name__ == "__main__":
    main()

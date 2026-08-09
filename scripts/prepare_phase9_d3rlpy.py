from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from edu_recommender.config import load_config, write_json
from edu_recommender.d3rlpy_adapter import D3RLPYTransitionAdapter, create_d3rlpy_algorithm
from edu_recommender.offline_rl_environment import (
    EnvironmentSettings,
    ProvisionalStateEncoder,
    build_environment,
)


def vocabulary_size(path: Path, field: str) -> int:
    return len(json.loads(path.read_text(encoding="utf-8"))[field])


def environment_settings(dataset: str, values: dict[str, object]) -> EnvironmentSettings:
    return EnvironmentSettings(
        dataset=dataset,
        split=str(values["split"]),
        state_dim=int(values["state_dim"]),
        max_history=int(values["max_history"]),
        min_train_support=int(values["min_train_support"]),
        mastery_threshold=float(values["mastery_threshold"]),
        enforce_prerequisites=bool(values["enforce_prerequisites"]),
        avoid_immediate_repeat=bool(values["avoid_immediate_repeat"]),
        max_episode_steps=int(values["max_episode_steps"]),
        ednet_session_gap_hours=float(values["ednet_session_gap_hours"]),
        reward_clip=float(values["reward_clip"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare and validate d3rlpy datasets without policy training"
    )
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "phase9_d3rlpy.json"
    )
    parser.add_argument("--dataset", choices=("both", "ednet", "oulad"), default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    args = parser.parse_args()

    config, _ = load_config(args.config)
    if bool(config["training_enabled"]):
        raise ValueError("This preparation command must keep training_enabled=false")
    requested_dataset = args.dataset or str(config["dataset"])
    datasets = ("ednet", "oulad") if requested_dataset == "both" else (requested_dataset,)
    max_episodes = args.max_episodes or int(config["development_max_episodes"])
    sequence_root = Path(str(config["paths"]["sequence_root"]))
    output_root = Path(str(config["paths"]["output_root"]))
    output_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "status": "running",
        "library": "d3rlpy",
        "library_version": __import__("d3rlpy").__version__,
        "training_started": False,
        "algorithms": ["discrete_bc", "discrete_cql"],
        "discrete_iql_available": False,
        "warning": (
            "Development artifacts use provisional states. Regenerate them with the selected "
            "Phase 7 checkpoint before fitting BC or CQL."
        ),
        "datasets": {},
    }
    for dataset in datasets:
        settings = environment_settings(dataset, dict(config["environment"]))
        vocabulary_path = sequence_root / dataset / "vocabularies.json"
        mastery_dim = vocabulary_size(vocabulary_path, "concept_ids")
        module_count = vocabulary_size(vocabulary_path, "module_id")
        encoder = ProvisionalStateEncoder(settings.state_dim, mastery_dim, settings.max_history)
        environment = build_environment(settings, sequence_root, encoder, max_episodes=max_episodes)
        adapter = D3RLPYTransitionAdapter(module_count=module_count)
        bundle = adapter.from_environment(environment)
        native_dataset = bundle.to_mdp_dataset()

        # Construct both algorithm objects to validate configuration and action-space
        # compatibility. Deliberately do not call fit or fit_online.
        algorithm_types: dict[str, str] = {}
        for name, values in dict(config["algorithms"]).items():
            values = dict(values)
            algorithm = create_d3rlpy_algorithm(
                name,
                batch_size=int(values["batch_size"]),
                learning_rate=float(values["learning_rate"]),
                gamma=float(values["gamma"]),
                cql_alpha=float(values.get("alpha", 1.0)),
                device=False,
            )
            algorithm_types[name] = type(algorithm).__name__

        dataset_root = output_root / dataset
        dataset_root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            dataset_root / "development_dataset.npz",
            observations=bundle.observations,
            actions=bundle.actions,
            rewards=bundle.rewards,
            terminals=bundle.terminals,
            timeouts=bundle.timeouts,
        )
        summary = {
            **bundle.summary(),
            "native_action_size": int(native_dataset.dataset_info.action_size),
            "module_feature_size": adapter.module_feature_size,
            "algorithms_constructed_not_fitted": algorithm_types,
            "dense_action_masks_saved": False,
            "training_started": False,
        }
        write_json(summary, dataset_root / "validation.json")
        manifest["datasets"][dataset] = summary

    manifest["status"] = "complete"
    write_json(manifest, output_root / "manifest.json")
    print(f"Phase 9 d3rlpy preparation complete (training not started): {output_root / 'manifest.json'}")


if __name__ == "__main__":
    main()

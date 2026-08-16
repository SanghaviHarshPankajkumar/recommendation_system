from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
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
    RewardWeights,
    TorchStudentStateEncoder,
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
        preserve_trajectory_continuity=bool(
            values.get("preserve_trajectory_continuity", True)
        ),
        ednet_session_gap_hours=float(values["ednet_session_gap_hours"]),
        reward_clip=float(values["reward_clip"]),
        mastery_progression_scale=float(values.get("mastery_progression_scale", 10.0)),
        reward_weights=RewardWeights(**dict(values.get("reward_weights", {}))),
    )


def save_bundle(bundle, destination: Path) -> None:
    eligibility_offsets = np.zeros(len(bundle.eligible_actions) + 1, dtype=np.int64)
    eligibility_offsets[1:] = np.cumsum(
        [len(actions) for actions in bundle.eligible_actions], dtype=np.int64
    )
    eligibility_values = np.concatenate(bundle.eligible_actions).astype(np.int32, copy=False)
    np.savez_compressed(
        destination,
        observations=bundle.observations,
        actions=bundle.actions,
        rewards=bundle.rewards,
        terminals=bundle.terminals,
        timeouts=bundle.timeouts,
        eligibility_offsets=eligibility_offsets,
        eligibility_values=eligibility_values,
        episode_ids=np.asarray(bundle.episode_ids, dtype=np.str_),
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
    parser.add_argument(
        "--allow-provisional",
        action="store_true",
        help="Allow development-only statistical states when no Phase 7 checkpoint is configured",
    )
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
        "algorithms": ["discrete_bc"],
        "unsupported_algorithms": {
            "discrete_cql": (
                "Disabled: stock d3rlpy CQL ignores dynamic eligibility masks "
                "during Bellman and conservative-loss updates."
            )
        },
        "discrete_iql_available": False,
        "warning": None,
        "datasets": {},
    }
    for dataset in datasets:
        settings = environment_settings(dataset, dict(config["environment"]))
        vocabulary_path = sequence_root / dataset / "vocabularies.json"
        mastery_dim = vocabulary_size(vocabulary_path, "concept_ids")
        module_count = vocabulary_size(vocabulary_path, "module_id")
        checkpoint_settings = dict(config["paths"].get("state_checkpoints", {}))
        checkpoint_value = checkpoint_settings.get(dataset)
        if checkpoint_value:
            checkpoint_path = Path(str(checkpoint_value))
            if not checkpoint_path.is_absolute():
                checkpoint_path = PROJECT_ROOT / checkpoint_path
            if not checkpoint_path.exists():
                raise FileNotFoundError(checkpoint_path)
            encoder = TorchStudentStateEncoder(
                dataset=dataset,
                checkpoint_path=checkpoint_path,
                graph_root=Path(str(config["paths"]["graph_root"])),
                sequence_root=sequence_root,
                max_history=settings.max_history,
                max_concepts_per_event=settings.max_concepts_per_event,
                encode_batch_size=int(config.get("state_encode_batch_size", 128)),
                device="cpu",
            )
        elif args.allow_provisional:
            encoder = ProvisionalStateEncoder(
                settings.state_dim, mastery_dim, settings.max_history
            )
            manifest["warning"] = (
                "Development-only provisional states were explicitly allowed. "
                "These artifacts must not be used for final BC/RL training or evaluation."
            )
        else:
            raise ValueError(
                f"No Phase 7 checkpoint configured for {dataset}. Set "
                f"paths.state_checkpoints.{dataset}, or pass --allow-provisional "
                "only for a bounded development run."
            )
        environment = build_environment(settings, sequence_root, encoder, max_episodes=max_episodes)
        adapter = D3RLPYTransitionAdapter(module_count=module_count)
        bundle = adapter.from_environment(environment)
        native_dataset = bundle.to_mdp_dataset()
        evaluation_settings = replace(settings, split=str(config["evaluation_split"]))
        evaluation_environment = build_environment(
            evaluation_settings, sequence_root, encoder, max_episodes=max_episodes
        )
        evaluation_bundle = adapter.from_environment(evaluation_environment)

        # Construct both algorithm objects to validate configuration and action-space
        # compatibility. Deliberately do not call fit or fit_online.
        algorithm_types: dict[str, str] = {}
        for name, values in dict(config["algorithms"]).items():
            values = dict(values)
            if name == "discrete_cql":
                algorithm_types[name] = "disabled_dynamic_action_masks_not_supported"
                continue
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
        save_bundle(bundle, dataset_root / "development_dataset.npz")
        save_bundle(evaluation_bundle, dataset_root / "development_validation_dataset.npz")
        summary = {
            **bundle.summary(),
            "native_action_size": int(native_dataset.dataset_info.action_size),
            "module_feature_size": adapter.module_feature_size,
            "algorithm_validation": algorithm_types,
            "dense_action_masks_saved": False,
            "sparse_action_masks_saved": True,
            "training_started": False,
        }
        write_json(summary, dataset_root / "validation.json")
        evaluation_summary = {
            **evaluation_bundle.summary(),
            "split": evaluation_settings.split,
            "native_action_size": int(evaluation_bundle.to_mdp_dataset().dataset_info.action_size),
            "module_feature_size": adapter.module_feature_size,
            "dense_action_masks_saved": False,
            "sparse_action_masks_saved": True,
            "training_started": False,
        }
        write_json(evaluation_summary, dataset_root / "evaluation_validation.json")
        summary["held_out_evaluation"] = evaluation_summary
        manifest["datasets"][dataset] = summary

    manifest["status"] = "complete"
    write_json(manifest, output_root / "manifest.json")
    print(f"Phase 9 d3rlpy preparation complete (training not started): {output_root / 'manifest.json'}")


if __name__ == "__main__":
    main()

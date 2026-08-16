from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from edu_recommender.config import load_config, write_json
from edu_recommender.offline_policy_evaluation import (
    D3RLPYPolicyScorer,
    EvaluationError,
    OfflineEvaluationDataset,
    OfflinePolicyEvaluator,
    PopularityPolicyScorer,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a popularity, d3rlpy BC, or d3rlpy CQL policy on fixed data"
    )
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "phase9_evaluation.json"
    )
    parser.add_argument("--dataset", choices=("both", "ednet", "oulad"), default=None)
    parser.add_argument(
        "--policy", choices=("popularity", "discrete_bc", "discrete_cql"), default=None
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Trained .d3 checkpoint; allowed only when evaluating one dataset",
    )
    parser.add_argument("--max-transitions", type=int, default=None)
    parser.add_argument(
        "--allow-provisional",
        action="store_true",
        help="Allow checkpoint evaluation on provisional states (development only)",
    )
    args = parser.parse_args()

    config, _ = load_config(args.config)
    requested_dataset = args.dataset or str(config["dataset"])
    datasets = ("ednet", "oulad") if requested_dataset == "both" else (requested_dataset,)
    policy_name = args.policy or str(config["policy"])
    if args.checkpoint is not None and len(datasets) != 1:
        raise ValueError("--checkpoint requires --dataset ednet or --dataset oulad")
    max_transitions = args.max_transitions
    if max_transitions is None and config.get("max_transitions") is not None:
        max_transitions = int(config["max_transitions"])

    phase9_root = Path(str(config["paths"]["phase9_root"]))
    output_root = Path(str(config["paths"]["output_root"]))
    evaluator = OfflinePolicyEvaluator(
        ks=list(config["ranking_ks"]),
        bootstrap_replicates=int(config["bootstrap_replicates"]),
        seed=int(config["seed"]),
    )
    manifest: dict[str, object] = {
        "status": "running",
        "policy": policy_name,
        "datasets": {},
        "fqe_status": "pending a trained policy and separately trained FQE model",
    }

    for dataset_name in datasets:
        dataset_root = phase9_root / dataset_name
        dataset = OfflineEvaluationDataset.load(
            dataset_root / "development_validation_dataset.npz",
            dataset_root / "evaluation_validation.json",
        )
        if policy_name == "popularity":
            training_dataset = OfflineEvaluationDataset.load(
                dataset_root / "development_dataset.npz",
                dataset_root / "validation.json",
            )
            scorer = PopularityPolicyScorer(training_dataset.actions, dataset.action_size)
        else:
            checkpoint_value = (
                str(args.checkpoint)
                if args.checkpoint is not None
                else config["checkpoints"].get(dataset_name)
            )
            if not checkpoint_value:
                raise EvaluationError(
                    f"No trained {policy_name} checkpoint configured for {dataset_name}"
                )
            checkpoint = Path(str(checkpoint_value))
            if not checkpoint.is_absolute():
                checkpoint = PROJECT_ROOT / checkpoint
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            allow_provisional = bool(config["allow_provisional_model_evaluation"]) or args.allow_provisional
            if dataset.provisional_states and not allow_provisional:
                raise EvaluationError(
                    "Refusing to report trained-policy metrics on provisional states. "
                    "Regenerate Phase 9 with the selected Phase 7 checkpoint or pass "
                    "--allow-provisional for an explicitly developmental run."
                )
            import d3rlpy

            algorithm = d3rlpy.load_learnable(str(checkpoint), device=False)
            expected_type = "DiscreteBC" if policy_name == "discrete_bc" else "DiscreteCQL"
            if type(algorithm).__name__ != expected_type:
                raise EvaluationError(
                    f"Checkpoint contains {type(algorithm).__name__}, expected {expected_type}"
                )
            scorer = D3RLPYPolicyScorer(algorithm, policy_name)

        result = evaluator.evaluate(dataset, scorer, max_transitions=max_transitions)
        destination = output_root / dataset_name / policy_name / "metrics.json"
        write_json(result, destination)
        manifest["datasets"][dataset_name] = result

    manifest["status"] = "complete"
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(manifest, output_root / f"{policy_name}_manifest.json")
    print(f"Phase 9 evaluation complete: {output_root / f'{policy_name}_manifest.json'}")


if __name__ == "__main__":
    main()

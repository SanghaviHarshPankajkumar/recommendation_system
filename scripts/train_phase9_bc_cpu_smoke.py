from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from d3rlpy.logging import NoopAdapterFactory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from edu_recommender.config import load_config, write_json
from edu_recommender.d3rlpy_adapter import create_d3rlpy_algorithm
from edu_recommender.offline_policy_evaluation import OfflineEvaluationDataset


def masked_bc_accuracy(
    algorithm: Any,
    dataset: OfflineEvaluationDataset,
    *,
    max_transitions: int | None,
    batch_size: int,
) -> float:
    """Compute next-action accuracy after applying each row's eligibility mask."""

    if algorithm.impl is None:
        raise RuntimeError("The BC model must be built before accuracy is computed")
    rows = len(dataset.actions)
    if max_transitions is not None:
        rows = min(rows, int(max_transitions))
    if rows <= 0 or batch_size <= 0:
        raise ValueError("Accuracy row count and batch_size must be positive")

    correct = 0
    device = algorithm._device  # d3rlpy 2.8.1 integration point
    scaler = algorithm.config.observation_scaler
    for start in range(0, rows, batch_size):
        stop = min(start + batch_size, rows)
        observations = torch.as_tensor(dataset.observations[start:stop], device=device)
        if scaler is not None:
            observations = scaler.transform(observations)
        with torch.no_grad():
            logits = algorithm.impl.modules.imitator(observations).logits
        for offset, row in enumerate(range(start, stop)):
            eligible = torch.as_tensor(
                dataset.eligible_actions[row], dtype=torch.long, device=logits.device
            )
            selected = int(eligible[torch.argmax(logits[offset, eligible])].item())
            correct += int(selected == int(dataset.actions[row]))
    return float(correct / rows)


def _load_dataset(dataset_root: Path, split: str) -> OfflineEvaluationDataset:
    stem = "development" if split == "train" else "development_validation"
    metadata = "validation.json" if split == "train" else "evaluation_validation.json"
    return OfflineEvaluationDataset.load(dataset_root / f"{stem}_dataset.npz", dataset_root / metadata)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Behaviour Cloning on CPU with periodic accuracy reporting"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "phase9_bc_cpu_smoke.json",
    )
    parser.add_argument("--dataset", choices=("ednet", "oulad"), default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument(
        "--allow-provisional",
        action="store_true",
        help="Permit a development-only run with provisional Phase 8 states",
    )
    args = parser.parse_args()

    config, _ = load_config(args.config)
    settings = dict(config["training"])
    dataset_name = args.dataset or str(config["dataset"])
    n_steps = args.steps if args.steps is not None else int(settings["n_steps"])
    if n_steps <= 0:
        raise ValueError("--steps must be positive")

    phase9_root = Path(str(config["paths"]["phase9_root"]))
    output_root = Path(str(config["paths"]["output_root"])) / dataset_name
    dataset_root = phase9_root / dataset_name
    train_data = _load_dataset(dataset_root, "train")
    validation_data = _load_dataset(dataset_root, "validation")
    if train_data.provisional_states or validation_data.provisional_states:
        if not args.allow_provisional:
            raise RuntimeError(
                "Refusing to train on provisional states. For an explicitly developmental "
                "CPU smoke run, pass --allow-provisional."
            )

    seed = int(settings["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(settings.get("cpu_threads", max(1, min(4, torch.get_num_threads())))))
    algorithm = create_d3rlpy_algorithm(
        "discrete_bc",
        batch_size=int(settings["batch_size"]),
        learning_rate=float(settings["learning_rate"]),
        gamma=float(settings["gamma"]),
        bc_beta=float(settings.get("beta", 0.5)),
        device=False,
    )
    native_dataset = train_data_to_mdp_dataset(train_data)
    algorithm.build_with_dataset(native_dataset)

    accuracy_limit = settings.get("accuracy_max_transitions")
    accuracy_limit = None if accuracy_limit is None else int(accuracy_limit)
    accuracy_batch_size = int(settings["accuracy_batch_size"])
    history: list[dict[str, float | int]] = []

    def report(step: int) -> None:
        train_accuracy = masked_bc_accuracy(
            algorithm,
            train_data,
            max_transitions=accuracy_limit,
            batch_size=accuracy_batch_size,
        )
        validation_accuracy = masked_bc_accuracy(
            algorithm,
            validation_data,
            max_transitions=accuracy_limit,
            batch_size=accuracy_batch_size,
        )
        previous = history[-1] if history else None
        train_delta = train_accuracy - float(previous["train_accuracy"]) if previous else 0.0
        validation_delta = (
            validation_accuracy - float(previous["validation_accuracy"]) if previous else 0.0
        )
        row: dict[str, float | int] = {
            "step": step,
            "train_accuracy": train_accuracy,
            "train_accuracy_change": train_delta,
            "validation_accuracy": validation_accuracy,
            "validation_accuracy_change": validation_delta,
        }
        history.append(row)
        print(
            f"[BC CPU step {step:03d}/{n_steps:03d}] "
            f"train_accuracy={train_accuracy:.4f} ({train_delta:+.4f}) | "
            f"validation_accuracy={validation_accuracy:.4f} ({validation_delta:+.4f})",
            flush=True,
        )
        print(f"METRIC_JSON:{json.dumps({'model': 'bc', **row})}", flush=True)

    print(
        f"Starting CPU-only BC training: dataset={dataset_name}, "
        f"train_rows={len(train_data.actions)}, validation_rows={len(validation_data.actions)}, "
        f"optimizer_steps={n_steps}, provisional_states={train_data.provisional_states}",
        flush=True,
    )
    report(0)

    metric_interval = int(settings.get("metric_interval", 1))
    if metric_interval <= 0:
        raise ValueError("metric_interval must be positive")

    def callback(_algorithm: Any, _epoch: int, total_step: int) -> None:
        if int(total_step) % metric_interval == 0 or int(total_step) == n_steps:
            report(int(total_step))

    algorithm.fit(
        native_dataset,
        n_steps=n_steps,
        n_steps_per_epoch=1,
        logging_steps=1,
        logger_adapter=NoopAdapterFactory(),
        show_progress=False,
        save_interval=max(n_steps + 1, 2),
        callback=callback,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint = output_root / "discrete_bc_cpu.d3"
    algorithm.save(str(checkpoint))
    result = {
        "status": "complete",
        "dataset": dataset_name,
        "device": "cpu",
        "provisional_states": train_data.provisional_states,
        "development_only": bool(train_data.provisional_states),
        "training_rows": len(train_data.actions),
        "validation_rows": len(validation_data.actions),
        "accuracy_rows_per_split": accuracy_limit,
        "optimizer_steps": n_steps,
        "checkpoint": str(checkpoint),
        "history": history,
    }
    write_json(result, output_root / "training_history.json")
    print(f"BC CPU training complete: {output_root / 'training_history.json'}", flush=True)


def train_data_to_mdp_dataset(dataset: OfflineEvaluationDataset):
    """Convert saved Phase 9 arrays back to d3rlpy's native replay dataset."""

    from d3rlpy.dataset import MDPDataset

    return MDPDataset(
        observations=dataset.observations,
        actions=dataset.actions,
        rewards=dataset.rewards,
        terminals=dataset.terminals,
        timeouts=dataset.timeouts,
        action_size=dataset.action_size,
    )


if __name__ == "__main__":
    main()

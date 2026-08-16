from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from .phase6_training import collate_windows, process_memory_bytes
from .sequence_building import LazySequenceWindows, SPLIT_IDS
from .student_state_model import (
    GraphTensorBuilder,
    KnowledgeAwareMultiTaskLoss,
    KnowledgeAwareStudentStateModel,
    model_config_from_vocabularies,
)


NEURAL_VARIANTS = {"sequence_only", "graph_only", "sequence_graph", "sequence_mastery", "full"}
ALL_VARIANTS = NEURAL_VARIANTS | {"popularity"}


class Phase7ConfigurationError(ValueError):
    pass


class Phase7ConfigValidator:
    """Reject unsafe or internally inconsistent pretraining configurations."""

    @staticmethod
    def validate(config: dict[str, object]) -> dict[str, object]:
        required_paths = ["graph_root", "sequence_root", "output_root"]
        missing = [key for key in required_paths if key not in config]
        if missing:
            raise Phase7ConfigurationError(f"Missing paths: {missing}")
        datasets = list(config.get("datasets", []))
        if not datasets or any(dataset not in {"ednet", "oulad"} for dataset in datasets):
            raise Phase7ConfigurationError("datasets must contain ednet and/or oulad")
        variants = list(config.get("variants", []))
        unknown_variants = set(variants).difference(ALL_VARIANTS)
        if not variants or unknown_variants:
            raise Phase7ConfigurationError(f"Unknown or empty variants: {sorted(unknown_variants)}")
        for key in ("epochs", "batch_size", "max_train_windows", "max_validation_windows"):
            if int(config.get(key, 0)) <= 0:
                raise Phase7ConfigurationError(f"{key} must be positive")
        if int(config.get("warmup_steps", 0)) < 0:
            raise Phase7ConfigurationError("warmup_steps cannot be negative")
        if config.get("selection_split", "validation") != "validation":
            raise Phase7ConfigurationError("Model selection must use validation, never test")
        if "test" in json.dumps(config.get("selection_metric", "validation_total_loss")).lower():
            raise Phase7ConfigurationError("Test metrics cannot select checkpoints")
        sequence_root = Path(str(config["sequence_root"]))
        graph_root = Path(str(config["graph_root"]))
        path_checks: dict[str, bool] = {}
        for dataset in datasets:
            expected = [
                sequence_root / dataset / "vocabularies.json",
                sequence_root / dataset / "candidate_catalog.csv.gz",
                sequence_root / dataset / "packed",
                graph_root / dataset / "nodes.csv.gz",
                graph_root / dataset / "edges_explicit.csv.gz",
            ]
            for path in expected:
                path_checks[str(path)] = path.exists()
        missing_files = [path for path, present in path_checks.items() if not present]
        if missing_files:
            raise Phase7ConfigurationError(f"Missing Phase 4/5 inputs: {missing_files}")
        return {
            "valid": True,
            "training_executed": False,
            "datasets": datasets,
            "variants": variants,
            "path_checks": path_checks,
            "selection_split": "validation",
        }


class WarmupCosineSchedule:
    """Picklable LambdaLR callable with linear warm-up and cosine decay."""

    def __init__(self, warmup_steps: int, total_steps: int, min_ratio: float = 0.1):
        if total_steps <= 0 or warmup_steps < 0 or warmup_steps >= total_steps:
            raise ValueError("Require 0 <= warmup_steps < total_steps")
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_ratio = min_ratio

    def __call__(self, step: int) -> float:
        if self.warmup_steps and step < self.warmup_steps:
            return max((step + 1) / self.warmup_steps, 1 / self.warmup_steps)
        progress = (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_ratio + (1.0 - self.min_ratio) * cosine


@dataclass
class EarlyStopping:
    patience: int
    min_delta: float = 0.0
    mode: str = "min"
    best: float | None = None
    bad_epochs: int = 0

    def update(self, value: float) -> tuple[bool, bool]:
        if self.mode not in {"min", "max"}:
            raise ValueError("mode must be min or max")
        improved = self.best is None or (
            value < self.best - self.min_delta if self.mode == "min" else value > self.best + self.min_delta
        )
        if improved:
            self.best = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return improved, self.bad_epochs >= self.patience


class WindowBatchStream:
    """Bounded packed-window stream with epoch-seeded training shuffling."""

    def __init__(
        self,
        packed_dir: Path,
        split: str,
        max_length: int,
        batch_size: int,
        max_windows: int,
        seed: int,
        shuffle: bool,
        max_concepts_per_event: int = 9,
    ):
        if split not in {"train", "validation"}:
            raise ValueError("Phase 7 streams may use only train or validation")
        self.packed_dir = Path(packed_dir)
        self.split = split
        self.max_length = max_length
        self.batch_size = batch_size
        self.max_windows = max_windows
        self.seed = seed
        self.shuffle = shuffle
        self.max_concepts_per_event = max_concepts_per_event

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        rng = np.random.default_rng(self.seed)
        paths = sorted(self.packed_dir.glob("part-*.npz"))
        if self.shuffle:
            rng.shuffle(paths)
        pending: list[dict[str, np.ndarray]] = []
        observed = 0
        split_id = SPLIT_IDS[self.split]
        for path in paths:
            loader = LazySequenceWindows(path, self.max_length, self.max_concepts_per_event)
            indices = np.flatnonzero(loader.arrays["window_split_ids"] == split_id)
            if self.shuffle:
                rng.shuffle(indices)
            for index in indices:
                pending.append(loader.get(int(index)))
                observed += 1
                if len(pending) == self.batch_size:
                    yield collate_windows(pending)
                    pending = []
                if observed >= self.max_windows:
                    break
            loader.arrays.close()
            if observed >= self.max_windows:
                break
        if pending:
            yield collate_windows(pending)


class BinaryMetrics:
    def __init__(self, calibration_bins: int = 10):
        self.probabilities: list[np.ndarray] = []
        self.targets: list[np.ndarray] = []
        self.calibration_bins = calibration_bins

    def update(self, probabilities: Tensor, targets: Tensor) -> None:
        self.probabilities.append(probabilities.detach().float().cpu().numpy())
        self.targets.append(targets.detach().float().cpu().numpy())

    def compute(self) -> dict[str, float | None]:
        if not self.targets:
            return {name: None for name in ("auc", "accuracy", "precision", "recall", "f1", "brier", "ece")}
        probabilities = np.concatenate(self.probabilities).astype(np.float64)
        targets = np.concatenate(self.targets).astype(np.int64)
        predictions = probabilities >= 0.5
        tp = int(np.sum(predictions & (targets == 1)))
        fp = int(np.sum(predictions & (targets == 0)))
        fn = int(np.sum(~predictions & (targets == 1)))
        accuracy = float(np.mean(predictions == targets))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {
            "auc": self._auc(probabilities, targets),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "brier": float(np.mean((probabilities - targets) ** 2)),
            "ece": self._ece(probabilities, targets),
        }

    @staticmethod
    def _auc(scores: np.ndarray, targets: np.ndarray) -> float | None:
        positives = int(np.sum(targets == 1))
        negatives = int(np.sum(targets == 0))
        if not positives or not negatives:
            return None
        order = np.argsort(scores, kind="stable")
        sorted_scores = scores[order]
        ranks = np.empty(len(scores), dtype=np.float64)
        start = 0
        while start < len(scores):
            end = start + 1
            while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
                end += 1
            ranks[order[start:end]] = ((start + 1) + end) / 2.0
            start = end
        positive_rank_sum = float(ranks[targets == 1].sum())
        return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)

    def _ece(self, probabilities: np.ndarray, targets: np.ndarray) -> float:
        error = 0.0
        boundaries = np.linspace(0, 1, self.calibration_bins + 1)
        for index in range(self.calibration_bins):
            lower, upper = boundaries[index], boundaries[index + 1]
            mask = (probabilities >= lower) & (probabilities < upper if index + 1 < self.calibration_bins else probabilities <= upper)
            if np.any(mask):
                error += float(mask.mean()) * abs(float(probabilities[mask].mean()) - float(targets[mask].mean()))
        return error


class CandidateRankingMetrics:
    """Candidate-supported single-positive Hit@K, NDCG@K, and MRR."""

    def __init__(self, catalog_path: Path, ks: list[int], max_examples: int):
        catalog = pd.read_csv(catalog_path)
        catalog = catalog[catalog["train_support"] >= 5]
        self.by_module = {
            int(module): group["item_token"].drop_duplicates().to_numpy(dtype=np.int64)
            for module, group in catalog.groupby("module_token")
        }
        self.global_candidates = catalog["item_token"].drop_duplicates().to_numpy(dtype=np.int64)
        self.support_scores = dict(zip(catalog["item_token"].astype(int), np.log1p(catalog["train_support"].astype(float))))
        self.ks = sorted(ks)
        self.max_examples = max_examples
        self.ranks: list[float] = []

    def update_model(self, logits: Tensor, targets: Tensor, modules: Tensor, target_mask: Tensor) -> None:
        positions = torch.nonzero(target_mask, as_tuple=False)
        for batch_index, sequence_index in positions.tolist():
            if len(self.ranks) >= self.max_examples:
                return
            target = int(targets[batch_index, sequence_index])
            candidates = self._candidates(int(modules[batch_index, sequence_index]))
            if target not in candidates:
                continue
            candidate_tensor = torch.tensor(candidates.copy(), device=logits.device, dtype=torch.long)
            scores = logits[batch_index, sequence_index, candidate_tensor]
            target_score = logits[batch_index, sequence_index, target]
            rank = 1.0 + float(torch.sum(scores > target_score).item()) + 0.5 * float(torch.sum(scores == target_score).item() - 1)
            self.ranks.append(rank)

    def update_popularity(self, targets: Tensor, modules: Tensor, target_mask: Tensor) -> None:
        positions = torch.nonzero(target_mask, as_tuple=False)
        for batch_index, sequence_index in positions.tolist():
            if len(self.ranks) >= self.max_examples:
                return
            target = int(targets[batch_index, sequence_index])
            candidates = self._candidates(int(modules[batch_index, sequence_index]))
            if target not in candidates:
                continue
            target_score = self.support_scores.get(target, 0.0)
            scores = np.asarray([self.support_scores.get(int(candidate), 0.0) for candidate in candidates])
            rank = 1.0 + float(np.sum(scores > target_score)) + 0.5 * float(np.sum(scores == target_score) - 1)
            self.ranks.append(rank)

    def _candidates(self, module_token: int) -> np.ndarray:
        return self.by_module.get(module_token, self.global_candidates)

    def compute(self) -> dict[str, float | int]:
        if not self.ranks:
            return {"examples": 0, "mrr": 0.0, **{f"hit_rate@{k}": 0.0 for k in self.ks}, **{f"ndcg@{k}": 0.0 for k in self.ks}}
        ranks = np.asarray(self.ranks)
        result: dict[str, float | int] = {"examples": len(ranks), "mrr": float(np.mean(1.0 / ranks))}
        for k in self.ks:
            result[f"hit_rate@{k}"] = float(np.mean(ranks <= k))
            result[f"ndcg@{k}"] = float(np.mean(np.where(ranks <= k, 1.0 / np.log2(ranks + 1), 0.0)))
        return result


class ValidationMetrics:
    def __init__(self, ranking: CandidateRankingMetrics):
        self.ranking = ranking
        self.correctness = BinaryMetrics()
        self.mastery = BinaryMetrics()
        self.loss_sums: dict[str, float] = {name: 0.0 for name in ("total", "item", "action", "correctness", "mastery")}
        self.target_count = 0
        self.action_correct = 0

    def update(self, outputs: dict[str, Tensor], losses: dict[str, Tensor], batch: dict[str, Tensor]) -> None:
        mask = batch["target_mask"].bool()
        count = int(mask.sum())
        self.target_count += count
        for name, value in losses.items():
            self.loss_sums[name] += float(value.detach().cpu()) * count
        correctness_targets = batch["target_correctness"].float()
        correctness_mask = mask & correctness_targets.ge(0)
        if torch.any(correctness_mask):
            self.correctness.update(
                torch.sigmoid(outputs["correctness_logits"][correctness_mask]), correctness_targets[correctness_mask]
            )
        target_concepts = batch["target_concept_tokens"].long()
        mastery_mask = target_concepts.ne(0) & mask.unsqueeze(-1) & correctness_targets.ge(0).unsqueeze(-1)
        if torch.any(mastery_mask):
            mastery_probabilities = outputs["mastery_probabilities"].gather(-1, target_concepts.clamp_min(0))
            mastery_targets = correctness_targets.unsqueeze(-1).expand_as(mastery_probabilities)
            self.mastery.update(mastery_probabilities[mastery_mask], mastery_targets[mastery_mask])
        self.action_correct += int(
            (outputs["action_logits"].argmax(-1)[mask] == batch["target_action_tokens"].long()[mask]).sum()
        )
        self.ranking.update_model(
            outputs["item_logits"],
            batch["target_item_tokens"],
            batch["module_tokens"][:, 1:],
            mask,
        )

    def compute(self) -> dict[str, object]:
        denominator = max(self.target_count, 1)
        return {
            "target_count": self.target_count,
            "losses": {name: value / denominator for name, value in self.loss_sums.items()},
            "correctness": self.correctness.compute(),
            "mastery": self.mastery.compute(),
            "action_accuracy": self.action_correct / denominator,
            "candidate_ranking": self.ranking.compute(),
        }


class AtomicCheckpointManager:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        name: str,
        model: KnowledgeAwareStudentStateModel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        epoch: int,
        global_step: int,
        best_metric: float | None,
        history: list[dict[str, object]],
    ) -> Path:
        destination = self.output_dir / name
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "model_config": asdict(model.config),
                "epoch": epoch,
                "global_step": global_step,
                "best_metric": best_metric,
                "history": history,
                "torch_rng_state": torch.get_rng_state(),
            },
            temporary,
        )
        temporary.replace(destination)
        return destination

    @staticmethod
    def restore(
        path: Path,
        model: KnowledgeAwareStudentStateModel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
    ) -> dict[str, object]:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        device = next(model.parameters()).device
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, Tensor):
                    state[key] = value.to(device)
        torch.set_rng_state(checkpoint["torch_rng_state"])
        return checkpoint


class Phase7PretrainingPipeline:
    def __init__(
        self,
        config: dict[str, object],
        progress_callback: Callable[[dict[str, object]], None] | None = None,
    ):
        self.config = config
        self.progress_callback = progress_callback
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.get("use_cuda", True) else "cpu")

    @classmethod
    def from_json(cls, path: Path) -> "Phase7PretrainingPipeline":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def validate_only(self) -> dict[str, object]:
        result = Phase7ConfigValidator.validate(self.config)
        result["device_if_executed"] = str(self.device)
        result["planned_windows"] = {
            "train_per_epoch": int(self.config["max_train_windows"]),
            "validation_per_epoch": int(self.config["max_validation_windows"]),
        }
        return result

    def run(self) -> dict[str, object]:
        Phase7ConfigValidator.validate(self.config)
        torch.manual_seed(int(self.config["seed"]))
        np.random.seed(int(self.config["seed"]))
        random.seed(int(self.config["seed"]))
        torch.set_num_threads(int(self.config["cpu_threads"]))
        output_root = Path(str(self.config["output_root"]))
        output_root.mkdir(parents=True, exist_ok=True)
        summary: dict[str, object] = {"status": "running", "device": str(self.device), "datasets": {}}
        for dataset in self.config["datasets"]:
            dataset_results: dict[str, object] = {}
            for variant in self.config["variants"]:
                dataset_results[variant] = (
                    self._run_popularity(dataset) if variant == "popularity" else self._run_neural(dataset, variant)
                )
            summary["datasets"][dataset] = dataset_results
            (output_root / "comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summary["status"] = "complete"
        (output_root / "comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    def _stream(self, dataset: str, split: str, epoch: int, shuffle: bool) -> WindowBatchStream:
        return WindowBatchStream(
            Path(str(self.config["sequence_root"])) / dataset / "packed",
            split,
            int(self.config["window_length"]),
            int(self.config["batch_size"]),
            int(self.config["max_train_windows"] if split == "train" else self.config["max_validation_windows"]),
            int(self.config["seed"]) + epoch,
            shuffle,
            int(self.config.get("max_concepts_per_event", 9)),
        )

    def _run_popularity(self, dataset: str) -> dict[str, object]:
        sequence_dir = Path(str(self.config["sequence_root"])) / dataset
        ranking = CandidateRankingMetrics(
            sequence_dir / "candidate_catalog.csv.gz",
            list(self.config["ranking_ks"]),
            int(self.config["max_ranking_examples"]),
        )
        target_count = 0
        for batch in self._stream(dataset, "validation", 0, False):
            mask = batch["target_mask"].bool()
            target_count += int(mask.sum())
            ranking.update_popularity(batch["target_item_tokens"], batch["module_tokens"][:, 1:], mask)
        result = {"status": "complete", "training_required": False, "validation_targets": target_count, "candidate_ranking": ranking.compute()}
        output_dir = Path(str(self.config["output_root"])) / dataset / "popularity"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "validation_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    def _run_neural(self, dataset: str, variant: str) -> dict[str, object]:
        sequence_root = Path(str(self.config["sequence_root"]))
        graph = GraphTensorBuilder(dataset, Path(str(self.config["graph_root"])), sequence_root).build().to(self.device)
        model_options = dict(self.config["model"])
        model_options["variant"] = variant
        model_config = model_config_from_vocabularies(sequence_root / dataset / "vocabularies.json", model_options)
        model = KnowledgeAwareStudentStateModel(model_config, graph).to(self.device)
        loss_options = dict(self.config["loss_weights"])
        if variant == "graph_only":
            loss_options["mastery_weight"] = 0.0
        loss_function = KnowledgeAwareMultiTaskLoss(**loss_options)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(self.config["learning_rate"]), weight_decay=float(self.config["weight_decay"])
        )
        steps_per_epoch = math.ceil(int(self.config["max_train_windows"]) / int(self.config["batch_size"]))
        total_steps = steps_per_epoch * int(self.config["epochs"])
        schedule_function = WarmupCosineSchedule(
            min(int(self.config["warmup_steps"]), total_steps - 1), total_steps, float(self.config["minimum_lr_ratio"])
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_function)
        output_dir = Path(str(self.config["output_root"])) / dataset / variant
        checkpoints = AtomicCheckpointManager(output_dir)
        stopper = EarlyStopping(
            int(self.config["early_stopping_patience"]), float(self.config["early_stopping_min_delta"]), "min"
        )
        history: list[dict[str, object]] = []
        start_epoch = global_step = 0
        resume_path = self._resolve_resume_checkpoint(dataset, variant)
        if resume_path:
            restored = checkpoints.restore(Path(str(resume_path)), model, optimizer, scheduler)
            start_epoch = int(restored["epoch"]) + 1
            global_step = int(restored["global_step"])
            stopper.best = restored.get("best_metric")
            history = list(restored.get("history", []))

        peak_memory = process_memory_bytes()
        started = time.perf_counter()
        for epoch in range(start_epoch, int(self.config["epochs"])):
            train_result, global_step = self._train_epoch(
                model, graph, loss_function, optimizer, scheduler, dataset, epoch, global_step
            )
            validation_result = self._validate(model, graph, loss_function, dataset)
            if self.progress_callback is not None:
                self.progress_callback(
                    {
                        "model": "state",
                        "dataset": dataset,
                        "variant": variant,
                        "epoch": epoch + 1,
                        "step": global_step,
                        "split": "validation",
                        "validation_total_loss": validation_result["losses"]["total"],
                        "validation_correctness_accuracy": validation_result["correctness"]["accuracy"],
                        "validation_correctness_auc": validation_result["correctness"]["auc"],
                        "validation_action_accuracy": validation_result["action_accuracy"],
                        "validation_mrr": validation_result["candidate_ranking"]["mrr"],
                        "validation_hit_rate_5": validation_result["candidate_ranking"].get("hit_rate@5"),
                        "validation_ndcg_10": validation_result["candidate_ranking"].get("ndcg@10"),
                        "evaluation_split": "validation",
                        "seed_count": 1,
                        "full_held_out_temporal_test": False,
                        "leakage_check_passed": None,
                        "provisional_states": False,
                    }
                )
            selection_value = float(validation_result["losses"]["total"])
            improved, should_stop = stopper.update(selection_value)
            record = {
                "epoch": epoch,
                "train": train_result,
                "validation": validation_result,
                "learning_rate": scheduler.get_last_lr()[0],
                "improved": improved,
            }
            history.append(record)
            checkpoints.save("last_checkpoint.pt", model, optimizer, scheduler, epoch, global_step, stopper.best, history)
            if improved:
                checkpoints.save("best_checkpoint.pt", model, optimizer, scheduler, epoch, global_step, stopper.best, history)
            (output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            (output_dir / "validation_metrics.json").write_text(json.dumps(validation_result, indent=2), encoding="utf-8")
            peak_memory = max(peak_memory, process_memory_bytes())
            if should_stop:
                break
        result = {
            "status": "complete",
            "variant": variant,
            "epochs_completed": len(history),
            "best_validation_total_loss": stopper.best,
            "global_steps": global_step,
            "runtime_seconds": time.perf_counter() - started,
            "peak_process_memory_mb": peak_memory / 1_000_000,
            "best_checkpoint": str(output_dir / "best_checkpoint.pt"),
        }
        (output_dir / "runtime_profile.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    def _resolve_resume_checkpoint(self, dataset: str, variant: str) -> str | None:
        setting = self.config.get("resume_checkpoint")
        if setting is None or isinstance(setting, str):
            return setting
        if isinstance(setting, dict):
            dataset_setting = setting.get(dataset)
            if isinstance(dataset_setting, dict):
                value = dataset_setting.get(variant)
                return str(value) if value else None
            value = setting.get(f"{dataset}:{variant}")
            return str(value) if value else None
        raise Phase7ConfigurationError("resume_checkpoint must be null, a path, or a dataset/variant mapping")

    def _train_epoch(self, model, graph, loss_function, optimizer, scheduler, dataset, epoch, global_step):
        model.train()
        totals = {name: 0.0 for name in ("total", "item", "action", "correctness", "mastery")}
        targets = 0
        correctness_correct = correctness_count = action_correct = 0
        for cpu_batch in self._stream(dataset, "train", epoch, True):
            batch = {name: value.to(self.device) for name, value in cpu_batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch, graph)
            losses = loss_function(outputs, batch)
            losses["total"].backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(self.config["gradient_clip_norm"]))
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("Non-finite gradient norm")
            optimizer.step()
            scheduler.step()
            count = int(batch["target_mask"].sum())
            targets += count
            for name, value in losses.items():
                totals[name] += float(value.detach().cpu()) * count
            mask = batch["target_mask"].bool()
            correctness_targets = batch["target_correctness"].float()
            correctness_mask = mask & correctness_targets.ge(0)
            if torch.any(correctness_mask):
                correctness_predictions = outputs["correctness_logits"][correctness_mask].ge(0)
                correctness_correct += int(
                    (correctness_predictions == correctness_targets[correctness_mask].bool()).sum()
                )
                correctness_count += int(correctness_mask.sum())
            action_correct += int(
                (
                    outputs["action_logits"].argmax(-1)[mask]
                    == batch["target_action_tokens"].long()[mask]
                ).sum()
            )
            global_step += 1
            if self.progress_callback is not None:
                self.progress_callback(
                    {
                        "model": "state",
                        "dataset": dataset,
                        "variant": model.config.variant,
                        "epoch": epoch + 1,
                        "step": global_step,
                        "train_total_loss": totals["total"] / max(targets, 1),
                        "train_correctness_accuracy": correctness_correct / max(correctness_count, 1),
                        "train_action_accuracy": action_correct / max(targets, 1),
                        "learning_rate": scheduler.get_last_lr()[0],
                    }
                )
        return {"target_count": targets, "losses": {name: value / max(targets, 1) for name, value in totals.items()}}, global_step

    def _validate(self, model, graph, loss_function, dataset):
        model.eval()
        sequence_dir = Path(str(self.config["sequence_root"])) / dataset
        metrics = ValidationMetrics(
            CandidateRankingMetrics(
                sequence_dir / "candidate_catalog.csv.gz",
                list(self.config["ranking_ks"]),
                int(self.config["max_ranking_examples"]),
            )
        )
        with torch.no_grad():
            for cpu_batch in self._stream(dataset, "validation", 0, False):
                batch = {name: value.to(self.device) for name, value in cpu_batch.items()}
                outputs = model(batch, graph)
                metrics.update(outputs, loss_function(outputs, batch), batch)
        return metrics.compute()

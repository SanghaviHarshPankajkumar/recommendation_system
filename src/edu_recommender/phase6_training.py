from __future__ import annotations

import ctypes
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch import Tensor

from .sequence_building import LazySequenceWindows
from .student_state_model import (
    GraphTensorBuilder,
    KnowledgeAwareMultiTaskLoss,
    KnowledgeAwareStudentStateModel,
    model_config_from_vocabularies,
)


def process_memory_bytes() -> int:
    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
        ]
    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(counters)
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
    get_memory.argtypes = [ctypes.c_void_p, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), ctypes.c_ulong]
    get_memory.restype = ctypes.c_int
    if not get_memory(handle, ctypes.byref(counters), counters.cb):
        raise OSError("GetProcessMemoryInfo failed")
    return int(counters.WorkingSetSize)


class RealWindowBatchStream:
    def __init__(self, packed_dir: Path, split: str, max_length: int, batch_size: int):
        self.packed_dir = Path(packed_dir)
        self.split = split
        self.max_length = max_length
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        pending: list[dict[str, np.ndarray]] = []
        for path in sorted(self.packed_dir.glob("part-*.npz")):
            loader = LazySequenceWindows(path, self.max_length)
            for sample in loader.iter_split(self.split):
                pending.append(sample)
                if len(pending) == self.batch_size:
                    yield collate_windows(pending)
                    pending = []
        if pending:
            yield collate_windows(pending)


def collate_windows(samples: list[dict[str, np.ndarray]]) -> dict[str, Tensor]:
    fields = [
        "item_tokens", "action_tokens", "item_type_tokens", "module_tokens", "source_tokens",
        "relative_days", "time_gaps", "correctness", "scores", "elapsed_log1p", "engagement_log1p",
        "input_item_tokens", "input_attention_mask", "input_concept_tokens", "target_concept_tokens",
        "target_item_tokens", "target_action_tokens", "target_correctness", "target_mask",
    ]
    result: dict[str, Tensor] = {}
    for field in fields:
        values = np.stack([sample[field] for sample in samples])
        result[field] = torch.from_numpy(values)
    return result


class Phase6SmokeTrainer:
    def __init__(self, config: dict[str, object]):
        self.config = config
        self.output_root = Path(str(config["output_root"]))
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.get("use_cuda", True) else "cpu")

    @classmethod
    def from_json(cls, path: Path) -> "Phase6SmokeTrainer":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def run(self) -> dict[str, object]:
        torch.manual_seed(int(self.config.get("seed", 42)))
        torch.set_num_threads(int(self.config.get("cpu_threads", 2)))
        self.output_root.mkdir(parents=True, exist_ok=True)
        manifest: dict[str, object] = {
            "status": "running",
            "torch_version": torch.__version__,
            "device": str(self.device),
            "datasets": {},
        }
        for dataset in ("ednet", "oulad"):
            manifest["datasets"][dataset] = self._run_dataset(dataset)
            (self.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        manifest["status"] = "complete"
        (self.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    def _run_dataset(self, dataset: str) -> dict[str, object]:
        sequence_root = Path(str(self.config["sequence_root"]))
        graph = GraphTensorBuilder(dataset, Path(str(self.config["graph_root"])), sequence_root).build().to(self.device)
        model_config = model_config_from_vocabularies(
            sequence_root / dataset / "vocabularies.json", dict(self.config["model"])
        )
        model = KnowledgeAwareStudentStateModel(model_config, graph).to(self.device)
        loss_function = KnowledgeAwareMultiTaskLoss(**dict(self.config["loss_weights"]))
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(self.config["learning_rate"]), weight_decay=float(self.config["weight_decay"])
        )
        train_stream = iter(
            RealWindowBatchStream(
                sequence_root / dataset / "packed",
                "train",
                int(self.config["window_length"]),
                int(self.config["batch_size"]),
            )
        )
        step_metrics = []
        max_memory = process_memory_bytes()
        model.train()
        for step in range(int(self.config["smoke_steps"])):
            batch = {name: value.to(self.device) for name, value in next(train_stream).items()}
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch, graph)
            losses = loss_function(outputs, batch)
            losses["total"].backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(self.config["gradient_clip_norm"]))
            optimizer.step()
            duration = time.perf_counter() - start
            max_memory = max(max_memory, process_memory_bytes())
            step_metrics.append(
                {
                    "step": step + 1,
                    "seconds": duration,
                    "gradient_norm": float(gradient_norm.detach().cpu()),
                    **{f"{name}_loss": float(value.detach().cpu()) for name, value in losses.items()},
                }
            )

        model.eval()
        validation_batch = next(
            iter(
                RealWindowBatchStream(
                    sequence_root / dataset / "packed",
                    "validation",
                    int(self.config["window_length"]),
                    int(self.config["batch_size"]),
                )
            )
        )
        validation_batch = {name: value.to(self.device) for name, value in validation_batch.items()}
        with torch.no_grad():
            validation_losses = loss_function(model(validation_batch, graph), validation_batch)
        checkpoint_path = self.output_root / f"{dataset}_smoke_checkpoint.pt"
        torch.save(
            {
                "dataset": dataset,
                "model_config": asdict(model_config),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "steps": len(step_metrics),
            },
            checkpoint_path,
        )
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        return {
            "status": "passed",
            "model_config": asdict(model_config),
            "graph": {
                "nodes": graph.num_nodes,
                "directed_edges": int(graph.edge_index.shape[1]),
                "relations_including_reverse": graph.num_relations,
            },
            "parameters": parameter_count,
            "parameter_memory_mb_float32": parameter_count * 4 / 1_000_000,
            "peak_observed_process_memory_mb": max_memory / 1_000_000,
            "steps": step_metrics,
            "mean_step_seconds": sum(step["seconds"] for step in step_metrics) / len(step_metrics),
            "validation_losses": {name: float(value.cpu()) for name, value in validation_losses.items()},
            "checkpoint": str(checkpoint_path),
        }

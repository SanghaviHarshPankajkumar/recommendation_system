from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = PROJECT_ROOT / "dashboard" / "index.html"
RUN_ROOT = PROJECT_ROOT / "outputs" / "dashboard_runs"
FULL_WINDOW_COUNTS = {
    "oulad": {"train": 136_417, "validation": 42_530},
    "ednet": {"train": 1_172_909, "validation": 452_438},
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _bounded_number(
    values: dict[str, Any], name: str, default: float, minimum: float, maximum: float
) -> float:
    value = float(values.get(name, default))
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


@dataclass
class ExperimentRun:
    run_id: str
    model: str
    dataset: str
    parameters: dict[str, Any]
    status: str = "queued"
    stage: str = "queued"
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    logs: list[str] = field(default_factory=list)
    metrics: list[dict[str, Any]] = field(default_factory=list)
    return_code: int | None = None
    error: str | None = None
    output_dir: str | None = None
    checkpoint: str | None = None
    process: subprocess.Popen[str] | None = field(default=None, repr=False)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def public(self) -> dict[str, Any]:
        with self.lock:
            return {
                "id": self.run_id,
                "model": self.model,
                "dataset": self.dataset,
                "parameters": self.parameters,
                "status": self.status,
                "stage": self.stage,
                "created_at": self.created_at,
                "finished_at": self.finished_at,
                "logs": self.logs[-400:],
                "metrics": self.metrics[-500:],
                "return_code": self.return_code,
                "error": self.error,
                "output_dir": self.output_dir,
                "checkpoint": self.checkpoint,
            }


RUNS: dict[str, ExperimentRun] = {}
RUNS_LOCK = threading.Lock()


def _execute(run: ExperimentRun, stage: str, command: list[str]) -> None:
    with run.lock:
        run.stage = stage
        run.logs.append(f"$ {' '.join(command)}")
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace",
        env=environment,
    )
    with run.lock:
        run.process = process
    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.rstrip()
        with run.lock:
            run.logs.append(line)
            if len(run.logs) > 2000:
                del run.logs[:500]
            if line.startswith("METRIC_JSON:"):
                try:
                    run.metrics.append(json.loads(line.removeprefix("METRIC_JSON:")))
                except json.JSONDecodeError:
                    run.logs.append("Dashboard warning: could not parse metric event")
    return_code = process.wait()
    with run.lock:
        run.process = None
        run.return_code = return_code
    if return_code:
        raise RuntimeError(f"{stage} exited with code {return_code}")


def _run_bc(run: ExperimentRun, run_dir: Path) -> None:
    values = run.parameters
    full_data = str(values.get("data_scope", "bounded")) == "full"
    episodes = 2_147_483_647 if full_data else int(
        _bounded_number(values, "episodes", 50, 1, 1_000_000)
    )
    batch_size = int(_bounded_number(values, "batch_size", 64, 1, 4096))
    learning_rate = _bounded_number(values, "learning_rate", 0.001, 1e-7, 1.0)
    accuracy_rows = int(_bounded_number(values, "accuracy_rows", 512, 1, 100000))
    evaluation_rows = int(_bounded_number(values, "evaluation_rows", 512, 0, 100_000_000))
    seed = int(_bounded_number(values, "seed", 42, 0, 2**31 - 1))

    prep_config = _read_json(PROJECT_ROOT / "configs" / "phase9_d3rlpy.json")
    prep_config["dataset"] = run.dataset
    prep_config["development_max_episodes"] = episodes
    prep_config["evaluation_split"] = str(values.get("evaluation_split", "validation"))
    environment = prep_config["environment"]
    for key, default in (
        ("state_dim", 64), ("max_history", 127), ("min_train_support", 5),
        ("max_episode_steps", 128),
    ):
        environment[key] = int(values.get(key, default))
    for key, default in (("mastery_threshold", 0.7), ("reward_clip", 1.0)):
        environment[key] = float(values.get(key, default))
    environment["enforce_prerequisites"] = bool(values.get("enforce_prerequisites", False))
    environment["avoid_immediate_repeat"] = bool(values.get("avoid_immediate_repeat", False))
    prep_config["paths"]["sequence_root"] = str(PROJECT_ROOT / "outputs" / "phase5_sequences")
    prep_config["paths"]["output_root"] = str(run_dir / "phase9_data")
    prep_path = run_dir / "phase9_prepare.json"
    _write_json(prep_path, prep_config)
    _execute(
        run,
        "Preparing all available episodes" if full_data else f"Preparing {episodes}-episode data",
        [sys.executable, "scripts/prepare_phase9_d3rlpy.py", "--config", str(prep_path)],
    )

    preparation = _read_json(run_dir / "phase9_data" / run.dataset / "validation.json")
    if str(values.get("schedule", "steps")) == "epochs":
        epochs = int(_bounded_number(values, "epochs", 1, 1, 1000))
        steps = ((int(preparation["d3rlpy_transitions"]) + batch_size - 1) // batch_size) * epochs
    else:
        steps = int(_bounded_number(values, "steps", 50, 1, 100_000_000))

    training_config = {
        "paths": {
            "phase9_root": str(run_dir / "phase9_data"),
            "output_root": str(run_dir / "bc_model"),
        },
        "dataset": run.dataset,
        "training": {
            "n_steps": steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "gamma": float(values.get("gamma", 0.99)),
            "beta": float(values.get("beta", 0.5)),
            "accuracy_max_transitions": accuracy_rows,
            "accuracy_batch_size": min(accuracy_rows, 512),
            "metric_interval": int(values.get("metric_interval", 1)),
            "cpu_threads": int(values.get("cpu_threads", 4)),
            "seed": seed,
        },
    }
    training_path = run_dir / "bc_training.json"
    _write_json(training_path, training_config)
    _execute(
        run,
        "Training BC on CPU",
        [
            sys.executable,
            "scripts/train_phase9_bc.py",
            "--config",
            str(training_path),
            "--allow-provisional",
        ],
    )
    checkpoint = run_dir / "bc_model" / run.dataset / "discrete_bc_cpu.d3"
    run.checkpoint = str(checkpoint)

    if bool(values.get("run_validation", True)):
        evaluation_config = _read_json(PROJECT_ROOT / "configs" / "phase9_evaluation.json")
        evaluation_config["dataset"] = run.dataset
        evaluation_config["policy"] = "discrete_bc"
        evaluation_config["paths"]["phase9_root"] = str(run_dir / "phase9_data")
        evaluation_config["paths"]["output_root"] = str(run_dir / "evaluation")
        evaluation_config["bootstrap_replicates"] = int(values.get("bootstrap_replicates", 100))
        evaluation_config["allow_provisional_model_evaluation"] = True
        evaluation_path = run_dir / "bc_evaluation.json"
        _write_json(evaluation_path, evaluation_config)
        evaluation_command = [
            sys.executable,
            "scripts/evaluate_phase9_policy.py",
            "--config",
            str(evaluation_path),
            "--checkpoint",
            str(checkpoint),
            "--allow-provisional",
        ]
        if evaluation_rows > 0:
            evaluation_command.extend(["--max-transitions", str(evaluation_rows)])
        _execute(
            run,
            "Validating BC",
            evaluation_command,
        )
        metrics_path = run_dir / "evaluation" / run.dataset / "discrete_bc" / "metrics.json"
        metrics = _read_json(metrics_path)
        with run.lock:
            run.metrics.append(
                {
                    "model": "bc",
                    "split": "validation",
                    "step": steps,
                    "validation_accuracy": metrics["metrics"]["top1_agreement"]["estimate"],
                    "validation_mrr": metrics["metrics"]["mrr"]["estimate"],
                    "validation_mrr_ci95_low": metrics["metrics"]["mrr"]["ci95_low"],
                    "validation_mrr_ci95_high": metrics["metrics"]["mrr"]["ci95_high"],
                    "validation_hit_rate_5": metrics["metrics"]["hit_rate@5"]["estimate"],
                    "validation_hit_rate_5_ci95_low": metrics["metrics"]["hit_rate@5"]["ci95_low"],
                    "validation_hit_rate_5_ci95_high": metrics["metrics"]["hit_rate@5"]["ci95_high"],
                    "validation_ndcg_10": metrics["metrics"]["ndcg@10"]["estimate"],
                    "validation_ndcg_10_ci95_low": metrics["metrics"]["ndcg@10"]["ci95_low"],
                    "validation_ndcg_10_ci95_high": metrics["metrics"]["ndcg@10"]["ci95_high"],
                    "validation_hit_rate_10": metrics["metrics"]["hit_rate@10"]["estimate"],
                    "evaluated_transitions": metrics["evaluated_transitions"],
                    "evaluated_episodes": metrics["evaluated_episodes"],
                    "evaluation_split": str(values.get("evaluation_split", "validation")),
                    "seed_count": 1,
                    "confidence_interval_method": "episode_cluster_bootstrap_single_seed",
                    "full_held_out_temporal_test": bool(
                        str(values.get("evaluation_split", "validation")) == "test"
                        and evaluation_rows == 0
                    ),
                    "provisional_states": metrics["provisional_states"],
                    "leakage_check_passed": None,
                    "relative_improvement": None,
                    "baselines_compared": [],
                }
            )


def _run_state(run: ExperimentRun, run_dir: Path) -> None:
    values = run.parameters
    config = _read_json(PROJECT_ROOT / "configs" / "phase7_quick_pilot.json")
    config["datasets"] = [run.dataset]
    config["variants"] = [str(values.get("variant", "sequence_only"))]
    config["output_root"] = str(run_dir / "state_model")
    config["use_cuda"] = False
    config["epochs"] = int(_bounded_number(values, "epochs", 1, 1, 100))
    config["batch_size"] = int(_bounded_number(values, "batch_size", 2, 1, 512))
    full_data = str(values.get("data_scope", "bounded")) == "full"
    config["max_train_windows"] = (
        FULL_WINDOW_COUNTS[run.dataset]["train"] if full_data else int(
            _bounded_number(values, "max_train_windows", 50, 1, 2_000_000)
        )
    )
    config["max_validation_windows"] = (
        FULL_WINDOW_COUNTS[run.dataset]["validation"] if full_data else int(
            _bounded_number(values, "max_validation_windows", 25, 1, 1_000_000)
        )
    )
    config["learning_rate"] = _bounded_number(values, "learning_rate", 0.0003, 1e-7, 1.0)
    config["weight_decay"] = _bounded_number(values, "weight_decay", 0.0001, 0.0, 1.0)
    config["cpu_threads"] = int(_bounded_number(values, "cpu_threads", 2, 1, 64))
    config["seed"] = int(_bounded_number(values, "seed", 42, 0, 2**31 - 1))
    total_steps = max(
        1,
        ((config["max_train_windows"] + config["batch_size"] - 1) // config["batch_size"])
        * config["epochs"],
    )
    config["warmup_steps"] = min(int(values.get("warmup_steps", 2)), total_steps - 1)
    config["minimum_lr_ratio"] = float(values.get("minimum_lr_ratio", 0.1))
    config["gradient_clip_norm"] = float(values.get("gradient_clip_norm", 1.0))
    config["early_stopping_patience"] = int(values.get("early_stopping_patience", 2))
    config["early_stopping_min_delta"] = float(values.get("early_stopping_min_delta", 0.0001))
    config["window_length"] = int(values.get("window_length", 128))
    config["max_ranking_examples"] = int(values.get("max_ranking_examples", 1000))
    for key in (
        "state_dim", "num_heads", "transformer_layers", "graph_layers",
        "feedforward_dim", "max_sequence_length",
    ):
        if key in values:
            config["model"][key] = int(values[key])
    if "dropout" in values:
        config["model"]["dropout"] = float(values["dropout"])
    for key in ("item_weight", "action_weight", "correctness_weight", "mastery_weight"):
        if key in values:
            config["loss_weights"][key] = float(values[key])
    config_path = run_dir / "state_training.json"
    _write_json(config_path, config)
    _execute(
        run,
        "Training and validating state model on CPU",
        [
            sys.executable,
            "scripts/run_phase7_pretraining.py",
            "--config",
            str(config_path),
            "--execute",
        ],
    )
    variant = str(config["variants"][0])
    checkpoint = run_dir / "state_model" / run.dataset / variant / "best_checkpoint.pt"
    if checkpoint.exists():
        run.checkpoint = str(checkpoint)


def _worker(run: ExperimentRun) -> None:
    run_dir = RUN_ROOT / run.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    run.output_dir = str(run_dir)
    with run.lock:
        run.status = "running"
    try:
        if run.model == "bc":
            _run_bc(run, run_dir)
        elif run.model == "state":
            _run_state(run, run_dir)
        else:
            raise ValueError("model must be bc or state")
        with run.lock:
            run.status = "complete"
            run.stage = "Complete"
    except Exception as error:
        with run.lock:
            run.status = "failed"
            run.stage = "Failed"
            run.error = str(error)
            run.logs.append(f"Dashboard error: {error}")
    finally:
        with run.lock:
            run.finished_at = time.time()


def _current_results() -> dict[str, Any]:
    result: dict[str, Any] = {"bc": None, "state": []}
    bc_paths = list((PROJECT_ROOT / "outputs" / "phase9_bc_cpu_smoke").glob("*/training_history.json"))
    bc_paths.extend(RUN_ROOT.glob("*/bc_model/*/training_history.json"))
    if bc_paths:
        bc_path = max(bc_paths, key=lambda path: path.stat().st_mtime)
        history = _read_json(bc_path)
        if history.get("history"):
            result["bc"] = {**history["history"][-1], "dataset": history.get("dataset")}
    state_paths = list(
        (PROJECT_ROOT / "outputs").glob("phase7_pretraining*/**/validation_metrics.json")
    )
    state_paths.extend(RUN_ROOT.glob("*/state_model/**/validation_metrics.json"))
    for path in sorted(state_paths, key=lambda item: item.stat().st_mtime):
        try:
            metric = _read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if "correctness" in metric:
            result["state"].append(
                {
                    "path": str(path.relative_to(PROJECT_ROOT)),
                    "correctness_accuracy": metric["correctness"].get("accuracy"),
                    "action_accuracy": metric.get("action_accuracy"),
                    "mrr": metric.get("candidate_ranking", {}).get("mrr"),
                }
            )
    return result


def _restore_completed_runs() -> None:
    if not RUN_ROOT.exists():
        return
    for run_dir in RUN_ROOT.iterdir():
        if not run_dir.is_dir():
            continue
        bc_config_path = run_dir / "bc_training.json"
        state_config_path = run_dir / "state_training.json"
        try:
            if bc_config_path.exists():
                config = _read_json(bc_config_path)
                dataset = str(config["dataset"])
                history_path = run_dir / "bc_model" / dataset / "training_history.json"
                if not history_path.exists():
                    continue
                history = _read_json(history_path)
                run = ExperimentRun(
                    run_id=run_dir.name,
                    model="bc",
                    dataset=dataset,
                    parameters=dict(config["training"]),
                    status="complete",
                    stage="Complete",
                    created_at=run_dir.stat().st_ctime,
                    finished_at=history_path.stat().st_mtime,
                    metrics=[{"model": "bc", **metric} for metric in history.get("history", [])],
                    output_dir=str(run_dir),
                    checkpoint=str(
                        run_dir / "bc_model" / dataset / (
                            "discrete_bc_cpu.d3"
                            if (run_dir / "bc_model" / dataset / "discrete_bc_cpu.d3").exists()
                            else "discrete_bc_cpu_smoke.d3"
                        )
                    ),
                )
                evaluation_path = run_dir / "evaluation" / dataset / "discrete_bc" / "metrics.json"
                if evaluation_path.exists():
                    evaluation = _read_json(evaluation_path)
                    run.metrics.append(
                        {
                            "model": "bc",
                            "split": "validation",
                            "step": history.get("optimizer_steps"),
                            "validation_accuracy": evaluation["metrics"]["top1_agreement"]["estimate"],
                            "validation_mrr": evaluation["metrics"]["mrr"]["estimate"],
                            "validation_mrr_ci95_low": evaluation["metrics"]["mrr"]["ci95_low"],
                            "validation_mrr_ci95_high": evaluation["metrics"]["mrr"]["ci95_high"],
                            "validation_hit_rate_5": evaluation["metrics"]["hit_rate@5"]["estimate"],
                            "validation_hit_rate_5_ci95_low": evaluation["metrics"]["hit_rate@5"]["ci95_low"],
                            "validation_hit_rate_5_ci95_high": evaluation["metrics"]["hit_rate@5"]["ci95_high"],
                            "validation_ndcg_10": evaluation["metrics"]["ndcg@10"]["estimate"],
                            "validation_ndcg_10_ci95_low": evaluation["metrics"]["ndcg@10"]["ci95_low"],
                            "validation_ndcg_10_ci95_high": evaluation["metrics"]["ndcg@10"]["ci95_high"],
                            "evaluated_transitions": evaluation["evaluated_transitions"],
                            "evaluated_episodes": evaluation["evaluated_episodes"],
                            "evaluation_split": "validation",
                            "seed_count": 1,
                            "confidence_interval_method": "episode_cluster_bootstrap_single_seed",
                            "full_held_out_temporal_test": False,
                            "provisional_states": evaluation["provisional_states"],
                            "leakage_check_passed": None,
                            "relative_improvement": None,
                            "baselines_compared": [],
                        }
                    )
                RUNS[run.run_id] = run
            elif state_config_path.exists():
                config = _read_json(state_config_path)
                dataset = str(config["datasets"][0])
                variant = str(config["variants"][0])
                metrics_path = run_dir / "state_model" / dataset / variant / "validation_metrics.json"
                if not metrics_path.exists():
                    continue
                metric = _read_json(metrics_path)
                restored_steps = (
                    (int(config["max_train_windows"]) + int(config["batch_size"]) - 1)
                    // int(config["batch_size"])
                ) * int(config["epochs"])
                RUNS[run_dir.name] = ExperimentRun(
                    run_id=run_dir.name,
                    model="state",
                    dataset=dataset,
                    parameters={"variant": variant},
                    status="complete",
                    stage="Complete",
                    created_at=run_dir.stat().st_ctime,
                    finished_at=metrics_path.stat().st_mtime,
                    metrics=[
                        {
                            "model": "state",
                            "dataset": dataset,
                            "variant": variant,
                            "step": restored_steps,
                            "split": "validation",
                            "validation_correctness_accuracy": metric["correctness"]["accuracy"],
                            "validation_correctness_auc": metric["correctness"]["auc"],
                            "validation_action_accuracy": metric["action_accuracy"],
                            "validation_mrr": metric["candidate_ranking"]["mrr"],
                            "validation_hit_rate_5": metric["candidate_ranking"].get("hit_rate@5"),
                            "validation_ndcg_10": metric["candidate_ranking"].get("ndcg@10"),
                            "evaluation_split": "validation",
                            "seed_count": 1,
                            "full_held_out_temporal_test": False,
                            "provisional_states": False,
                            "leakage_check_passed": None,
                            "relative_improvement": None,
                            "baselines_compared": [],
                        }
                    ],
                    output_dir=str(run_dir),
                    checkpoint=str(run_dir / "state_model" / dataset / variant / "best_checkpoint.pt"),
                )
        except (KeyError, OSError, json.JSONDecodeError):
            continue


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "ExperimentDashboard/1.0"

    def _json(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 1_000_000:
            raise ValueError("Request is too large")
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            body = INDEX_PATH.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/status":
            self._json(_current_results())
            return
        if path == "/api/runs":
            with RUNS_LOCK:
                runs = sorted(RUNS.values(), key=lambda item: item.created_at, reverse=True)
            self._json([run.public() for run in runs])
            return
        if path.startswith("/api/runs/"):
            run_id = path.rsplit("/", 1)[-1]
            with RUNS_LOCK:
                run = RUNS.get(run_id)
            if run is None:
                self._json({"error": "Run not found"}, HTTPStatus.NOT_FOUND)
            else:
                self._json(run.public())
            return
        self._json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/runs":
            self._json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            return
        try:
            values = self._body()
            model = str(values.get("model", "bc"))
            dataset = str(values.get("dataset", "oulad"))
            if model not in {"bc", "state"}:
                raise ValueError("model must be bc or state")
            if dataset not in {"oulad", "ednet"}:
                raise ValueError("dataset must be oulad or ednet")
            run = ExperimentRun(
                run_id=uuid.uuid4().hex[:10],
                model=model,
                dataset=dataset,
                parameters=dict(values.get("parameters", {})),
            )
            with RUNS_LOCK:
                RUNS[run.run_id] = run
            threading.Thread(target=_worker, args=(run,), daemon=True).start()
            self._json(run.public(), HTTPStatus.ACCEPTED)
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

    def log_message(self, format: str, *args: Any) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Local UI for CPU model experiments")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if not INDEX_PATH.exists():
        raise FileNotFoundError(INDEX_PATH)
    _restore_completed_runs()
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"Experiment dashboard running at {url}", flush=True)
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

from edu_recommender.phase7_pretraining import Phase7PretrainingPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate or execute Phase 7 supervised pretraining")
    parser.add_argument("--config", type=Path, default=Path("configs/phase7_pretraining.json"))
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually run optimization. Without this flag, only validate the configuration and inputs.",
    )
    args = parser.parse_args()
    def print_progress(metric: dict[str, object]) -> None:
        print(f"METRIC_JSON:{json.dumps(metric)}", flush=True)

    pipeline = Phase7PretrainingPipeline.from_json(args.config)
    if args.execute:
        pipeline.progress_callback = print_progress
    result = pipeline.run() if args.execute else pipeline.validate_only()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

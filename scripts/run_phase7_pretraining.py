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
    pipeline = Phase7PretrainingPipeline.from_json(args.config)
    result = pipeline.run() if args.execute else pipeline.validate_only()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from edu_recommender.full_preprocessing import (  # noqa: E402
    FullScalePreprocessingPipeline,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run resumable bounded-memory full-scale preprocessing"
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "full_preprocessing.json"),
    )
    parser.add_argument(
        "--dataset",
        choices=["all", "ednet", "oulad"],
        default="all",
    )
    args = parser.parse_args()
    result = FullScalePreprocessingPipeline(args.config).run(args.dataset)
    summary = {
        key: {
            "status": value.get("status"),
            "events": value.get("events"),
            "units": value.get("units"),
        }
        for key, value in result.items()
        if isinstance(value, dict)
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()


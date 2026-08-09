from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from edu_recommender.pipeline import Phase13Pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Objective 1 phases 1-3")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "phase1_3.json"),
        help="Path to the JSON configuration",
    )
    args = parser.parse_args()
    summary = Phase13Pipeline(args.config).run()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

from edu_recommender.phase6_training import Phase6SmokeTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 6 real-data model smoke benchmark")
    parser.add_argument("--config", type=Path, default=Path("configs/phase6_model.json"))
    args = parser.parse_args()
    print(json.dumps(Phase6SmokeTrainer.from_json(args.config).run(), indent=2))


if __name__ == "__main__":
    main()

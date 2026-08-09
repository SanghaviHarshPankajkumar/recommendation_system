from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from edu_recommender.sequence_building import Phase5SequencePipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-safe Phase 5 packed sequences")
    parser.add_argument("--config", type=Path, default=Path("configs/phase5_sequences.json"))
    args = parser.parse_args()
    print(json.dumps(Phase5SequencePipeline.from_json(args.config).run(), indent=2))


if __name__ == "__main__":
    main()

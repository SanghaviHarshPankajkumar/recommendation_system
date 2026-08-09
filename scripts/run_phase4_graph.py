from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from edu_recommender.knowledge_graph import Phase4GraphPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and validate Phase 4 knowledge graphs")
    parser.add_argument("--config", type=Path, default=Path("configs/phase4_graph.json"))
    args = parser.parse_args()
    manifest = Phase4GraphPipeline.from_json(args.config).run()
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

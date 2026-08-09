from __future__ import annotations

import gzip
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from edu_recommender.knowledge_graph import (
    EDGE_COLUMNS,
    NODE_COLUMNS,
    EdNetPrerequisiteInference,
    GraphValidator,
    PrerequisiteThresholds,
)


class GraphValidatorTests(unittest.TestCase):
    def test_detects_cycle(self) -> None:
        nodes = pd.DataFrame(
            [["a", "skill", "test", "a", "{}"], ["b", "skill", "test", "b", "{}"]],
            columns=NODE_COLUMNS,
        )
        edges = pd.DataFrame(
            [
                ["a", "b", "candidate", "test", "unit", 1.0, "{}"],
                ["b", "a", "candidate", "test", "unit", 1.0, "{}"],
            ],
            columns=EDGE_COLUMNS,
        )
        with self.assertRaises(ValueError):
            GraphValidator.validate(nodes, edges, require_dag=True)


class PrerequisiteInferenceTests(unittest.TestCase):
    def test_train_sequence_metrics_and_dag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            train_dir = Path(directory)
            frame = pd.DataFrame(
                {
                    "student_id": ["s1", "s1", "s2", "s2", "s3", "s3", "s4", "s4"],
                    "concept_ids": ["1", "2", "1", "2", "1", "2", "2", "1"],
                    "correctness": [1, 1, 1, 1, 0, 0, 1, 1],
                    "final_response": [True] * 8,
                }
            )
            with gzip.open(train_dir / "part-00000.csv.gz", "wt", encoding="utf-8", newline="") as handle:
                frame.to_csv(handle, index=False)
            inference = EdNetPrerequisiteInference(
                train_dir,
                PrerequisiteThresholds(
                    min_transition_support=2,
                    min_conditional_support=1,
                    min_direction_confidence=0.6,
                    min_performance_lift=0.5,
                ),
                chunksize=3,
            )
            candidates, dag, stats = inference.infer(["ednet:skill:1", "ednet:skill:2"])
            forward = candidates[
                (candidates.source_skill_id == "ednet:skill:1")
                & (candidates.target_skill_id == "ednet:skill:2")
            ].iloc[0]
            self.assertEqual(forward.transition_support, 3)
            self.assertAlmostEqual(forward.direction_confidence, 0.75)
            self.assertAlmostEqual(forward.performance_lift, 1.0)
            self.assertEqual(len(dag), 1)
            self.assertEqual(stats["learner_sequences"], 4)


if __name__ == "__main__":
    unittest.main()

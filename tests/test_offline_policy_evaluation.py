from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from edu_recommender.offline_policy_evaluation import (
    EvaluationError,
    OfflineEvaluationDataset,
    OfflinePolicyEvaluator,
    PopularityPolicyScorer,
)


class OfflinePolicyEvaluationTests(unittest.TestCase):
    def _dataset(self) -> OfflineEvaluationDataset:
        return OfflineEvaluationDataset(
            observations=np.zeros((3, 4), dtype=np.float32),
            actions=np.asarray([0, 0, 1], dtype=np.int64),
            rewards=np.asarray([0.2, 0.3, -0.1], dtype=np.float32),
            terminals=np.asarray([0, 1, 1], dtype=np.float32),
            timeouts=np.zeros(3, dtype=np.float32),
            eligible_actions=(
                np.asarray([0, 1, 2], dtype=np.int32),
                np.asarray([0, 1, 2], dtype=np.int32),
                np.asarray([0, 1, 2], dtype=np.int32),
            ),
            episode_ids=np.asarray(["a", "a", "b"]),
            action_size=3,
            dataset_name="ednet",
            provisional_states=False,
        )

    def test_popularity_ranking_and_safety_metrics(self) -> None:
        dataset = self._dataset()
        scorer = PopularityPolicyScorer(dataset.actions, dataset.action_size)
        result = OfflinePolicyEvaluator(
            ks=(1, 2), bootstrap_replicates=0, seed=7
        ).evaluate(dataset, scorer)
        self.assertAlmostEqual(result["metrics"]["top1_agreement"]["estimate"], 2 / 3)
        self.assertAlmostEqual(result["metrics"]["hit_rate@1"]["estimate"], 2 / 3)
        self.assertAlmostEqual(result["metrics"]["mrr"]["estimate"], 5 / 6)
        self.assertEqual(result["safety"]["masked_eligible_action_rate"], 1.0)
        self.assertEqual(result["safety"]["raw_eligible_action_rate"], 1.0)
        self.assertAlmostEqual(result["coverage"]["catalog_action_coverage"], 1 / 3)
        self.assertFalse(result["limitations"]["fqe_included"])

    def test_episode_cluster_bootstrap_is_reproducible(self) -> None:
        dataset = self._dataset()
        scorer = PopularityPolicyScorer(dataset.actions, dataset.action_size)
        evaluator = OfflinePolicyEvaluator(ks=(2,), bootstrap_replicates=50, seed=42)
        first = evaluator.evaluate(dataset, scorer)
        second = evaluator.evaluate(dataset, scorer)
        self.assertEqual(first["metrics"], second["metrics"])

    def test_rejects_logged_action_missing_from_mask(self) -> None:
        with self.assertRaises(ValueError):
            OfflineEvaluationDataset(
                observations=np.zeros((1, 2), dtype=np.float32),
                actions=np.asarray([1]),
                rewards=np.asarray([0.0], dtype=np.float32),
                terminals=np.asarray([1.0], dtype=np.float32),
                timeouts=np.asarray([0.0], dtype=np.float32),
                eligible_actions=(np.asarray([0], dtype=np.int32),),
                episode_ids=np.asarray(["x"]),
                action_size=2,
                dataset_name="ednet",
                provisional_states=False,
            )

    def test_loader_requires_sparse_eligibility_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez_compressed(
                root / "dataset.npz",
                observations=np.zeros((1, 2), dtype=np.float32),
                actions=np.asarray([0]),
                rewards=np.asarray([0.0], dtype=np.float32),
                terminals=np.asarray([1.0], dtype=np.float32),
                timeouts=np.asarray([0.0], dtype=np.float32),
            )
            (root / "metadata.json").write_text(
                '{"action_size": 1, "dataset": "ednet"}', encoding="utf-8"
            )
            with self.assertRaises(EvaluationError):
                OfflineEvaluationDataset.load(root / "dataset.npz", root / "metadata.json")


if __name__ == "__main__":
    unittest.main()

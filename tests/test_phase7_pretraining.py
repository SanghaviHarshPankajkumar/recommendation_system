from __future__ import annotations

import unittest

import numpy as np
import torch

from edu_recommender.phase7_pretraining import (
    BinaryMetrics,
    EarlyStopping,
    ValidationMetrics,
    WarmupCosineSchedule,
)


class Phase7UtilityTests(unittest.TestCase):
    def test_warmup_cosine_schedule(self) -> None:
        schedule = WarmupCosineSchedule(warmup_steps=2, total_steps=10, min_ratio=0.1)
        self.assertAlmostEqual(schedule(0), 0.5)
        self.assertAlmostEqual(schedule(1), 1.0)
        self.assertAlmostEqual(schedule(2), 1.0)
        self.assertAlmostEqual(schedule(10), 0.1)

    def test_early_stopping(self) -> None:
        stopper = EarlyStopping(patience=2, min_delta=0.01)
        self.assertEqual(stopper.update(1.0), (True, False))
        self.assertEqual(stopper.update(0.995), (False, False))
        self.assertEqual(stopper.update(0.994), (False, True))
        self.assertEqual(stopper.update(0.8), (True, False))

    def test_binary_metrics_known_auc_and_calibration(self) -> None:
        metrics = BinaryMetrics(calibration_bins=2)
        metrics.update(torch.tensor([0.1, 0.4, 0.35, 0.8]), torch.tensor([0, 0, 1, 1]))
        result = metrics.compute()
        self.assertAlmostEqual(result["auc"], 0.75)
        self.assertAlmostEqual(result["accuracy"], 0.75)
        self.assertTrue(np.isfinite(result["ece"]))

    def test_validation_uses_known_module_and_task_specific_loss_counts(self) -> None:
        class CaptureRanking:
            def __init__(self) -> None:
                self.modules = None

            def update_model(self, logits, targets, modules, target_mask) -> None:
                self.modules = modules.clone()

            def compute(self) -> dict[str, float]:
                return {"mrr": 0.0}

        ranking = CaptureRanking()
        metrics = ValidationMetrics(
            ranking,
            {"item": 1.0, "action": 0.25, "correctness": 0.5, "mastery": 0.5},
        )
        batch = {
            "target_mask": torch.tensor([[1, 1]], dtype=torch.bool),
            "target_correctness": torch.tensor([[1.0, -1.0]]),
            "target_concept_tokens": torch.tensor([[[2, 1], [3, 0]]]),
            "target_action_tokens": torch.tensor([[1, 2]]),
            "target_item_tokens": torch.tensor([[2, 3]]),
            "module_tokens": torch.tensor([[7, 8, 9]]),
        }
        outputs = {
            "correctness_logits": torch.tensor([[2.0, -2.0]]),
            "mastery_probabilities": torch.full((1, 2, 4), 0.6),
            "action_logits": torch.tensor([[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]),
            "item_logits": torch.zeros((1, 2, 5)),
        }
        losses = {name: torch.tensor(value) for name, value in {
            "total": 9.0, "item": 1.0, "action": 2.0, "correctness": 3.0, "mastery": 4.0,
        }.items()}
        metrics.update(outputs, losses, batch)
        result = metrics.compute()
        self.assertTrue(torch.equal(ranking.modules, torch.tensor([[7, 8]])))
        self.assertEqual(result["loss_counts"], {"item": 2, "action": 2, "correctness": 1, "mastery": 1})
        self.assertAlmostEqual(result["losses"]["total"], 1.0 + 0.25 * 2.0 + 0.5 * 3.0 + 0.5 * 4.0)


if __name__ == "__main__":
    unittest.main()

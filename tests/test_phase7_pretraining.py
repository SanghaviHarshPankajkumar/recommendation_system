from __future__ import annotations

import unittest

import numpy as np
import torch

from edu_recommender.phase7_pretraining import BinaryMetrics, EarlyStopping, WarmupCosineSchedule


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


if __name__ == "__main__":
    unittest.main()

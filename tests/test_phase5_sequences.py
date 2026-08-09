from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from edu_recommender.sequence_building import (  # noqa: E402
    CategoricalVocabulary,
    GraphAwareCandidateProvider,
    LazySequenceWindows,
    PackedSequenceValidator,
    PackedSequenceWriter,
    SPLIT_IDS,
    WindowConfig,
)


class PackedSequenceTests(unittest.TestCase):
    def setUp(self) -> None:
        values = {
            "item_id": ["i1", "i2", "i3"],
            "action_type": ["respond"],
            "item_type": ["question"],
            "concept_ids": ["c1", "c2"],
            "module_id": ["m1"],
            "source": ["s"],
        }
        self.vocab = {field: CategoricalVocabulary(items) for field, items in values.items()}

    def test_targets_are_covered_once_and_keep_prior_context(self) -> None:
        frame = pd.DataFrame(
            {
                "student_id": ["u1"] * 5,
                "timestamp": [1, 2, 3, 4, 5],
                "relative_day": [np.nan] * 5,
                "item_id": ["i1", "i2", "i3", "i1", "i2"],
                "item_type": ["question"] * 5,
                "concept_ids": ["c1", "c1;c2", "c2", "c1", "c2"],
                "action_type": ["respond"] * 5,
                "correctness": [0, 1, 1, 0, 1],
                "score": [np.nan] * 5,
                "elapsed_time_ms": [np.nan] * 5,
                "engagement": [1] * 5,
                "source": ["s"] * 5,
                "module_id": ["m1"] * 5,
                "final_response": [True] * 5,
                "is_banked": [np.nan] * 5,
                "_split_id": [0, 0, 0, 1, 2],
            }
        )
        writer = PackedSequenceWriter("ednet", Path("."), Path("."), self.vocab, WindowConfig(4, 2))
        arrays = writer._encode(frame)
        validation = PackedSequenceValidator.validate(arrays, WindowConfig(4, 2))
        self.assertEqual(validation["target_counts"], {"train": 2, "validation": 1, "test": 1})
        self.assertEqual(len(arrays["concept_values"]), 6)
        validation_window = np.flatnonzero(arrays["window_split_ids"] == SPLIT_IDS["validation"])[0]
        self.assertLess(arrays["window_starts"][validation_window], 3)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.npz"
            np.savez_compressed(path, **arrays)
            sample = next(LazySequenceWindows(path, 4).iter_split("validation"))
            self.assertEqual(sample["target_mask"].sum(), 1)
            self.assertEqual(len(sample["input_item_tokens"]), 3)
            self.assertEqual(sample["input_attention_mask"].shape, (3,))
            self.assertEqual(sample["input_concept_tokens"].shape, (3, 9))

    def test_unknown_token_is_not_added_after_fit(self) -> None:
        self.assertEqual(self.vocab["item_id"].encode_one("validation_only"), 1)


class CandidateProviderTests(unittest.TestCase):
    def test_support_module_and_prerequisite_filters(self) -> None:
        catalog = pd.DataFrame(
            {
                "item_token": [2, 3, 4],
                "module_id": ["m1", "m1", "m2"],
                "train_support": [10, 2, 10],
                "prerequisite_ids_json": ['["skill:a"]', "[]", "[]"],
            }
        )
        provider = GraphAwareCandidateProvider(catalog, min_train_support=5)
        self.assertEqual(provider.eligible("m1").tolist(), [2])
        self.assertEqual(provider.eligible("m1", set(), True).tolist(), [])
        self.assertEqual(provider.eligible("m1", {"skill:a"}, True).tolist(), [2])


if __name__ == "__main__":
    unittest.main()

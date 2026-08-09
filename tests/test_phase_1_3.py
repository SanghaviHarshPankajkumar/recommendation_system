from __future__ import annotations

import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from edu_recommender.schema import COMMON_EVENT_COLUMNS, ensure_common_schema  # noqa: E402
from edu_recommender.preprocess_ednet import EdNetPreprocessor  # noqa: E402
from edu_recommender.preprocess_oulad import OULADPreprocessor  # noqa: E402
from edu_recommender.partitioned import PartitionValidator, PartitionedCsvWriter  # noqa: E402
from edu_recommender.splitting import TemporalSplitter  # noqa: E402


class SchemaTests(unittest.TestCase):
    def test_schema_adds_all_columns(self) -> None:
        frame = ensure_common_schema(
            pd.DataFrame(
                {
                    "dataset": ["test"],
                    "student_id": ["s1"],
                    "timestamp": [1],
                    "item_id": ["i1"],
                }
            )
        )
        self.assertEqual(list(frame.columns), COMMON_EVENT_COLUMNS)


class TemporalSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.splitter = TemporalSplitter(0.7, 0.15, 0.15, 5)

    def test_split_preserves_order(self) -> None:
        events = ensure_common_schema(
            pd.DataFrame(
                {
                    "dataset": ["test"] * 10,
                    "student_id": ["s1"] * 10,
                    "timestamp": list(range(10)),
                    "item_id": [f"i{x}" for x in range(10)],
                }
            )
        )
        result = self.splitter.split(events)
        self.assertEqual(result["split"].tolist()[:7], ["train"] * 7)
        self.assertEqual(result["split"].tolist()[7], "validation")
        self.assertEqual(result["split"].tolist()[8:], ["test", "test"])

    def test_short_history_stays_in_training(self) -> None:
        events = ensure_common_schema(
            pd.DataFrame(
                {
                    "dataset": ["test"] * 3,
                    "student_id": ["s1"] * 3,
                    "timestamp": [1, 2, 3],
                    "item_id": ["i1", "i2", "i3"],
                }
            )
        )
        result = self.splitter.split(events)
        self.assertEqual(set(result["split"]), {"train"})


class ClassApiTests(unittest.TestCase):
    def test_preprocessors_expose_run_method(self) -> None:
        self.assertTrue(callable(getattr(EdNetPreprocessor, "run")))
        self.assertTrue(callable(getattr(OULADPreprocessor, "run")))

    def test_ednet_only_marks_responses_followed_by_matching_submit(self) -> None:
        preprocessor = EdNetPreprocessor(
            PROJECT_ROOT / "datasets" / "ednet" / "EdNet-KT3.zip",
            PROJECT_ROOT / "datasets" / "ednet" / "metadata" / "contents" / "questions.csv",
            PROJECT_ROOT / "datasets" / "ednet" / "metadata" / "contents" / "lectures.csv",
            max_users=1,
        )
        frame = pd.DataFrame(
            {
                "timestamp": [1, 2, 3, 4],
                "action_type": ["respond", "respond", "submit", "respond"],
                "item_id": ["q2319", "q2319", "b1707", "q2322"],
            }
        )
        final = preprocessor._mark_final_responses(frame)
        self.assertEqual(final.tolist(), [False, True, False, False])

    def test_ednet_pairs_enter_and_quit_duration(self) -> None:
        frame = pd.DataFrame(
            {
                "timestamp": [1000, 1500, 3000],
                "action_type": ["enter", "quit", "quit"],
                "item_id": ["l1", "l1", "l1"],
            }
        )
        duration = EdNetPreprocessor._elapsed_time(frame)
        self.assertEqual(float(duration.iloc[1]), 500.0)
        self.assertTrue(pd.isna(duration.iloc[2]))

    def test_ednet_batch_parser_matches_individual_parser(self) -> None:
        preprocessor = EdNetPreprocessor(
            PROJECT_ROOT / "datasets" / "ednet" / "EdNet-KT3.zip",
            PROJECT_ROOT / "datasets" / "ednet" / "metadata" / "contents" / "questions.csv",
            PROJECT_ROOT / "datasets" / "ednet" / "metadata" / "contents" / "lectures.csv",
            max_users=10,
        )
        with zipfile.ZipFile(preprocessor.zip_path) as archive:
            members = preprocessor.list_members(archive)[:10]
            batch_frames, batch_missing = preprocessor.normalize_members(archive, members)
            individual_results = [
                preprocessor.normalize_member(archive, member) for member in members
            ]
        batch = pd.concat(batch_frames, ignore_index=True)
        individual = pd.concat(
            [frame for frame, _ in individual_results if not frame.empty],
            ignore_index=True,
        )
        order = ["student_id", "timestamp", "item_id", "action_type"]
        batch = batch.sort_values(order, kind="stable").reset_index(drop=True)
        individual = individual.sort_values(order, kind="stable").reset_index(drop=True)
        pd.testing.assert_frame_equal(batch, individual, check_dtype=False)
        self.assertEqual(batch_missing, sum(missing for _, missing in individual_results))


class PartitionWriterTests(unittest.TestCase):
    def test_writer_can_resume_and_validate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            events = ensure_common_schema(
                pd.DataFrame(
                    {
                        "dataset": ["test"] * 6,
                        "student_id": ["s1"] * 6,
                        "timestamp": list(range(6)),
                        "item_id": [f"i{x}" for x in range(6)],
                    }
                )
            )
            events["split"] = ["train"] * 4 + ["validation", "test"]
            writer = PartitionedCsvWriter(temporary, "test", {"version": 1})
            writer.write(events, {"offset": 6}, units=1)
            resumed = PartitionedCsvWriter(temporary, "test", {"version": 1})
            self.assertEqual(resumed.progress["offset"], 6)
            validation = PartitionValidator(temporary, "test").run()
            self.assertTrue(validation["passed"])
            self.assertEqual(validation["events"], 6)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from edu_recommender.offline_rl_environment import (
    ActionCatalog,
    CounterfactualActionError,
    EdNetOfflineEnv,
    EncodedObservation,
    OfflineEpisode,
    OfflineStep,
    OULADOfflineEnv,
    PackedEpisodeBuilder,
    ProvisionalStateEncoder,
    RewardWeights,
    TorchStudentStateEncoder,
    EnvironmentSettings,
    MasteryOrientedReward,
)


def candidate_frame(dataset: str) -> pd.DataFrame:
    module_ids = ["", ""] if dataset == "ednet" else ["AAA:2013J", "BBB:2013J"]
    module_tokens = [1, 1] if dataset == "ednet" else [2, 3]
    return pd.DataFrame(
        {
            "item_id": [f"{dataset}:item:a", f"{dataset}:item:b"],
            "item_type": ["question", "lecture"] if dataset == "ednet" else ["vle_activity", "assessment"],
            "module_id": module_ids,
            "train_support": [10, 12],
            "item_token": [10, 11],
            "module_token": module_tokens,
            "prerequisite_ids_json": ['["ednet:skill:1"]', "[]"] if dataset == "ednet" else ["[]", "[]"],
        }
    )


def observation(eligible: list[int], module: int = 1) -> EncodedObservation:
    return EncodedObservation(
        student_state=np.zeros(64, dtype=np.float32),
        mastery=np.asarray([0.0, 0.0, 0.8], dtype=np.float32),
        recent_features=np.zeros(8, dtype=np.float32),
        module=module,
        eligible_actions=np.asarray(eligible, dtype=np.int32),
    )


class OfflineEnvironmentTests(unittest.TestCase):
    def _environment(self, dataset: str):
        catalog = ActionCatalog(dataset, candidate_frame(dataset), ["<PAD>", "<UNK>", "1"])
        module = 1 if dataset == "ednet" else 2
        step = OfflineStep(
            observation=observation([0, 1] if dataset == "ednet" else [0], module),
            logged_action=0,
            reward=0.25,
            next_observation=observation([0, 1] if dataset == "ednet" else [0], module),
            terminated=True,
            truncated=False,
            info={"logged_action_forced_eligible": False},
        )
        episode = OfflineEpisode("student:episode", (step,))
        environment_type = EdNetOfflineEnv if dataset == "ednet" else OULADOfflineEnv
        return environment_type(
            episodes=[episode],
            action_catalog=catalog,
            state_dim=64,
            mastery_dim=3,
            module_count=4,
        )

    def test_ednet_logged_replay_and_spaces(self) -> None:
        environment = self._environment("ednet")
        state, info = environment.reset(options={"episode_index": 0})
        self.assertTrue(environment.observation_space.contains(state))
        self.assertEqual(info["logged_action"], 0)
        next_state, reward, terminated, truncated, _ = environment.step(0)
        self.assertTrue(environment.observation_space.contains(next_state))
        self.assertEqual(reward, 0.25)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(environment.validate_dataset()["passed"])

    def test_oulad_module_mask(self) -> None:
        environment = self._environment("oulad")
        state, _ = environment.reset(options={"episode_index": 0})
        self.assertEqual(state["action_mask"].tolist(), [1, 0])

    def test_counterfactual_action_is_rejected(self) -> None:
        environment = self._environment("ednet")
        environment.reset(options={"episode_index": 0})
        with self.assertRaises(CounterfactualActionError):
            environment.step(1)

    def test_logged_replay_has_an_action_free_api(self) -> None:
        environment = self._environment("ednet")
        environment.reset(options={"episode_index": 0})
        _, reward, terminated, truncated, _ = environment.replay_logged_step()
        self.assertEqual(reward, 0.25)
        self.assertTrue(terminated)
        self.assertFalse(truncated)

    def test_dataset_specific_environment_rejects_wrong_catalog(self) -> None:
        catalog = ActionCatalog("oulad", candidate_frame("oulad"), ["<PAD>", "<UNK>", "1"])
        with self.assertRaises(ValueError):
            EdNetOfflineEnv(
                episodes=[OfflineEpisode("x", (OfflineStep(observation([0]), 0, 0.0, observation([0]), True, False, {}),))],
                action_catalog=catalog,
                state_dim=64,
                mastery_dim=3,
                module_count=4,
            )


class CandidateAndEncoderTests(unittest.TestCase):
    def test_ednet_question_decision_precedes_response_block(self) -> None:
        builder = PackedEpisodeBuilder.__new__(PackedEpisodeBuilder)
        builder.settings = EnvironmentSettings(dataset="ednet")
        builder.item_types = ["<PAD>", "<UNK>", "bundle", "question"]
        arrays = {
            "item_tokens": np.asarray([20, 10, 10, 20], dtype=np.int32),
            "item_type_tokens": np.asarray([2, 3, 3, 2], dtype=np.int16),
        }
        self.assertEqual(builder._decision_index(arrays, 2, 0), 1)

    def test_prerequisite_and_repeat_filtering(self) -> None:
        catalog = ActionCatalog("ednet", candidate_frame("ednet"), ["<PAD>", "<UNK>", "1"])
        not_mastered = np.asarray([0.0, 0.0, 0.2], dtype=np.float32)
        eligible = catalog.eligible_actions(1, not_mastered, 0.7, True, 11, True)
        self.assertEqual(eligible.tolist(), [])
        mastered = np.asarray([0.0, 0.0, 0.9], dtype=np.float32)
        eligible = catalog.eligible_actions(1, mastered, 0.7, True, 11, True)
        self.assertEqual(eligible.tolist(), [0])

    def test_provisional_encoder_is_future_safe(self) -> None:
        arrays = {
            "correctness": np.asarray([1, 0, 1], dtype=np.int8),
            "scores": np.asarray([np.nan, np.nan, np.nan], dtype=np.float32),
            "engagement_log1p": np.asarray([1.0, 1.0, 99.0], dtype=np.float32),
            "elapsed_log1p": np.asarray([1.0, 1.0, 99.0], dtype=np.float32),
            "time_gaps": np.asarray([0.0, 1.0, 99.0], dtype=np.float32),
            "relative_days": np.asarray([0.0, 1.0, 99.0], dtype=np.float32),
            "final_response": np.asarray([1, 1, 1], dtype=np.uint8),
            "concept_offsets": np.asarray([0, 1, 2, 3], dtype=np.int64),
            "concept_values": np.asarray([2, 2, 2], dtype=np.int32),
        }
        encoder = ProvisionalStateEncoder(64, 3)
        state_before, mastery_before, _ = encoder.encode(arrays, 0, 2)
        arrays["correctness"][2] = 0
        arrays["engagement_log1p"][2] = -99
        state_after, mastery_after, _ = encoder.encode(arrays, 0, 2)
        np.testing.assert_array_equal(state_before, state_after)
        np.testing.assert_array_equal(mastery_before, mastery_after)

    def test_state_ranges_are_deduplicated_and_encoded_in_one_batch(self) -> None:
        class CountingEncoder:
            state_dim = 64
            mastery_dim = 3
            name = "counting"

            def __init__(self) -> None:
                self.calls = 0
                self.ranges = []

            def encode_many(self, arrays, ranges):
                del arrays
                self.calls += 1
                self.ranges = list(ranges)
                return [
                    (
                        np.zeros(64, dtype=np.float32),
                        np.zeros(3, dtype=np.float32),
                        np.zeros(8, dtype=np.float32),
                    )
                    for _ in ranges
                ]

        builder = PackedEpisodeBuilder.__new__(PackedEpisodeBuilder)
        builder.settings = EnvironmentSettings(dataset="oulad", max_history=127)
        builder.encoder = CountingEncoder()
        cache = builder._encoding_cache({}, 0, [1, 2])
        self.assertEqual(builder.encoder.calls, 1)
        self.assertEqual(builder.encoder.ranges, [(0, 1), (0, 2), (0, 3)])
        self.assertEqual(set(cache), {(0, 1), (0, 2), (0, 3)})

    def test_trajectory_continuity_is_preserved_by_default(self) -> None:
        builder = PackedEpisodeBuilder.__new__(PackedEpisodeBuilder)
        builder.settings = EnvironmentSettings(dataset="oulad", max_episode_steps=2)
        self.assertEqual(builder._candidate_chunks([1, 2, 3]), [([1, 2, 3], False)])
        builder.settings = EnvironmentSettings(
            dataset="oulad",
            max_episode_steps=2,
            preserve_trajectory_continuity=False,
        )
        self.assertEqual(
            builder._candidate_chunks([1, 2, 3]),
            [([1, 2], True), ([3], False)],
        )

    def test_torch_encoder_builds_a_padded_batch(self) -> None:
        encoder = TorchStudentStateEncoder.__new__(TorchStudentStateEncoder)
        encoder.max_concepts_per_event = 2
        encoder.device = __import__("torch").device("cpu")
        arrays = {
            name: np.arange(4, dtype=dtype)
            for name, dtype in {
                "item_tokens": np.int32,
                "action_tokens": np.int16,
                "item_type_tokens": np.int16,
                "module_tokens": np.int16,
                "source_tokens": np.int16,
                "time_gaps": np.float32,
                "elapsed_log1p": np.float32,
                "engagement_log1p": np.float32,
                "scores": np.float32,
                "relative_days": np.float32,
                "correctness": np.int8,
            }.items()
        }
        arrays["concept_offsets"] = np.asarray([0, 1, 2, 3, 4], dtype=np.int64)
        arrays["concept_values"] = np.asarray([2, 3, 2, 3], dtype=np.int32)
        batch, lengths = encoder._batch_many(arrays, [(0, 2), (1, 4)])
        self.assertEqual(lengths, [2, 3])
        self.assertEqual(tuple(batch["input_item_tokens"].shape), (2, 3))
        self.assertEqual(
            batch["input_attention_mask"].tolist(),
            [[True, True, False], [True, True, True]],
        )

    def test_reward_prefers_learning_opportunity_over_mastered_easy_item(self) -> None:
        catalog = ActionCatalog("ednet", candidate_frame("ednet"), ["<PAD>", "<UNK>", "1"])
        settings = EnvironmentSettings(
            dataset="ednet",
            reward_weights=RewardWeights(),
        )
        reward = MasteryOrientedReward(settings, catalog)
        arrays = {
            "concept_offsets": np.asarray([0, 1], dtype=np.int64),
            "concept_values": np.asarray([2], dtype=np.int32),
            "correctness": np.asarray([1], dtype=np.int8),
            "scores": np.asarray([100.0], dtype=np.float32),
            "engagement_log1p": np.asarray([0.0], dtype=np.float32),
            "elapsed_log1p": np.asarray([0.0], dtype=np.float32),
        }
        medium = observation([0, 1])
        mastered = observation([0, 1])
        medium.mastery[2] = 0.5
        mastered.mastery[2] = 0.99
        medium_reward, _ = reward.calculate(arrays, 0, medium, medium, 1, False, False)
        mastered_reward, _ = reward.calculate(arrays, 0, mastered, mastered, 1, False, False)
        self.assertGreater(medium_reward, mastered_reward)
        self.assertLessEqual(RewardWeights().correctness + RewardWeights().score, 0.05)


if __name__ == "__main__":
    unittest.main()

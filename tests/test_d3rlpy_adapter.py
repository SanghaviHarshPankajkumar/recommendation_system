from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from edu_recommender.d3rlpy_adapter import (
    D3RLPYTransitionAdapter,
    MaskedD3RLPYActionSelector,
    create_d3rlpy_algorithm,
)
from edu_recommender.offline_rl_environment import EncodedObservation, OfflineEpisode, OfflineStep


def observation(eligible: list[int], module: int) -> EncodedObservation:
    return EncodedObservation(
        student_state=np.arange(4, dtype=np.float32),
        mastery=np.asarray([0.0, 0.0, 0.8], dtype=np.float32),
        recent_features=np.arange(8, dtype=np.float32),
        module=module,
        eligible_actions=np.asarray(eligible, dtype=np.int32),
    )


class D3RLPYAdapterTests(unittest.TestCase):
    def _episode(self) -> OfflineEpisode:
        first = OfflineStep(
            observation=observation([0, 2], 1),
            logged_action=2,
            reward=0.25,
            next_observation=observation([1, 2], 1),
            terminated=False,
            truncated=False,
            info={"encoder": "selected_phase7_checkpoint"},
        )
        second = OfflineStep(
            observation=observation([1, 2], 1),
            logged_action=1,
            reward=-0.1,
            next_observation=observation([0, 1], 1),
            terminated=True,
            truncated=False,
            info={"encoder": "selected_phase7_checkpoint"},
        )
        return OfflineEpisode("student:0", (first, second))

    def test_conversion_builds_native_discrete_dataset(self) -> None:
        bundle = D3RLPYTransitionAdapter(module_count=3).convert(
            "ednet", [self._episode()], action_size=3
        )
        self.assertEqual(bundle.observations.shape, (2, 18))
        self.assertEqual(bundle.actions.tolist(), [2, 1])
        self.assertEqual(bundle.terminals.tolist(), [0.0, 1.0])
        self.assertFalse(bundle.provisional_states)
        native = bundle.to_mdp_dataset()
        self.assertEqual(len(native.episodes), 1)
        self.assertEqual(native.dataset_info.action_size, 3)

    def test_module_is_one_hot_and_action_mask_is_not_flattened(self) -> None:
        adapter = D3RLPYTransitionAdapter(module_count=3)
        flattened = adapter.flatten_observation(observation([0, 2], 1))
        np.testing.assert_array_equal(flattened[-3:], [0.0, 1.0, 0.0])
        self.assertEqual(flattened.size, 4 + 3 + 8 + 3)

    def test_logged_action_must_be_eligible(self) -> None:
        bad_step = OfflineStep(
            observation=observation([0], 1),
            logged_action=2,
            reward=0.0,
            next_observation=observation([0], 1),
            terminated=True,
            truncated=False,
            info={"encoder": "selected_phase7_checkpoint"},
        )
        with self.assertRaises(ValueError):
            D3RLPYTransitionAdapter(3).convert(
                "ednet", [OfflineEpisode("bad", (bad_step,))], action_size=3
            )

    def test_constructs_bc_and_rejects_unmasked_cql(self) -> None:
        bc = create_d3rlpy_algorithm(
            "discrete_bc", batch_size=4, learning_rate=1e-3, gamma=0.99
        )
        with self.assertRaisesRegex(ValueError, "dynamic eligibility masks"):
            create_d3rlpy_algorithm(
                "discrete_cql", batch_size=4, learning_rate=1e-4, gamma=0.99
            )
        cql = create_d3rlpy_algorithm(
            "discrete_cql",
            batch_size=4,
            learning_rate=1e-4,
            gamma=0.99,
            dynamic_action_masks=False,
        )
        self.assertEqual(type(bc).__name__, "DiscreteBC")
        self.assertEqual(type(cql).__name__, "DiscreteCQL")
        self.assertIsNone(bc.impl)
        self.assertIsNone(cql.impl)

    def test_cql_selector_never_returns_masked_action(self) -> None:
        class FakeCQL:
            impl = SimpleNamespace()

            @staticmethod
            def predict_value(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
                del states
                return actions.astype(np.float32)

        selector = MaskedD3RLPYActionSelector(FakeCQL(), "discrete_cql", score_batch_size=2)
        chosen = selector.select(np.zeros(5, dtype=np.float32), np.asarray([1, 3, 7]))
        self.assertEqual(chosen, 7)


if __name__ == "__main__":
    unittest.main()

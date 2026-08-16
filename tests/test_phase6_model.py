from __future__ import annotations

import unittest

import torch

from edu_recommender.student_state_model import (
    GraphTensorBundle,
    KnowledgeAwareMultiTaskLoss,
    KnowledgeAwareStudentStateModel,
    StudentStateModelConfig,
)


def tiny_graph() -> GraphTensorBundle:
    return GraphTensorBundle(
        node_type_ids=torch.tensor([0, 1, 1]),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        edge_type_ids=torch.tensor([0, 1, 0, 1]),
        item_to_node=torch.tensor([-1, -1, 0, 1, 2, -1]),
        concept_to_node=torch.tensor([-1, -1, 1, 2]),
        node_ids=["i0", "c0", "c1"],
        relation_names=["tests", "reverse:tests"],
    )


def tiny_batch() -> dict[str, torch.Tensor]:
    batch_size, full_length = 2, 6
    item = torch.tensor([[2, 3, 4, 2, 3, 0], [3, 4, 2, 3, 0, 0]])
    action = torch.tensor([[2, 2, 2, 2, 2, 0], [2, 2, 2, 2, 0, 0]])
    correctness = torch.tensor([[0, 1, 1, 0, 1, -1], [1, 0, 1, 1, -1, -1]])
    concepts = torch.zeros(batch_size, full_length, 2, dtype=torch.long)
    concepts[:, :, 0] = torch.tensor([[2, 3, 2, 3, 2, 0], [3, 2, 3, 2, 0, 0]])
    attention = item.ne(0).to(torch.uint8)
    target_mask = attention[:, 1:].clone()
    return {
        "item_tokens": item,
        "action_tokens": action,
        "item_type_tokens": torch.where(item.ne(0), 2, 0),
        "module_tokens": torch.where(item.ne(0), 2, 0),
        "source_tokens": torch.where(item.ne(0), 2, 0),
        "relative_days": torch.zeros(batch_size, full_length),
        "time_gaps": torch.zeros(batch_size, full_length),
        "correctness": correctness,
        "scores": torch.full((batch_size, full_length), float("nan")),
        "elapsed_log1p": torch.zeros(batch_size, full_length),
        "engagement_log1p": torch.ones(batch_size, full_length),
        "input_item_tokens": item[:, :-1],
        "input_attention_mask": attention[:, :-1],
        "input_concept_tokens": concepts[:, :-1],
        "target_concept_tokens": concepts[:, 1:],
        "target_item_tokens": item[:, 1:],
        "target_action_tokens": action[:, 1:],
        "target_correctness": correctness[:, 1:],
        "target_mask": target_mask,
    }


class StudentStateModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.graph = tiny_graph()
        self.config = StudentStateModelConfig(
            item_vocab_size=6,
            action_vocab_size=3,
            item_type_vocab_size=3,
            concept_vocab_size=4,
            module_vocab_size=3,
            source_vocab_size=3,
            state_dim=16,
            num_heads=4,
            transformer_layers=1,
            graph_layers=1,
            feedforward_dim=32,
            max_sequence_length=5,
            dropout=0.0,
        )

    def test_shapes_loss_and_gradients(self) -> None:
        model = KnowledgeAwareStudentStateModel(self.config, self.graph)
        batch = tiny_batch()
        outputs = model(batch, self.graph)
        self.assertEqual(outputs["student_states"].shape, (2, 5, 16))
        self.assertEqual(outputs["item_logits"].shape, (2, 5, 6))
        self.assertEqual(outputs["mastery_probabilities"].shape, (2, 5, 4))
        losses = KnowledgeAwareMultiTaskLoss()(outputs, batch)
        self.assertTrue(torch.isfinite(losses["total"]))
        losses["total"].backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_future_token_does_not_change_earlier_state(self) -> None:
        model = KnowledgeAwareStudentStateModel(self.config, self.graph).eval()
        original = tiny_batch()
        changed = {name: value.clone() for name, value in original.items()}
        changed["input_item_tokens"][0, 4] = 4
        changed["item_tokens"][0, 4] = 4
        with torch.no_grad():
            first = model(original, self.graph)["student_states"]
            second = model(changed, self.graph)["student_states"]
        self.assertTrue(torch.allclose(first[0, :4], second[0, :4], atol=1e-6))

    def test_correctness_is_conditioned_on_candidate_without_changing_state(self) -> None:
        model = KnowledgeAwareStudentStateModel(self.config, self.graph).eval()
        first_batch = tiny_batch()
        second_batch = {name: value.clone() for name, value in first_batch.items()}
        second_batch["target_item_tokens"][0, 1] = 5
        second_batch["target_concept_tokens"][0, 1] = torch.tensor([0, 0])
        with torch.no_grad():
            first = model(first_batch, self.graph)
            second = model(second_batch, self.graph)
        self.assertTrue(torch.allclose(first["student_states"], second["student_states"], atol=1e-6))
        self.assertFalse(
            torch.allclose(
                first["correctness_logits"][0, 1],
                second["correctness_logits"][0, 1],
                atol=1e-6,
            )
        )

    def test_state_encoding_does_not_require_a_candidate_query(self) -> None:
        model = KnowledgeAwareStudentStateModel(self.config, self.graph).eval()
        batch = tiny_batch()
        del batch["target_item_tokens"]
        del batch["target_concept_tokens"]
        with torch.no_grad():
            outputs = model(batch, self.graph)
        self.assertEqual(outputs["student_states"].shape, (2, 5, 16))
        self.assertNotIn("correctness_logits", outputs)

    def test_unknown_concept_is_not_used_as_mastery_supervision(self) -> None:
        model = KnowledgeAwareStudentStateModel(self.config, self.graph)
        batch = tiny_batch()
        batch["target_concept_tokens"].fill_(1)
        outputs = model(batch, self.graph)
        losses = KnowledgeAwareMultiTaskLoss()(outputs, batch)
        self.assertEqual(float(losses["mastery"].detach()), 0.0)


if __name__ == "__main__":
    unittest.main()

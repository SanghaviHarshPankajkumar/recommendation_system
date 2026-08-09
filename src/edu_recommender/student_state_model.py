from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class GraphTensorBundle:
    node_type_ids: Tensor
    edge_index: Tensor
    edge_type_ids: Tensor
    item_to_node: Tensor
    concept_to_node: Tensor
    node_ids: list[str]
    relation_names: list[str]

    def to(self, device: torch.device) -> "GraphTensorBundle":
        return GraphTensorBundle(
            node_type_ids=self.node_type_ids.to(device),
            edge_index=self.edge_index.to(device),
            edge_type_ids=self.edge_type_ids.to(device),
            item_to_node=self.item_to_node.to(device),
            concept_to_node=self.concept_to_node.to(device),
            node_ids=self.node_ids,
            relation_names=self.relation_names,
        )

    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def num_node_types(self) -> int:
        return int(self.node_type_ids.max().item() + 1) if len(self.node_type_ids) else 0

    @property
    def num_relations(self) -> int:
        return len(self.relation_names)


class GraphTensorBuilder:
    """Convert Phase 4 CSV graphs and Phase 5 vocabularies into model tensors."""

    def __init__(self, dataset: str, graph_root: Path, sequence_root: Path):
        self.dataset = dataset
        self.graph_root = Path(graph_root)
        self.sequence_root = Path(sequence_root)

    def build(self) -> GraphTensorBundle:
        graph_dir = self.graph_root / self.dataset
        sequence_dir = self.sequence_root / self.dataset
        nodes = pd.read_csv(graph_dir / "nodes.csv.gz")
        explicit = pd.read_csv(graph_dir / "edges_explicit.csv.gz")
        edge_frames = [explicit]
        prerequisite_path = graph_dir / "edges_prerequisite_dag.csv.gz"
        if prerequisite_path.exists():
            edge_frames.append(pd.read_csv(prerequisite_path))
        edges = pd.concat(edge_frames, ignore_index=True)

        node_ids = nodes["node_id"].astype(str).tolist()
        node_index = {node_id: index for index, node_id in enumerate(node_ids)}
        node_types = sorted(nodes["node_type"].astype(str).unique())
        node_type_index = {name: index for index, name in enumerate(node_types)}
        node_type_ids = torch.tensor(
            [node_type_index[str(value)] for value in nodes["node_type"]], dtype=torch.long
        )

        relation_names = sorted(edges["edge_type"].astype(str).unique())
        directed_relations = relation_names + [f"reverse:{name}" for name in relation_names]
        relation_index = {name: index for index, name in enumerate(directed_relations)}
        sources: list[int] = []
        targets: list[int] = []
        relation_ids: list[int] = []
        for source, target, relation in edges[["source_id", "target_id", "edge_type"]].itertuples(index=False):
            source_id, target_id, relation = str(source), str(target), str(relation)
            if source_id not in node_index or target_id not in node_index:
                raise ValueError("Graph edge references an unknown node")
            sources.extend([node_index[source_id], node_index[target_id]])
            targets.extend([node_index[target_id], node_index[source_id]])
            relation_ids.extend([relation_index[relation], relation_index[f"reverse:{relation}"]])
        edge_index = torch.tensor([sources, targets], dtype=torch.long)
        edge_type_ids = torch.tensor(relation_ids, dtype=torch.long)

        vocabularies = json.loads((sequence_dir / "vocabularies.json").read_text(encoding="utf-8"))
        item_vocab = vocabularies["item_id"]
        concept_vocab = vocabularies["concept_ids"]
        item_to_node = torch.full((max(item_vocab.values()) + 1,), -1, dtype=torch.long)
        concept_to_node = torch.full((max(concept_vocab.values()) + 1,), -1, dtype=torch.long)
        candidate_path = sequence_dir / "candidate_catalog.csv.gz"
        if candidate_path.exists():
            candidates = pd.read_csv(candidate_path)
            for item_token, graph_node_id in candidates[["item_token", "graph_node_id"]].itertuples(index=False):
                if str(graph_node_id) in node_index:
                    current = int(item_to_node[int(item_token)])
                    if current not in (-1, node_index[str(graph_node_id)]):
                        raise ValueError(f"Item token {item_token} maps to multiple graph nodes")
                    item_to_node[int(item_token)] = node_index[str(graph_node_id)]
        self._map_non_candidate_items(item_vocab, item_to_node, node_index)
        self._map_concepts(concept_vocab, concept_to_node, node_index)
        return GraphTensorBundle(
            node_type_ids=node_type_ids,
            edge_index=edge_index,
            edge_type_ids=edge_type_ids,
            item_to_node=item_to_node,
            concept_to_node=concept_to_node,
            node_ids=node_ids,
            relation_names=directed_relations,
        )

    def _map_non_candidate_items(
        self, item_vocab: dict[str, int], item_to_node: Tensor, node_index: dict[str, int]
    ) -> None:
        for item, token in item_vocab.items():
            if item.startswith("<") or item_to_node[token] >= 0:
                continue
            node_id: str | None = None
            if self.dataset == "ednet" and item.startswith("ednet:"):
                raw = item.removeprefix("ednet:")
                node_type = {"q": "question", "b": "bundle", "e": "explanation", "l": "lecture"}.get(raw[:1])
                if node_type:
                    node_id = f"ednet:{node_type}:{raw}"
            elif self.dataset == "oulad" and item.startswith("oulad:module:"):
                node_id = "oulad:presentation:" + item.removeprefix("oulad:module:")
            if node_id in node_index:
                item_to_node[token] = node_index[node_id]

    def _map_concepts(
        self, concept_vocab: dict[str, int], concept_to_node: Tensor, node_index: dict[str, int]
    ) -> None:
        for concept, token in concept_vocab.items():
            if concept.startswith("<"):
                continue
            node_id = (
                f"ednet:skill:{concept}"
                if self.dataset == "ednet"
                else f"oulad:module:{concept.removeprefix('module:')}"
            )
            if node_id in node_index:
                concept_to_node[token] = node_index[node_id]


class RelationalGraphLayer(nn.Module):
    def __init__(self, state_dim: int, num_relations: int, dropout: float):
        super().__init__()
        self.relation_weights = nn.Parameter(torch.empty(num_relations, state_dim, state_dim))
        self.self_linear = nn.Linear(state_dim, state_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(state_dim))
        self.norm = nn.LayerNorm(state_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.relation_weights)

    def forward(self, node_states: Tensor, edge_index: Tensor, edge_types: Tensor) -> Tensor:
        source, target = edge_index
        aggregate = torch.zeros_like(node_states)
        degree = torch.zeros(node_states.shape[0], device=node_states.device, dtype=node_states.dtype)
        for relation_id in range(self.relation_weights.shape[0]):
            mask = edge_types == relation_id
            if not torch.any(mask):
                continue
            relation_source = source[mask]
            relation_target = target[mask]
            messages = node_states[relation_source] @ self.relation_weights[relation_id]
            aggregate.index_add_(0, relation_target, messages)
            degree.index_add_(0, relation_target, torch.ones_like(relation_target, dtype=node_states.dtype))
        aggregate = aggregate / degree.clamp_min(1).unsqueeze(-1)
        updated = self.self_linear(node_states) + aggregate + self.bias
        return self.norm(node_states + self.dropout(F.gelu(updated)))


class RelationalGraphEncoder(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        num_node_types: int,
        num_relations: int,
        state_dim: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.node_embeddings = nn.Embedding(num_nodes, state_dim)
        self.node_type_embeddings = nn.Embedding(num_node_types, state_dim)
        self.layers = nn.ModuleList(
            [RelationalGraphLayer(state_dim, num_relations, dropout) for _ in range(num_layers)]
        )
        self.input_norm = nn.LayerNorm(state_dim)
        nn.init.normal_(self.node_embeddings.weight, std=0.02)
        nn.init.normal_(self.node_type_embeddings.weight, std=0.02)

    def forward(self, graph: GraphTensorBundle) -> Tensor:
        states = self.input_norm(
            self.node_embeddings.weight + self.node_type_embeddings(graph.node_type_ids)
        )
        for layer in self.layers:
            states = layer(states, graph.edge_index, graph.edge_type_ids)
        return states


@dataclass(frozen=True)
class StudentStateModelConfig:
    item_vocab_size: int
    action_vocab_size: int
    item_type_vocab_size: int
    concept_vocab_size: int
    module_vocab_size: int
    source_vocab_size: int
    state_dim: int = 64
    num_heads: int = 4
    transformer_layers: int = 2
    graph_layers: int = 2
    feedforward_dim: int = 128
    max_sequence_length: int = 127
    dropout: float = 0.1
    variant: str = "full"


class KnowledgeAwareStudentStateModel(nn.Module):
    """Causal learner encoder with relational graph context and mastery-gated fusion."""

    def __init__(self, config: StudentStateModelConfig, graph: GraphTensorBundle):
        super().__init__()
        valid_variants = {"sequence_only", "graph_only", "sequence_graph", "sequence_mastery", "full"}
        if config.variant not in valid_variants:
            raise ValueError(f"Unknown model variant {config.variant!r}; expected one of {sorted(valid_variants)}")
        self.config = config
        dim = config.state_dim
        self.item_embedding = nn.Embedding(config.item_vocab_size, dim, padding_idx=0)
        self.action_embedding = nn.Embedding(config.action_vocab_size, dim, padding_idx=0)
        self.item_type_embedding = nn.Embedding(config.item_type_vocab_size, dim, padding_idx=0)
        self.concept_embedding = nn.Embedding(config.concept_vocab_size, dim, padding_idx=0)
        self.module_embedding = nn.Embedding(config.module_vocab_size, dim, padding_idx=0)
        self.source_embedding = nn.Embedding(config.source_vocab_size, dim, padding_idx=0)
        self.correctness_embedding = nn.Embedding(3, dim)
        self.position_embedding = nn.Embedding(config.max_sequence_length, dim)
        self.continuous_projection = nn.Sequential(nn.Linear(6, dim), nn.GELU(), nn.LayerNorm(dim))
        self.input_norm = nn.LayerNorm(dim)
        self.input_dropout = nn.Dropout(config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=config.num_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.history_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.transformer_layers, norm=nn.LayerNorm(dim), enable_nested_tensor=False
        )
        self.graph_encoder = RelationalGraphEncoder(
            graph.num_nodes,
            graph.num_node_types,
            graph.num_relations,
            dim,
            config.graph_layers,
            config.dropout,
        )
        self.graph_projection = nn.Linear(dim, dim)
        self.mastery_head = nn.Linear(dim, config.concept_vocab_size)
        self.mastery_projection = nn.Linear(dim, dim)
        self.modality_transforms = nn.ModuleList([nn.Linear(dim, dim) for _ in range(3)])
        self.fusion_gate = nn.Linear(dim * 3, 3)
        self.fusion_norm = nn.LayerNorm(dim)
        self.action_head = nn.Linear(dim, config.action_vocab_size)
        self.correctness_head = nn.Linear(dim, 1)
        self.item_bias = nn.Parameter(torch.zeros(config.item_vocab_size))
        self.register_buffer("item_to_node", graph.item_to_node.clone(), persistent=True)
        self.register_buffer("concept_to_node", graph.concept_to_node.clone(), persistent=True)
        self._reset_embeddings()

    def _reset_embeddings(self) -> None:
        for embedding in (
            self.item_embedding, self.action_embedding, self.item_type_embedding, self.concept_embedding,
            self.module_embedding, self.source_embedding, self.correctness_embedding, self.position_embedding,
        ):
            nn.init.normal_(embedding.weight, std=0.02)
            if embedding.padding_idx is not None:
                with torch.no_grad():
                    embedding.weight[embedding.padding_idx].zero_()

    def forward(self, batch: dict[str, Tensor], graph: GraphTensorBundle) -> dict[str, Tensor]:
        item_tokens = batch["input_item_tokens"].long()
        attention_mask = batch["input_attention_mask"].bool()
        concept_tokens = batch["input_concept_tokens"].long()
        batch_size, sequence_length = item_tokens.shape
        if sequence_length > self.config.max_sequence_length:
            raise ValueError("Input sequence exceeds configured maximum length")
        position_ids = torch.arange(sequence_length, device=item_tokens.device).unsqueeze(0)
        correctness_index = (batch["correctness"].long()[:, :-1] + 1).clamp(0, 2)
        concept_mask = concept_tokens.ne(0).unsqueeze(-1)
        concept_sum = (self.concept_embedding(concept_tokens) * concept_mask).sum(dim=2)
        concept_mean = concept_sum / concept_mask.sum(dim=2).clamp_min(1)
        continuous = torch.stack(
            [
                batch["time_gaps"][:, :-1],
                batch["elapsed_log1p"][:, :-1],
                batch["engagement_log1p"][:, :-1],
                torch.nan_to_num(batch["scores"][:, :-1] / 100.0),
                torch.isfinite(batch["scores"][:, :-1]).float(),
                torch.nan_to_num(batch["relative_days"][:, :-1] / 365.0).clamp(-2, 2),
            ],
            dim=-1,
        )
        inputs = (
            self.item_embedding(item_tokens)
            + self.action_embedding(batch["action_tokens"].long()[:, :-1])
            + self.item_type_embedding(batch["item_type_tokens"].long()[:, :-1])
            + self.module_embedding(batch["module_tokens"].long()[:, :-1])
            + self.source_embedding(batch["source_tokens"].long()[:, :-1])
            + self.correctness_embedding(correctness_index)
            + concept_mean
            + self.position_embedding(position_ids)
            + self.continuous_projection(continuous)
        )
        inputs = self.input_dropout(self.input_norm(inputs))
        causal_mask = torch.triu(
            torch.ones(sequence_length, sequence_length, device=item_tokens.device, dtype=torch.bool), diagonal=1
        )
        history_states = self.history_encoder(
            inputs,
            mask=causal_mask,
            src_key_padding_mask=~attention_mask,
            is_causal=True,
        )

        graph_states = self.graph_encoder(graph)
        item_graph = self._gather_graph(item_tokens, self.item_to_node, graph_states)
        concept_graph = self._gather_concept_graph(concept_tokens, graph_states)
        graph_context = self.graph_projection(item_graph + concept_graph)

        mastery_logits = self.mastery_head(history_states)
        mastery_probabilities = torch.sigmoid(mastery_logits)
        if mastery_probabilities.shape[-1] >= 2:
            valid_mastery = mastery_probabilities.clone()
            valid_mastery[..., :2] = 0
        else:
            valid_mastery = mastery_probabilities
        mastery_context = self.mastery_projection(valid_mastery @ self.concept_embedding.weight)

        modalities = [history_states, graph_context, mastery_context]
        transformed = torch.stack(
            [layer(value) for layer, value in zip(self.modality_transforms, modalities)], dim=-2
        )
        active_modalities = {
            "sequence_only": (True, False, False),
            "graph_only": (False, True, False),
            "sequence_graph": (True, True, False),
            "sequence_mastery": (True, False, True),
            "full": (True, True, True),
        }[self.config.variant]
        gate_logits = self.fusion_gate(torch.cat(modalities, dim=-1))
        active_mask = torch.tensor(active_modalities, device=gate_logits.device, dtype=torch.bool)
        gate_logits = gate_logits.masked_fill(~active_mask, torch.finfo(gate_logits.dtype).min)
        gate_weights = torch.softmax(gate_logits, dim=-1)
        residual = history_states if active_modalities[0] else torch.zeros_like(history_states)
        fused = self.fusion_norm(residual + (transformed * gate_weights.unsqueeze(-1)).sum(dim=-2))
        return {
            "student_states": fused,
            "history_states": history_states,
            "graph_context": graph_context,
            "mastery_logits": mastery_logits,
            "mastery_probabilities": mastery_probabilities,
            "fusion_weights": gate_weights,
            "item_logits": F.linear(fused, self.item_embedding.weight, self.item_bias),
            "action_logits": self.action_head(fused),
            "correctness_logits": self.correctness_head(fused).squeeze(-1),
        }

    @staticmethod
    def _gather_graph(tokens: Tensor, mapping: Tensor, graph_states: Tensor) -> Tensor:
        node_ids = mapping[tokens]
        present = node_ids.ge(0).unsqueeze(-1)
        return graph_states[node_ids.clamp_min(0)] * present

    def _gather_concept_graph(self, concept_tokens: Tensor, graph_states: Tensor) -> Tensor:
        node_ids = self.concept_to_node[concept_tokens]
        present = node_ids.ge(0).unsqueeze(-1)
        gathered = graph_states[node_ids.clamp_min(0)] * present
        return gathered.sum(dim=2) / present.sum(dim=2).clamp_min(1)


class KnowledgeAwareMultiTaskLoss(nn.Module):
    def __init__(
        self,
        item_weight: float = 1.0,
        action_weight: float = 0.25,
        correctness_weight: float = 0.5,
        mastery_weight: float = 0.5,
    ):
        super().__init__()
        self.weights = {
            "item": item_weight,
            "action": action_weight,
            "correctness": correctness_weight,
            "mastery": mastery_weight,
        }

    def forward(self, outputs: dict[str, Tensor], batch: dict[str, Tensor]) -> dict[str, Tensor]:
        mask = batch["target_mask"].bool()
        item_targets = batch["target_item_tokens"].long()
        action_targets = batch["target_action_tokens"].long()
        correctness_targets = batch["target_correctness"].float()
        item_loss = F.cross_entropy(outputs["item_logits"][mask], item_targets[mask])
        action_loss = F.cross_entropy(outputs["action_logits"][mask], action_targets[mask])
        correctness_mask = mask & correctness_targets.ge(0)
        if torch.any(correctness_mask):
            correctness_loss = F.binary_cross_entropy_with_logits(
                outputs["correctness_logits"][correctness_mask], correctness_targets[correctness_mask]
            )
        else:
            correctness_loss = outputs["correctness_logits"].sum() * 0

        target_concepts = batch["target_concept_tokens"].long()
        concept_mask = target_concepts.ne(0) & mask.unsqueeze(-1) & correctness_targets.ge(0).unsqueeze(-1)
        if torch.any(concept_mask):
            gathered_mastery = outputs["mastery_logits"].gather(-1, target_concepts.clamp_min(0))
            expanded_correctness = correctness_targets.unsqueeze(-1).expand_as(gathered_mastery)
            mastery_loss = F.binary_cross_entropy_with_logits(
                gathered_mastery[concept_mask], expanded_correctness[concept_mask]
            )
        else:
            mastery_loss = outputs["mastery_logits"].sum() * 0
        total = (
            self.weights["item"] * item_loss
            + self.weights["action"] * action_loss
            + self.weights["correctness"] * correctness_loss
            + self.weights["mastery"] * mastery_loss
        )
        return {
            "total": total,
            "item": item_loss,
            "action": action_loss,
            "correctness": correctness_loss,
            "mastery": mastery_loss,
        }


def model_config_from_vocabularies(
    vocabulary_path: Path, model_options: dict[str, object]
) -> StudentStateModelConfig:
    vocabularies = json.loads(Path(vocabulary_path).read_text(encoding="utf-8"))
    sizes = {name: max(mapping.values()) + 1 for name, mapping in vocabularies.items()}
    return StudentStateModelConfig(
        item_vocab_size=sizes["item_id"],
        action_vocab_size=sizes["action_type"],
        item_type_vocab_size=sizes["item_type"],
        concept_vocab_size=sizes["concept_ids"],
        module_vocab_size=sizes["module_id"],
        source_vocab_size=sizes["source"],
        **model_options,
    )

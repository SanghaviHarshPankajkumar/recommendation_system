from __future__ import annotations

import gzip
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


NODE_COLUMNS = ["node_id", "node_type", "dataset", "label", "attributes_json"]
EDGE_COLUMNS = [
    "source_id",
    "target_id",
    "edge_type",
    "dataset",
    "provenance",
    "confidence",
    "attributes_json",
]


def _json_attributes(**values: object) -> str:
    clean = {key: value for key, value in values.items() if pd.notna(value)}
    return json.dumps(clean, sort_keys=True, separators=(",", ":"))


def _split_tags(value: object) -> tuple[str, ...]:
    if pd.isna(value):
        return ()
    return tuple(sorted({part.strip() for part in str(value).split(";") if part.strip()}))


class GraphValidator:
    """Validate graph referential integrity and candidate-DAG assumptions."""

    @staticmethod
    def validate(nodes: pd.DataFrame, edges: pd.DataFrame, require_dag: bool = False) -> dict[str, object]:
        missing_node_columns = set(NODE_COLUMNS).difference(nodes.columns)
        missing_edge_columns = set(EDGE_COLUMNS).difference(edges.columns)
        if missing_node_columns or missing_edge_columns:
            raise ValueError(
                f"Graph schema mismatch; missing nodes={sorted(missing_node_columns)}, "
                f"edges={sorted(missing_edge_columns)}"
            )
        if nodes["node_id"].isna().any() or nodes["node_id"].duplicated().any():
            raise ValueError("node_id must be non-null and unique")
        node_ids = set(nodes["node_id"].astype(str))
        referenced = set(edges["source_id"].astype(str)) | set(edges["target_id"].astype(str))
        unknown = referenced.difference(node_ids)
        if unknown:
            raise ValueError(f"Edges reference {len(unknown)} unknown nodes")
        edge_key = ["source_id", "target_id", "edge_type"]
        if edges.duplicated(edge_key).any():
            raise ValueError("Duplicate graph edges found")
        if (edges["source_id"] == edges["target_id"]).any():
            raise ValueError("Self-loop found")
        if require_dag and not GraphValidator.is_dag(edges):
            raise ValueError("Prerequisite candidate graph contains a cycle")
        return {
            "passed": True,
            "node_count": int(len(nodes)),
            "edge_count": int(len(edges)),
            "dag_required": require_dag,
        }

    @staticmethod
    def is_dag(edges: pd.DataFrame) -> bool:
        adjacency: dict[str, list[str]] = {}
        indegree: dict[str, int] = {}
        for source, target in edges[["source_id", "target_id"]].itertuples(index=False):
            source, target = str(source), str(target)
            adjacency.setdefault(source, []).append(target)
            indegree.setdefault(source, 0)
            indegree[target] = indegree.get(target, 0) + 1
        stack = [node for node, degree in indegree.items() if degree == 0]
        visited = 0
        while stack:
            node = stack.pop()
            visited += 1
            for target in adjacency.get(node, []):
                indegree[target] -= 1
                if indegree[target] == 0:
                    stack.append(target)
        return visited == len(indegree)


class EdNetGraphBuilder:
    def __init__(self, metadata_dir: Path):
        self.metadata_dir = Path(metadata_dir)

    def build(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        questions = pd.read_csv(self.metadata_dir / "questions.csv", dtype=str)
        lectures = pd.read_csv(self.metadata_dir / "lectures.csv", dtype=str)
        nodes: dict[str, dict[str, object]] = {}
        edges: list[dict[str, object]] = []

        def add_node(node_id: str, node_type: str, label: str, **attributes: object) -> None:
            nodes[node_id] = {
                "node_id": node_id,
                "node_type": node_type,
                "dataset": "ednet",
                "label": label,
                "attributes_json": _json_attributes(**attributes),
            }

        def add_edge(source: str, target: str, edge_type: str, **attributes: object) -> None:
            edges.append(
                {
                    "source_id": source,
                    "target_id": target,
                    "edge_type": edge_type,
                    "dataset": "ednet",
                    "provenance": "official_metadata",
                    "confidence": 1.0,
                    "attributes_json": _json_attributes(**attributes),
                }
            )

        for row in questions.itertuples(index=False):
            question = f"ednet:question:{row.question_id}"
            bundle = f"ednet:bundle:{row.bundle_id}"
            explanation = f"ednet:explanation:{row.explanation_id}"
            part = f"ednet:part:{row.part}"
            add_node(question, "question", row.question_id, correct_answer=row.correct_answer, deployed_at=row.deployed_at)
            add_node(bundle, "bundle", row.bundle_id)
            add_node(explanation, "explanation", row.explanation_id)
            add_node(part, "part", f"Part {row.part}")
            add_edge(question, bundle, "belongs_to_bundle")
            add_edge(question, explanation, "explained_by")
            add_edge(bundle, part, "belongs_to_part")
            for tag in _split_tags(row.tags):
                skill = f"ednet:skill:{tag}"
                add_node(skill, "skill", f"Skill {tag}")
                add_edge(question, skill, "tests")

        for row in lectures.itertuples(index=False):
            lecture = f"ednet:lecture:{row.lecture_id}"
            part = f"ednet:part:{row.part}"
            add_node(lecture, "lecture", row.lecture_id, video_length=row.video_length, deployed_at=row.deployed_at)
            add_node(part, "part", f"Part {row.part}")
            add_edge(lecture, part, "belongs_to_part")
            for tag in _split_tags(row.tags):
                skill = f"ednet:skill:{tag}"
                add_node(skill, "skill", f"Skill {tag}")
                add_edge(lecture, skill, "teaches")

        node_frame = pd.DataFrame(nodes.values(), columns=NODE_COLUMNS).sort_values("node_id").reset_index(drop=True)
        edge_frame = pd.DataFrame(edges, columns=EDGE_COLUMNS).drop_duplicates(
            ["source_id", "target_id", "edge_type"]
        ).sort_values(["edge_type", "source_id", "target_id"]).reset_index(drop=True)
        return node_frame, edge_frame


class OULADGraphBuilder:
    def __init__(self, raw_dir: Path):
        self.raw_dir = Path(raw_dir)

    def build(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        courses = pd.read_csv(self.raw_dir / "courses.csv", dtype=str)
        assessments = pd.read_csv(self.raw_dir / "assessments.csv", dtype=str)
        vle = pd.read_csv(self.raw_dir / "vle.csv", dtype=str)
        nodes: dict[str, dict[str, object]] = {}
        edges: list[dict[str, object]] = []

        def add_node(node_id: str, node_type: str, label: str, **attributes: object) -> None:
            nodes[node_id] = {
                "node_id": node_id,
                "node_type": node_type,
                "dataset": "oulad",
                "label": label,
                "attributes_json": _json_attributes(**attributes),
            }

        def add_edge(source: str, target: str, edge_type: str) -> None:
            edges.append(
                {
                    "source_id": source,
                    "target_id": target,
                    "edge_type": edge_type,
                    "dataset": "oulad",
                    "provenance": "official_metadata",
                    "confidence": 1.0,
                    "attributes_json": "{}",
                }
            )

        for row in courses.itertuples(index=False):
            module = f"oulad:module:{row.code_module}"
            presentation = f"oulad:presentation:{row.code_module}:{row.code_presentation}"
            add_node(module, "module", row.code_module)
            add_node(presentation, "module_presentation", f"{row.code_module} {row.code_presentation}", length_days=row.module_presentation_length)
            add_edge(presentation, module, "presentation_of")

        for row in assessments.itertuples(index=False):
            presentation = f"oulad:presentation:{row.code_module}:{row.code_presentation}"
            assessment = f"oulad:assessment:{row.id_assessment}"
            assessment_type = f"oulad:assessment_type:{row.assessment_type}"
            add_node(assessment, "assessment", row.id_assessment, date=row.date, weight=row.weight)
            add_node(assessment_type, "assessment_type", row.assessment_type)
            add_edge(assessment, presentation, "belongs_to_presentation")
            add_edge(assessment, assessment_type, "has_assessment_type")

        for row in vle.itertuples(index=False):
            presentation = f"oulad:presentation:{row.code_module}:{row.code_presentation}"
            activity = f"oulad:vle_activity:{row.code_module}:{row.code_presentation}:{row.id_site}"
            activity_type = f"oulad:activity_type:{row.activity_type}"
            add_node(activity, "vle_activity", row.id_site, week_from=row.week_from, week_to=row.week_to)
            add_node(activity_type, "activity_type", row.activity_type)
            add_edge(activity, presentation, "belongs_to_presentation")
            add_edge(activity, activity_type, "has_activity_type")

        node_frame = pd.DataFrame(nodes.values(), columns=NODE_COLUMNS).sort_values("node_id").reset_index(drop=True)
        edge_frame = pd.DataFrame(edges, columns=EDGE_COLUMNS).drop_duplicates(
            ["source_id", "target_id", "edge_type"]
        ).sort_values(["edge_type", "source_id", "target_id"]).reset_index(drop=True)
        return node_frame, edge_frame


@dataclass(frozen=True)
class PrerequisiteThresholds:
    min_transition_support: int = 100
    min_conditional_support: int = 30
    min_direction_confidence: float = 0.65
    min_performance_lift: float = 0.05


class EdNetPrerequisiteInference:
    """Infer non-causal skill-order candidates from adjacent train interactions."""

    def __init__(self, train_dir: Path, thresholds: PrerequisiteThresholds, chunksize: int = 250_000):
        self.train_dir = Path(train_dir)
        self.thresholds = thresholds
        self.chunksize = chunksize

    def infer(self, known_skills: Iterable[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
        skills = sorted({str(skill).removeprefix("ednet:skill:") for skill in known_skills})
        skill_index = {skill: index for index, skill in enumerate(skills)}
        size = len(skills)
        transitions = np.zeros((size, size), dtype=np.int64)
        previous_correct = np.zeros((size, size), dtype=np.int64)
        previous_incorrect = np.zeros((size, size), dtype=np.int64)
        next_correct_after_correct = np.zeros((size, size), dtype=np.int64)
        next_correct_after_incorrect = np.zeros((size, size), dtype=np.int64)
        final_events = learner_sequences = 0
        tag_cache: dict[str, tuple[int, ...]] = {}

        def indices(value: object) -> tuple[int, ...]:
            key = str(value)
            if key not in tag_cache:
                tag_cache[key] = tuple(skill_index[tag] for tag in _split_tags(value) if tag in skill_index)
            return tag_cache[key]

        for path in sorted(self.train_dir.glob("part-*.csv.gz")):
            previous: tuple[str, tuple[int, ...], int] | None = None
            for chunk in pd.read_csv(
                path,
                usecols=["student_id", "concept_ids", "correctness", "final_response"],
                chunksize=self.chunksize,
            ):
                mask = chunk["final_response"].astype(str).str.lower().eq("true") & chunk["correctness"].notna() & chunk["concept_ids"].notna()
                filtered = chunk.loc[mask, ["student_id", "concept_ids", "correctness"]]
                final_events += len(filtered)
                for student, concept_ids, correctness in filtered.itertuples(index=False, name=None):
                    current = (str(student), indices(concept_ids), int(float(correctness) >= 0.5))
                    if previous is not None and previous[0] == current[0]:
                        previous_tags, current_tags = previous[1], current[1]
                        for source in previous_tags:
                            for target in current_tags:
                                if source == target:
                                    continue
                                transitions[source, target] += 1
                                if previous[2] == 1:
                                    previous_correct[source, target] += 1
                                    next_correct_after_correct[source, target] += current[2]
                                else:
                                    previous_incorrect[source, target] += 1
                                    next_correct_after_incorrect[source, target] += current[2]
                    else:
                        learner_sequences += 1
                    previous = current

        rows: list[dict[str, object]] = []
        for source, target in zip(*np.nonzero(transitions)):
            support = int(transitions[source, target])
            reverse = int(transitions[target, source])
            correct_n = int(previous_correct[source, target])
            incorrect_n = int(previous_incorrect[source, target])
            p_after_correct = float(next_correct_after_correct[source, target] / correct_n) if correct_n else math.nan
            p_after_incorrect = float(next_correct_after_incorrect[source, target] / incorrect_n) if incorrect_n else math.nan
            rows.append(
                {
                    "source_skill_id": f"ednet:skill:{skills[source]}",
                    "target_skill_id": f"ednet:skill:{skills[target]}",
                    "transition_support": support,
                    "reverse_transition_support": reverse,
                    "direction_confidence": support / (support + reverse) if support + reverse else 0.0,
                    "previous_correct_support": correct_n,
                    "previous_incorrect_support": incorrect_n,
                    "p_next_correct_after_previous_correct": p_after_correct,
                    "p_next_correct_after_previous_incorrect": p_after_incorrect,
                    "performance_lift": p_after_correct - p_after_incorrect,
                }
            )
        columns = [
            "source_skill_id", "target_skill_id", "transition_support", "reverse_transition_support",
            "direction_confidence", "previous_correct_support", "previous_incorrect_support",
            "p_next_correct_after_previous_correct", "p_next_correct_after_previous_incorrect", "performance_lift",
        ]
        candidates = pd.DataFrame(rows, columns=columns)
        if not candidates.empty:
            candidates = candidates.sort_values(["transition_support", "direction_confidence"], ascending=False).reset_index(drop=True)
        filtered = candidates[
            (candidates["transition_support"] >= self.thresholds.min_transition_support)
            & (candidates["previous_correct_support"] >= self.thresholds.min_conditional_support)
            & (candidates["previous_incorrect_support"] >= self.thresholds.min_conditional_support)
            & (candidates["direction_confidence"] >= self.thresholds.min_direction_confidence)
            & (candidates["performance_lift"] >= self.thresholds.min_performance_lift)
        ].copy()
        dag_rows = self._greedy_dag(filtered)
        dag_edges = pd.DataFrame(dag_rows, columns=EDGE_COLUMNS)
        return candidates, dag_edges, {"final_response_events": final_events, "learner_sequences": learner_sequences}

    @staticmethod
    def _greedy_dag(candidates: pd.DataFrame) -> list[dict[str, object]]:
        ordered = candidates.sort_values(
            ["direction_confidence", "performance_lift", "transition_support"], ascending=False
        )
        adjacency: dict[str, set[str]] = {}
        accepted: list[dict[str, object]] = []

        def has_path(start: str, destination: str) -> bool:
            stack, seen = [start], set()
            while stack:
                node = stack.pop()
                if node == destination:
                    return True
                if node not in seen:
                    seen.add(node)
                    stack.extend(adjacency.get(node, ()))
            return False

        for row in ordered.itertuples(index=False):
            if has_path(row.target_skill_id, row.source_skill_id):
                continue
            adjacency.setdefault(row.source_skill_id, set()).add(row.target_skill_id)
            accepted.append(
                {
                    "source_id": row.source_skill_id,
                    "target_id": row.target_skill_id,
                    "edge_type": "empirical_prerequisite_candidate",
                    "dataset": "ednet",
                    "provenance": "train_adjacent_final_responses",
                    "confidence": row.direction_confidence,
                    "attributes_json": _json_attributes(
                        transition_support=row.transition_support,
                        performance_lift=row.performance_lift,
                        previous_correct_support=row.previous_correct_support,
                        previous_incorrect_support=row.previous_incorrect_support,
                    ),
                }
            )
        return accepted


class Phase4GraphPipeline:
    def __init__(self, config: dict[str, object]):
        self.config = config
        self.output_root = Path(str(config["output_root"]))

    @classmethod
    def from_json(cls, path: Path) -> "Phase4GraphPipeline":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def run(self) -> dict[str, object]:
        self.output_root.mkdir(parents=True, exist_ok=True)
        ednet_dir, oulad_dir = self.output_root / "ednet", self.output_root / "oulad"
        ednet_dir.mkdir(exist_ok=True)
        oulad_dir.mkdir(exist_ok=True)

        ednet_nodes, ednet_explicit = EdNetGraphBuilder(Path(str(self.config["ednet_metadata_dir"]))).build()
        oulad_nodes, oulad_explicit = OULADGraphBuilder(Path(str(self.config["oulad_raw_dir"]))).build()
        thresholds = PrerequisiteThresholds(**dict(self.config["prerequisite_thresholds"]))
        inference = EdNetPrerequisiteInference(
            Path(str(self.config["ednet_train_dir"])), thresholds, int(self.config.get("chunksize", 250_000))
        )
        skill_ids = ednet_nodes.loc[ednet_nodes["node_type"] == "skill", "node_id"]
        candidates, prerequisite_dag, inference_stats = inference.infer(skill_ids)

        ednet_validation = GraphValidator.validate(ednet_nodes, ednet_explicit)
        dag_validation = GraphValidator.validate(ednet_nodes, prerequisite_dag, require_dag=True)
        oulad_validation = GraphValidator.validate(oulad_nodes, oulad_explicit)
        self._write_csv(ednet_nodes, ednet_dir / "nodes.csv.gz")
        self._write_csv(ednet_explicit, ednet_dir / "edges_explicit.csv.gz")
        self._write_csv(candidates, ednet_dir / "prerequisite_candidates.csv.gz")
        self._write_csv(prerequisite_dag, ednet_dir / "edges_prerequisite_dag.csv.gz")
        self._write_csv(oulad_nodes, oulad_dir / "nodes.csv.gz")
        self._write_csv(oulad_explicit, oulad_dir / "edges_explicit.csv.gz")

        manifest = {
            "status": "complete",
            "methodology": {
                "explicit_edges": "official dataset metadata",
                "candidate_edges": "train-only adjacent final responses; associative, not causal ground truth",
                "cycle_policy": "greedy highest-confidence acyclic subset",
                "thresholds": thresholds.__dict__,
            },
            "ednet": {
                "validation": ednet_validation,
                "dag_validation": dag_validation,
                "candidate_count": int(len(candidates)),
                "filtered_dag_edge_count": int(len(prerequisite_dag)),
                **inference_stats,
                "node_type_counts": ednet_nodes["node_type"].value_counts().sort_index().to_dict(),
                "explicit_edge_type_counts": ednet_explicit["edge_type"].value_counts().sort_index().to_dict(),
            },
            "oulad": {
                "validation": oulad_validation,
                "node_type_counts": oulad_nodes["node_type"].value_counts().sort_index().to_dict(),
                "explicit_edge_type_counts": oulad_explicit["edge_type"].value_counts().sort_index().to_dict(),
            },
        }
        (self.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    @staticmethod
    def _write_csv(frame: pd.DataFrame, path: Path) -> None:
        with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False)

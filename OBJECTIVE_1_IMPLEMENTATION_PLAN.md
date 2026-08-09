# Objective 1: Knowledge-Aware Offline Reinforcement Learning Framework

## Objective

Develop a unified knowledge-aware reinforcement-learning framework that combines a knowledge graph, sequential student modelling, and offline policy optimization to recommend adaptive learning paths. EdNet-KT3 is the primary skill-level benchmark; OULAD is a second benchmark for course activity, assessment progression, engagement, and withdrawal behaviour.

The datasets are not concatenated. Separate dataset adapters create the same normalized event schema, after which the same graph, sequence-model, policy, and evaluation interfaces are used. A separate model instance is trained and evaluated for each dataset.

## Phase 1 - Project and experiment setup

1. Create folders for configuration, source code, scripts, tests, outputs, and notebooks.
2. Store dataset paths, preprocessing limits, temporal split ratios, random seeds, model settings, and RL settings in versioned configuration files.
3. Fix random seeds and record the complete configuration for every run.
4. Write outputs into run-specific folders containing profiles, processed events, split manifests, checkpoints, metrics, and logs.
5. Keep downloaded archives and generated outputs out of Git.

**Deliverable:** reproducible project scaffold and a command that runs Phases 1-3.

### Class-based implementation

- `Phase13Pipeline` owns and executes the complete Phase 1-3 workflow.
- `EdNetPreprocessor` owns EdNet paths, sampling limits, and session configuration and exposes `run()`.
- `OULADPreprocessor` owns the OULAD directory, row limit, and chunk size and exposes `run()`.
- `TemporalSplitter` owns split ratios and minimum-history policy and exposes `split()` and `manifest()`.

Small parsing utilities remain private pure functions where they do not need mutable state; dataset preprocessing and orchestration use class APIs.

## Phase 2 - Dataset exploration and validation

### EdNet-KT3

1. Count user files and sampled interactions without extracting the archive.
2. Measure sequence lengths, action frequencies, item types, time range, platforms, and sources.
3. Inspect question and lecture metadata, skill-tag coverage, missing answers, and missing item mappings.
4. Validate timestamp ordering within each student.
5. Record limitations, especially that prerequisite relations and counterfactual recommendations are not observed.

### OULAD

1. Count rows and inspect schemas for all seven tables.
2. Measure students, modules, presentations, VLE activities, assessments, scores, withdrawals, final outcomes, and missing values.
3. Validate joins among module/presentation, student, assessment, and VLE identifiers.
4. Check the relative-day timeline used by OULAD.

**Deliverable:** machine-readable JSON profiles for both datasets.

## Phase 3 - Preprocessing and temporal splitting

### Common event schema

Every adapter produces:

| Field | Meaning |
|---|---|
| `dataset` | `ednet` or `oulad` |
| `student_id` | Dataset-scoped learner identifier |
| `timestamp` | Sortable time value within the dataset |
| `relative_day` | Relative presentation day when available |
| `item_id` | Question, lecture, activity, assessment, or other resource |
| `item_type` | Semantic resource type |
| `concept_ids` | Semicolon-separated concepts/tags |
| `action_type` | Observed learner action |
| `correctness` | Correctness/mastery proxy in `[0,1]` when observable |
| `score` | Original assessment score when available |
| `elapsed_time_ms` | Duration inferred from enter/quit actions |
| `engagement` | Interaction intensity such as VLE clicks |
| `source` | Dataset-specific interaction source |
| `session_id` | Chronological learning-session identifier |
| `module_id` | Module/presentation identifier |
| `final_response` | Whether an EdNet response is the last response before submission |
| `outcome` | Course outcome when available |

### EdNet adapter

1. Stream user CSVs from `EdNet-KT3.zip`.
2. Sort each user chronologically and create sessions from inactivity gaps.
3. Join questions and lectures to content metadata.
4. Map questions, lectures, explanations, bundles, and skills.
5. Mark the last question response before a bundle submission as the final response.
6. Compare final responses with the answer key.
7. Pair `enter` and `quit` events to estimate time spent.
8. Write compressed normalized event partitions.

### OULAD adapter

1. Join assessment submissions to assessment definitions.
2. Join daily VLE activity to activity metadata.
3. add registration and withdrawal events.
4. Attach module presentation and final outcome.
5. Convert relative presentation days to a stable sortable time index.
6. Write normalized events using the common schema.

### Temporal split

1. Sort each student's events chronologically.
2. Allocate the earliest 70% to training, the next 15% to validation, and the latest 15% to testing.
3. Keep very short histories in training until a separate cold-start protocol is implemented.
4. Fit encoders, graph edges, and statistics using training data only.

**Deliverable:** compressed event files for `train`, `validation`, and `test`, plus a manifest containing counts and time boundaries.

## Phase 4 - Knowledge-graph construction

1. Define EdNet nodes for skills, questions, lectures, explanations, bundles, and parts.
2. Define OULAD nodes for modules, assessments, VLE activities, and activity types.
3. Create explicit metadata edges such as `question-tests-skill`, `lecture-teaches-skill`, and `assessment-belongs-to-module`.
4. Infer prerequisite candidates from training data using temporal precedence, transition support, and conditional mastery improvement.
5. Remove low-support, circular, and contradictory edges.
6. Label inferred edges as empirical rather than ground-truth prerequisites.
7. Export node tables, typed edge tables, mappings, and train-only graph statistics.

## Phase 5 - Student sequence construction

1. Construct histories containing items, concepts, actions, correctness, duration, engagement, and time gaps.
2. Apply a configurable sequence length, truncation, padding, and attention masks.
3. Create next-item, next-action, and correctness targets.
4. Build eligible candidate sets using availability, course context, mastery, and prerequisite constraints.
5. Ensure every feature at time `t` uses only information available at or before `t`.

## Phase 6 - Knowledge-aware student-state model

1. Encode graph nodes with a relational GNN, GraphSAGE, GAT, or heterogeneous graph Transformer.
2. Encode chronological learner histories with a Transformer.
3. Estimate a concept-mastery vector.
4. Fuse sequence state, graph context, and mastery using gated fusion or cross-attention.
5. Produce a knowledge-aware state consumed by supervised heads and the RL policy.

## Phase 7 - Supervised pretraining

1. Pretrain next-item, next-action, correctness, masked-item, and optional time-to-event objectives.
2. Evaluate AUC, accuracy, F1, log loss, calibration, and NDCG@K.
3. Save the best validation checkpoint and frozen/finetunable state encoder.

## Phase 8 - Offline RL environment

1. Define state as the knowledge-aware learner representation, mastery, engagement, correctness, and time context.
2. Define EdNet actions as eligible questions, lectures, or explanations.
3. Define OULAD actions at the supported activity/resource-category level.
4. Convert histories into `(state, action, reward, next_state, done)` transitions.
5. Use sessions or student-module presentations as episodes.
6. End episodes at withdrawal, module completion, history end, or maximum horizon.
7. Begin with a simple correctness/engagement reward; replace it with Objective 2's mastery-oriented reward.

## Phase 9 - Behaviour-policy estimation

1. Train a separate d3rlpy discrete behaviour-cloning model on each dataset's logged actions.
2. Estimate action support and historical action probabilities.
3. prevent the learned policy from selecting poorly supported out-of-distribution actions.
4. Track divergence between learned and logged behaviour.

## Phase 10 - Offline policy training

1. Establish behaviour cloning as the first policy baseline.
2. Train d3rlpy Discrete CQL. Evaluate discrete IQL only through a separately implemented and tested algorithm because d3rlpy's IQL is continuous-action only.
3. Optionally evaluate discrete BCQ.
4. Avoid online PPO unless a validated interactive student simulator becomes available.

## Phase 11 - Baselines

Compare against popularity, matrix factorization, DKT, SAKT, AKT, a sequential Transformer, graph-only modelling, behaviour cloning, RL without the graph, RL without prerequisites, and the complete framework.

## Phase 12 - Offline evaluation

1. Predictive: AUC, accuracy, F1, log loss, calibration.
2. Ranking: Hit Rate@K, Recall@K, NDCG@K, MRR.
3. Educational proxies: mastery consistency, prerequisite-violation rate, concept coverage, and repetition.
4. Policy evaluation: Fitted Q Evaluation and, where assumptions are supportable, importance-sampling and doubly robust estimators.
5. Report that historical offline evaluation does not establish causal learning improvement.

## Phase 13 - Ablation and robustness

Remove the graph, inferred prerequisites, Transformer, time features, or candidate masking in turn. Test sequence lengths, sparse/long histories, unseen students, temporal periods, graph encoders, and both datasets independently.

## Implementation order

1. Phases 1-3 foundation.
2. EdNet knowledge graph.
3. Sequence builder and Transformer baseline.
4. Graph encoder and fused model.
5. Supervised pretraining.
6. Offline transition builder and behaviour cloning.
7. Conservative offline-RL policy.
8. Evaluation and ablations.
9. OULAD replication through the shared interfaces.

## Current execution mode

Phases 1-4 have now completed at full scale. The earlier smoke pipeline remains available for rapid development checks, while the validated partitioned artifacts under `outputs/full_preprocessing` are the authoritative inputs for subsequent phases. Phase 4 graph artifacts are under `outputs/phase4_graphs`.

## Full-scale preprocessing resource estimate

These estimates were calculated from the actual downloaded archives and the measured memory/output size of the successful smoke run. They are planning estimates rather than exact guarantees because sequence length and compression vary between students.

### Source scale

- EdNet-KT3: 727.44 MB compressed, 3.49 GiB of CSV content, and 297,915 user files.
- OULAD: approximately 443 MB extracted, including 10,655,280 VLE rows.
- Current free workspace drive capacity at measurement time: approximately 7.68 GiB.

### Estimated normalized output

| Dataset | Estimated events | Uncompressed CSV | Compressed CSV.GZ |
|---|---:|---:|---:|
| EdNet-KT3 | about 89.1 million | about 9.85 GiB | about 0.99 GiB |
| OULAD | about 10.87 million | about 1.55 GiB | about 0.11 GiB |
| Combined | about 100 million | about 11.4 GiB | about 1.1 GiB |

Keeping the source archives, extracted OULAD tables, compressed processed partitions, manifests, and temporary checkpoints should require approximately 2.3-3.0 GiB in total. A minimum of 4 GiB free space is recommended; 6 GiB or more gives safer room for interrupted-run files and validation outputs. Writing the normalized datasets uncompressed is not suitable for the current drive capacity.

### RAM

The smoke output used approximately 13.61 MB of pandas memory for 24,483 EdNet events and 214.31 MB for 316,532 OULAD events. Linear projection gives approximately 48 GiB for the complete EdNet dataframe and 7 GiB for OULAD. The current implementation would additionally create sorting, concatenation, and split copies, producing an unsafe estimated peak of approximately 90-120 GiB RAM.

The planned bounded-memory implementation will process EdNet users and OULAD rows in partitions and write each completed partition immediately. Its target is 1-2 GiB peak RAM, with a conservative recommended machine configuration of 4 GiB available RAM for preprocessing. This estimate covers preprocessing only; later Transformer, graph-neural-network, and reinforcement-learning training will have separate CPU/GPU memory requirements.

## Research-grounded preprocessing decisions

The preprocessing contract was checked against the official dataset descriptions before the full-scale run.

### EdNet

- The [official EdNet repository](https://github.com/riiid/ednet) states that KT2/KT3 may contain repeated `respond` actions and that only the last response before `submit` is treated as the submitted response. The implementation therefore marks the last response for each question in the submitted bundle as `final_response` before comparing it with `correct_answer`.
- The repository states that explanation and lecture study time can be inferred by subtracting paired `enter` and `quit` timestamps. The implementation records this value on the closing event as `elapsed_time_ms`.
- EdNet timestamps are shifted for privacy. They preserve each learner's order but should not be interpreted as synchronized real-world time across learners. Consequently, event holdouts are chronological within each learner; we do not claim that a cross-user calendar split represents a real deployment date.
- KT3 does not provide an explicit session identifier. A configurable 30-minute inactivity threshold is used as a modelling heuristic, not a dataset fact. Clickstream research reports 30 minutes as a common educational-session definition, while broader session-identification research warns that fixed thresholds are inherently approximate. We will later report sensitivity at 15, 30, and 60 minutes rather than treating 30 minutes as ground truth: [educational clickstream example](https://link.springer.com/article/10.1007/s10639-023-12372-6), [session-identification study](https://arxiv.org/abs/1411.2878).

### OULAD

- The [official OULAD data descriptor](https://www.nature.com/articles/sdata2017171) defines a learning record by the `student-module-presentation` triplet. Full preprocessing therefore scopes `student_id` to this enrolment triplet rather than merging separate course presentations into one episode.
- OULAD VLE interactions are daily click summaries. The pipeline preserves `sum_click` as engagement and uses a day-scoped `session_id`; it does not invent within-day click order or a sub-day session duration.
- Assessment scores range from 0 to 100, and the official descriptor interprets scores below 40 as fail. `score` preserves the original mark; `correctness` is the documented binary pass indicator (`score >= 40`). A later mastery model may derive a continuous normalized-score feature from `score` without changing the raw field.
- `is_banked` means that an assessment result was transferred from a previous presentation. The pipeline preserves this flag and labels the action `assessment_banked` so it can be excluded from current-presentation learning-gain rewards or evaluated separately.
- Registration, unregistration, assessment submission, and VLE dates are relative days within a module presentation. The pipeline preserves `relative_day` and creates a deterministic presentation-aware sort index; it does not present that index as an observed wall-clock timestamp.
- The official descriptor warns that B and J presentations may have different structures and recommends analyzing them separately. Module and presentation identifiers therefore remain in every event, and later evaluation will be stratified by presentation type.

### Leakage controls

- Splits are assigned before sequence windows, graph edges, scalers, vocabularies, or reward statistics are fitted.
- Each enrolment/student is ordered chronologically, with early events used for training and later events for validation/test.
- Knowledge-graph prerequisite inference will use training interactions only.
- Final course outcome and withdrawal fields are retained as labels/analysis fields but must not enter a policy state before they become observable.

## Full-scale execution result

The bounded-memory pipeline completed both downloaded datasets and passed a full read-back validation of every generated partition.

| Dataset | Units | Normalized events | Partitions | Validated files | Compressed output |
|---|---:|---:|---:|---:|---:|
| EdNet-KT3 | 297,915 students | 89,270,654 | 298 | 894 | 998.81 MB |
| OULAD | 32,587 enrolments | 10,871,812 | 33 | 99 | 63.60 MB |
| Combined | - | 100,142,466 | 331 | 993 | 1,062.41 MB |

EdNet split counts are 62,360,581 training, 13,391,181 validation, and 13,518,892 test events. OULAD split counts are 7,599,060 training, 1,629,456 validation, and 1,643,296 test events. Both manifests have `status: complete`, validation event totals exactly match written event totals, and OULAD's temporary SQLite staging database was removed after validation.

## Phase 4 - Knowledge-graph execution result

Phase 4 separates official metadata relationships from interaction-derived prerequisite hypotheses. This distinction is essential: chronological association and performance differences do not establish that one skill causally requires another.

### Research basis and assumptions

- The [AAAI 2024 causal skill-prerequisite study](https://ojs.aaai.org/index.php/AAAI/article/view/30046) motivates using learner-response evidence while accounting for latent mastery and causal uncertainty.
- Work on [inferring concept prerequisites from educational resources](https://arxiv.org/abs/1811.12640) and [textbook concept order](https://arxiv.org/abs/2011.10337) shows that ordering and semantic features can provide evidence, but normally require labels or additional validation. EdNet has no authoritative prerequisite labels, so the derived relationships are explicitly named `empirical_prerequisite_candidate`.
- The [EDM 2023 knowledge-tracing causal-discovery model](https://educationaldatamining.org/EDM2023/proceedings/2023.EDM-posters.42/index.html) further supports treating response-derived structure as a discovery problem rather than ground truth.
- Only EdNet training events are used for prerequisite-candidate inference. Validation and test interactions never influence graph selection.
- OULAD provides module, presentation, assessment, VLE-resource, and activity-type metadata but no skill taxonomy. It therefore receives an explicit educational-resource graph without invented skill or prerequisite nodes.

### Explicit graph structure

The EdNet graph contains question, bundle, explanation, lecture, part, and skill nodes. Its metadata-derived edge types are `tests`, `teaches`, `explained_by`, `belongs_to_bundle`, and `belongs_to_part`.

The OULAD graph contains module, module-presentation, assessment, assessment-type, VLE-activity, and activity-type nodes. Its edge types are `presentation_of`, `belongs_to_presentation`, `has_assessment_type`, and `has_activity_type`.

| Graph | Nodes | Explicit edges | Validation |
|---|---:|---:|---|
| EdNet | 33,561 | 66,672 | Passed |
| OULAD | 6,622 | 13,162 | Passed |

### EdNet prerequisite candidates

The inference scan used 12,536,505 final-response training events from 295,839 learner sequences. For every ordered skill pair observed in adjacent final-response events, it records:

- forward and reverse transition support;
- direction confidence `forward / (forward + reverse)`;
- support following a correct versus incorrect response on the source skill;
- probability of a correct target response under those two conditions; and
- their difference, recorded as `performance_lift`.

The default filter requires at least 100 forward transitions, at least 30 observations in each previous-correctness group, direction confidence of at least 0.65, and performance lift of at least 0.05. These are conservative, configurable modelling thresholds rather than facts supplied by EdNet and must be included in later sensitivity analysis.

The scan produced 34,034 measured directed pairs. Of these, 484 passed all thresholds. A greedy, confidence-ordered cycle check rejected 11 edges that would create cycles, leaving a validated directed acyclic graph of 473 empirical candidate edges. The complete measured candidate table is retained so that filtering choices remain reproducible and auditable.

### Phase 4 artifacts

- `outputs/phase4_graphs/manifest.json`: methodology, thresholds, counts, and validation results.
- `outputs/phase4_graphs/ednet/nodes.csv.gz`: EdNet graph nodes.
- `outputs/phase4_graphs/ednet/edges_explicit.csv.gz`: official metadata edges.
- `outputs/phase4_graphs/ednet/prerequisite_candidates.csv.gz`: all measured train-only candidate pairs and raw metrics.
- `outputs/phase4_graphs/ednet/edges_prerequisite_dag.csv.gz`: thresholded acyclic candidate subset.
- `outputs/phase4_graphs/oulad/nodes.csv.gz` and `edges_explicit.csv.gz`: validated OULAD educational-resource graph.

No candidate edge is claimed to be a causal or expert-verified prerequisite. Phase 12 evaluation and Phase 13 ablation must compare explicit-only, candidate-enhanced, and threshold-sensitivity variants.

## Next stage - Phase 5

Phase 5 has completed at full scale. The next stage is now Phase 6, the knowledge-aware student-state model.

## Phase 5 - Sequence and candidate-set execution result

Phase 5 converts the normalized chronological events into compact model-ready learner sequences. It stores each event once and records fixed-length window indices separately, avoiding the storage explosion that would result from materializing an overlapping 128-token history for every target.

### Research basis and assumptions

- [SAKT](https://arxiv.org/abs/1907.06837) models knowledge state from earlier question-answer interactions using self-attention and motivates attending selectively to relevant knowledge concepts.
- [SASRec](https://arxiv.org/abs/1808.09781) uses causally masked recent interaction histories for next-item prediction and supports truncating long histories to a fixed recent window.
- The [AAAI offline-RL recommendation framework](https://ojs.aaai.org/index.php/AAAI/article/view/16579) highlights distribution mismatch and action-support constraints. Candidate catalogs therefore retain logged training support, allowing later policies to exclude poorly supported actions.
- The configured maximum length of 128 and stride of 64 are engineering/model assumptions, not dataset facts. Phase 13 must compare shorter and longer histories.
- Stable item and concept identities are seeded from the official EdNet and OULAD catalogs. This does not use held-out learner behaviour. Action support, interaction frequencies, and non-catalog behavioral identifiers are fitted only on training events.

### Packed sequence schema

Each partition contains typed arrays for item, action, item type, module, source, timestamp, relative day, time gap, correctness, score, elapsed time, engagement, final-response status, banked-assessment status, and original split. Multi-concept events use compressed sparse-row offsets and concept-token values. Individual EdNet questions/lectures have at most seven metadata tags, while normalized bundle events can contain a union of up to nine concepts; model loaders therefore use a validated width of nine.

Time gaps preserve dataset resolution: `log1p(seconds)` for EdNet and `log1p(days)` for OULAD. Elapsed duration is stored as `log1p(seconds)` and engagement as `log1p(nonnegative engagement)`.

The lazy loader returns:

- right-padded causal inputs and attention masks;
- next-item, next-action, and next-correctness targets;
- fixed-width concept tensors;
- a target mask that ensures each non-initial learner event is trained or evaluated exactly once; and
- earlier chronological context for validation/test targets without reassigning those context events as held-out targets.

### Full-scale result

| Dataset | Events | Learners | Windows | Packed files | Unknown item rate |
|---|---:|---:|---:|---:|---:|
| EdNet | 89,270,654 | 297,915 | 2,078,098 | 298 | 0% |
| OULAD | 10,871,812 | 32,587 | 221,606 | 33 | 0% |
| Combined | 100,142,466 | 330,502 | 2,299,704 | 331 | 0% |

EdNet has 62,062,666 training targets, 13,391,181 validation targets, and 13,518,892 test targets. OULAD has 7,566,473 training targets, 1,629,456 validation targets, and 1,643,296 test targets. The difference between training event and target counts is exactly one initial context-only event per learner.

All 331 partitions passed validation for equal array lengths, learner offsets, chronological order, concept offsets, window bounds, split-contiguous target ranges, and exactly-once target coverage. Packed arrays occupy approximately 994 MB; all Phase 5 artifacts together occupy approximately 1.04 GB.

### Candidate catalogs

- EdNet: 21,015 content candidates observed in training; 20,990 have at least five supported observations.
- OULAD: 6,170 module-scoped assessment/VLE candidates observed in training; 5,325 have at least five supported observations.
- Every catalog entry resolves to a Phase 4 graph node.
- OULAD eligibility can be restricted to the current module presentation.
- EdNet eligibility can apply the Phase 4 candidate-prerequisite graph dynamically from the learner's currently mastered skills.
- Prerequisite enforcement remains configurable because the candidate graph is empirical rather than expert-verified.

### Phase 5 artifacts

- `outputs/phase5_sequences/manifest.json`: complete configuration, counts, validation, and unknown rates.
- `outputs/phase5_sequences/{dataset}/vocabularies.json`: stable token mappings.
- `outputs/phase5_sequences/{dataset}/candidate_catalog.csv.gz`: graph-aligned actions with training support and prerequisite metadata.
- `outputs/phase5_sequences/{dataset}/packed/part-*.npz`: compressed event arrays, learner offsets, concept offsets, and lazy window indices.
- `src/edu_recommender/sequence_building.py`: class-based writer, validator, lazy loader, and candidate provider.

## Phase 6 - Knowledge-aware student-state model result

Phase 6 is implemented and has passed real-data optimization, validation, causality, gradient, and checkpoint-reload tests. The current checkpoint is a smoke-trained initialization, not a converged research result; Phase 7 will perform proper supervised pretraining and evaluation.

### Research basis

- [Context-Aware Attentive Knowledge Tracing](https://arxiv.org/abs/2007.12324) motivates attention-based learner-state modelling, temporal context, and explicit concept-level performance prediction.
- [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103) motivates relation-specific transformations for multi-relational knowledge graphs.
- [Gated Multimodal Units](https://arxiv.org/abs/1702.01992) motivates learning multiplicative gates that control how separate information sources contribute to a fused representation.
- The [PyTorch TransformerEncoder documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html) distinguishes causal attention masks from per-example padding masks. The implementation supplies both explicitly and includes a unit test showing that modifying a future token cannot alter an earlier state.

### Implemented architecture

The baseline uses 64-dimensional states, four attention heads, two causal Transformer layers, two relational graph layers, a 128-dimensional feed-forward block, and dropout of 0.1.

Sequence inputs combine item, action, item type, multi-concept, module, source, correctness, position, time-gap, elapsed-time, engagement, score, and relative-day features. A strict upper-triangular attention mask prevents future-event leakage, while a separate padding mask excludes padded positions.

The graph encoder:

- initializes trainable node and node-type embeddings;
- includes official Phase 4 relations and the empirical prerequisite-candidate relation for EdNet;
- adds a separate reverse relation for every directed relation;
- performs relation-specific message transformation, normalized neighborhood aggregation, self transformation, residual connection, GELU, dropout, and layer normalization; and
- maps item and concept tokens back to graph nodes for every history position.

The mastery head estimates a probability for every concept at every time step. Gated fusion combines the causal history state, item/concept graph context, and mastery-weighted concept context. The fused student state feeds tied-embedding next-item logits, a next-action head, and a correctness head.

The supervised smoke objective combines next-item cross-entropy, next-action cross-entropy, correctness binary cross-entropy, and concept-level mastery binary cross-entropy. Mastery supervision uses only the next event's observed concepts and correctness.

### Runtime and smoke benchmark

PyTorch 2.13.0 CPU was installed in the project-local `.venv`. CUDA is not available on the current machine, so the benchmark used two CPU threads, batch size 2, and three optimization steps per dataset.

| Dataset | Parameters | Graph nodes | Directed graph edges | Mean train step | Validation total loss |
|---|---:|---:|---:|---:|---:|
| EdNet | 4,573,367 | 33,561 | 134,290 | 0.92 s | 11.4905 |
| OULAD | 1,030,422 | 6,622 | 26,324 | 0.14 s | 9.9105 |

The EdNet parameters occupy approximately 18.29 MB as float32 values; OULAD parameters occupy approximately 4.12 MB. Observed process memory reached approximately 476 MB after the EdNet run and 521 MB by the end of both sequential runs. The checkpoints are approximately 55.2 MB and 12.5 MB because they include model and AdamW optimizer state.

All losses were finite, all parameter groups received gradients, gradient clipping executed, validation inference completed, and both checkpoints reload successfully. Fifteen project tests pass. Because only three updates were performed, these loss values demonstrate executable learning behavior rather than model quality or convergence.

### Phase 6 artifacts

- `src/edu_recommender/student_state_model.py`: graph tensor builder, R-GCN layers, causal history model, mastery/fusion architecture, heads, and multitask loss.
- `src/edu_recommender/phase6_training.py`: real packed-window batching, CPU/GPU selection, optimization, validation, memory measurement, and checkpoint writing.
- `configs/phase6_model.json`: architecture, loss, optimizer, batch, and smoke settings.
- `scripts/run_phase6_smoke.py`: executable smoke benchmark.
- `tests/test_phase6_model.py`: shapes, finite loss, gradients, and causal future-token isolation tests.
- `outputs/phase6_model/manifest.json`: complete measured benchmark results.
- `outputs/phase6_model/ednet_smoke_checkpoint.pt` and `oulad_smoke_checkpoint.pt`: reload-validated smoke checkpoints.

## Next stage - Phase 7

The Phase 7 supervised-pretraining code is implemented and validated. After explicit authorization, a bounded micro-pilot was executed to verify the complete optimization, validation, metric, checkpoint, and comparison workflow. These checkpoints are pilot artifacts, not converged models.

### Implemented training controls

- Epoch-seeded shuffled training windows and deterministic validation windows.
- Hard limits on training and validation windows for controlled pilot experiments.
- Strict rejection of test-split model selection.
- AdamW optimization, gradient clipping, linear warm-up, and cosine learning-rate decay.
- Validation-total-loss checkpoint selection with configurable early stopping and minimum improvement.
- Atomic `last_checkpoint.pt` and `best_checkpoint.pt` writing.
- Model, optimizer, scheduler, epoch, global-step, history, best-metric, and random-state restoration.
- Dataset/variant-specific resume-checkpoint routing.
- CPU/GPU selection and runtime/memory profiling.

The warm-up/decay implementation follows the [PyTorch LambdaLR contract](https://docs.pytorch.org/docs/2.13/generated/torch.optim.lr_scheduler.LambdaLR.html), including stepping after the optimizer and saving both optimizer and scheduler state.

### Implemented evaluation

- Validation losses for next item, next action, correctness, mastery, and their weighted total.
- Correctness and mastery ROC-AUC, accuracy, precision, recall, F1, Brier score, and expected calibration error.
- Candidate-supported next-item Hit Rate@5/10/20, NDCG@5/10/20, and mean reciprocal rank.
- Next-action accuracy.
- Ranking limits that bound the cost of candidate evaluation on CPU.
- Popularity ranking computed only from Phase 5 training support.

### Implemented comparisons

The code supports `popularity`, `sequence_only`, `graph_only`, `sequence_graph`, `sequence_mastery`, and `full`. The initial controlled configuration schedules popularity, sequence-only, and full comparisons on both EdNet and OULAD. Additional ablations can be enabled through configuration without changing model code.

### Safety against accidental training

`scripts/run_phase7_pretraining.py` performs configuration and path validation by default. Actual optimization requires the explicit `--execute` flag. The validated default plan is five epochs with at most 10,000 training windows and 2,000 validation windows per epoch, batch size 2, on the CPU-only runtime. This is a pilot configuration, not a full 2.3-million-window run.

### Phase 7 code artifacts

- `src/edu_recommender/phase7_pretraining.py`: streams, metrics, ranking, early stopping, schedule, checkpointing, popularity evaluation, and training orchestration.
- `configs/phase7_pretraining.json`: controlled pilot and comparison configuration.
- `scripts/run_phase7_pretraining.py`: validation-only-by-default entrypoint with explicit execution guard.
- `tests/test_phase7_pretraining.py`: non-training tests for AUC/calibration, early stopping, and warm-up/cosine behavior.
- `src/edu_recommender/student_state_model.py`: configurable fusion variants for the Phase 7 ablation comparisons.

Eighteen project tests pass, and every required Phase 4/5 input path validates.

### Executed Phase 7 micro-pilot

The first 250-window attempt was stopped before its first neural checkpoint after the measured CPU runtime exceeded ten minutes. A separate initial attempt exposed that normalized EdNet bundle events can contain nine concept tags rather than the eight allowed by the first loader width. A full packed-data audit confirmed maxima of nine for EdNet and one for OULAD; the loader and configurations now use the validated width of nine.

The completed micro-pilot used one epoch, 20 training windows, 10 validation windows, batch size 2, and 20 candidate-ranking examples. This corresponds to 10 optimizer steps for each neural variant. Test data was not read.

| Dataset | Variant | Optimizer steps | Runtime | Peak RAM | Validation total loss |
|---|---|---:|---:|---:|---:|
| EdNet | Sequence only | 10 | 97.46 s | 506.56 MB | 11.3711 |
| EdNet | Full fused | 10 | 76.80 s | 494.95 MB | 11.3911 |
| OULAD | Sequence only | 10 | 56.04 s | 434.19 MB | 9.5638 |
| OULAD | Full fused | 10 | 56.40 s | 418.23 MB | 9.3652 |

Each neural experiment produced a `best_checkpoint.pt`, `last_checkpoint.pt`, training history, validation metrics, and runtime profile. All four best checkpoints were reconstructed into their exact model variants and loaded with every state-dictionary key matching.

The micro-pilot is too small for model comparison or educational conclusions. For example, the selected OULAD validation subset contained only one correctness class, so ROC-AUC is correctly reported as undefined. EdNet candidate Hit Rate@20 was zero for all pilot models, while the OULAD popularity baseline performed strongly on only 20 ranking examples. These are pipeline checks, not publishable results.

Artifacts are stored under `outputs/phase7_pretraining_micro_pilot`. The complete comparison file is `comparison_summary.json`; dataset/variant subdirectories contain their checkpoints and metrics.

## Phase 8 - Offline environment implementation result

The EdNet and OULAD offline-environment implementations have been completed in parallel with the remaining Phase 7 training work. They use Gymnasium-compatible interfaces but deliberately operate as strict logged-trajectory replayers: if a caller requests an action different from the historical action, the environment raises a counterfactual-action error instead of fabricating an unobserved student response.

### Implemented environment contract

- A shared `BaseOfflineEducationEnv` with dataset-specific `EdNetOfflineEnv` and `OULADOfflineEnv` subclasses.
- A structured observation containing the 64-dimensional student state, explicit mastery probabilities, eight recent correctness/score/engagement/time features, module context, and a dynamic action mask.
- EdNet `Discrete(20,990)` supported actions: 11,551 questions, 8,483 explanations, and 956 lectures.
- OULAD `Discrete(5,325)` supported actions: 5,152 VLE resources and 173 assessments, dynamically restricted to the learner's module presentation.
- Train-support filtering, optional EdNet prerequisite masking, configurable mastery thresholds, repetition handling, and compressed eligible-action indices.
- EdNet session segmentation using inactivity gaps and OULAD student-module-presentation episodes, with maximum-horizon truncation.
- Correct EdNet decision timing: a question recommendation state ends before the first response in its contiguous response block, preventing current-question responses from leaking into the state.
- Logged transitions containing observation, action, configurable reward, next observation, termination/truncation flags, and provenance metadata.
- A configurable preliminary reward with mastery progression, correctness, score, engagement, time cost, prerequisite violation, repetition, and OULAD withdrawal components. These weights are Phase 8 engineering defaults and are not yet the validated Objective 2 reward.
- A deterministic future-safe provisional encoder for environment development and a CPU-compatible `TorchStudentStateEncoder` that strictly loads Phase 7 checkpoints.

### Real-data validation

The bounded validation built and replayed four real training episodes per dataset:

| Dataset | Episodes | Transitions | Action count | Logged actions forced into mask | Validation |
|---|---:|---:|---:|---:|---|
| EdNet | 4 | 321 | 20,990 | 0 | Passed |
| OULAD | 4 | 486 | 5,325 | 0 | Passed |

The checkpoint adapter was also loaded separately for both pilot checkpoints and produced finite student-state, mastery, and recent-feature arrays with shapes `(64, 296, 8)` for EdNet and `(64, 9, 8)` for OULAD. The Phase 8 mechanics remain covered by seven tests for future safety, EdNet decision timing, prerequisite/repetition filtering, module masking, dataset separation, logged replay, and rejection of unsupported counterfactual actions.

### Phase 8 artifacts

- `src/edu_recommender/offline_rl_environment.py`: environments, spaces, encoders, action catalog, transition builder, reward, episode logic, and validation.
- `configs/phase8_offline_environment.json`: shared environment and preliminary reward configuration.
- `scripts/run_phase8_environment.py`: bounded real-data builder/replay validator.
- `tests/test_phase8_offline_environment.py`: Phase 8 unit and causality tests.
- `outputs/phase8_offline_environment`: real-data validation manifests.

The environment implementation is complete, but the current validation manifests intentionally use provisional history-statistics states. Full transition materialization and any behavior-policy or offline-policy training must wait for a meaningfully trained, validation-selected Phase 7 checkpoint; the transition artifacts must record that checkpoint's identity.

## Phase 9 - d3rlpy integration result

d3rlpy 2.8.1 is installed and locked as the offline-RL framework. The integration supports two separate native discrete datasets and later two separate policies: EdNet and OULAD are never combined because their observation dimensions, action catalogs, and learner semantics differ.

Implemented components:

- A `D3RLPYTransitionAdapter` that converts Phase 8 episodes into native `MDPDataset` arrays.
- Compact float observations containing the state-model output, mastery, recent features, and one-hot module context. The large dynamic action mask is deliberately excluded from the neural input and retained as sparse recommendation metadata.
- Untrained d3rlpy `DiscreteBC` and `DiscreteCQL` constructors with explicit hyperparameter configuration.
- A masked recommendation selector. CQL ranks only eligible actions with Q-values; BC applies the eligibility mask to its categorical logits before argmax.
- An explicit training guard in the preparation command. It validates and writes development artifacts but never calls `fit` or `fit_online`.

### Real-data d3rlpy validation

| Dataset | Source rows | Native trainable transitions | Episodes | Observation size | Action size |
|---|---:|---:|---:|---:|---:|
| EdNet | 321 | 320 | 4 | 370 | 20,990 |
| OULAD | 486 | 483 | 4 | 105 | 5,325 |

The difference between source rows and native transitions is expected: d3rlpy omits the final bootstrapping transition at artificial timeout boundaries. Both BC and CQL objects were constructed successfully for both native datasets, but neither was fitted. All 30 project tests pass.

### Phase 9 integration artifacts

- `src/edu_recommender/d3rlpy_adapter.py`: native dataset conversion, algorithm constructors, and masked selection.
- `configs/phase9_d3rlpy.json`: BC/CQL configuration with `training_enabled: false`.
- `scripts/prepare_phase9_d3rlpy.py`: bounded real-data preparation and compatibility validation.
- `tests/test_d3rlpy_adapter.py`: conversion, one-hot module context, action safety, construction, and masked-CQL tests.
- `outputs/phase9_d3rlpy`: development arrays and validation manifest.

These artifacts still use the provisional state encoder and therefore must not be used for actual BC/CQL fitting or accuracy claims.

## Next action

The next action remains a larger but controlled Phase 7 experiment. Linear extrapolation from the CPU micro-pilot suggests that the existing five-epoch, 10,000-training-window configuration across both neural variants and datasets could require roughly 130-160 CPU hours. Before authorizing that run, the training code should avoid recomputing unused graph branches for ablation variants and consider sampled/adaptive item softmax or candidate-restricted training. A 250-window, one-dataset, one-variant timing run is the safer next scale checkpoint. After validation selects an encoder checkpoint, regenerate Phase 8/9 arrays with that checkpoint, train BC first, and then train Discrete CQL. Discrete IQL remains a separate custom work item.

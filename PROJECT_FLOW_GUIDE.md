# Knowledge-Aware Educational Recommender: End-to-End Flow

This project learns to recommend the next useful educational resource from historical learner interactions. EdNet-KT3 and OULAD use the same pipeline interfaces but are processed, trained, and evaluated separately because their actions and educational meanings differ. EdNet focuses on questions, lectures, explanations, and skills; OULAD focuses on assessments, virtual-learning-environment resources, modules, engagement, and withdrawal.

## Section 1 — Model Creation: Dataset to Trained Policy

### 1. Dataset preprocessing

Raw interactions are converted to a common chronological event schema. Important fields include learner ID, timestamp, item ID, item type, concepts, action type, correctness, score, duration, engagement, module, and outcome. EdNet repeated question responses are handled so that the final response before submission is the observed question outcome. OULAD interactions remain scoped to a student-module-presentation enrolment.

Each learner's events are sorted chronologically and divided into approximately 70% training, 15% validation, and 15% test data. Training comes first in time. Vocabularies, action support, graph statistics, prerequisite candidates, and model-selection decisions are fitted using training data only. This prevents future information from leaking into past states.

### 2. Knowledge-graph construction

The EdNet graph contains questions, lectures, explanations, bundles, parts, and skills. Typed edges represent relations such as `tests`, `teaches`, `explained_by`, and `belongs_to_part`. Possible skill prerequisites are inferred only from training interactions using transition support, temporal direction, and performance lift. They are empirical candidates, not verified causal facts.

The OULAD graph connects modules, presentations, assessments, VLE activities, assessment types, and activity types. It does not invent a skill taxonomy that is absent from the source data.

The graph gives related resources shared representations and later supports eligibility rules, prerequisite checks, graph-aware student states, and educational evaluation.

### 3. Sequence construction

Chronological events are tokenized and packed into bounded learner-history windows. A window contains item, action, item type, concepts, module, source, correctness, time gap, elapsed time, engagement, score, and relative day. The configured maximum window length is 128 events. The initial event supplies context; later events become prediction targets. Causal attention ensures that the representation at time `t` cannot inspect later outcomes.

An action catalog is also created from resources with sufficient training support. With the current minimum support of five, EdNet has 20,990 actions and OULAD has 5,325 actions.

### 4. Student-state model

The sequence branch embeds categorical inputs, projects numerical inputs, and passes them through a two-layer causal Transformer. Its output at each history position is 64-dimensional.

The graph branch initializes node and node-type embeddings and applies two relational graph layers. Relation-specific message passing creates a 64-dimensional graph context for the items and concepts appearing in the learner history.

A mastery head estimates a probability for every concept: 296 entries for EdNet and 9 for OULAD in the current vocabularies. A learned gate fuses three sources—causal history, graph context, and mastery-weighted concept context—into the final 64-dimensional student state.

### 5. Supervised pretraining

Before reinforcement learning, the encoder is trained on observable targets: next item, next action, next-response correctness, and concept mastery. The training objective combines cross-entropy and binary cross-entropy losses. Validation loss selects the checkpoint, while ranking and calibration metrics measure representation quality. The purpose is to make the 64-dimensional state informative before it is frozen or fine-tuned for policy learning.

### 6. Offline-RL dataset and policy training

Historical sequences are converted into transitions:

`(state, logged action, engineered reward, next state, terminated/truncated)`

The action is the next supported resource recorded in the historical event log. It is not necessarily proof of an explicit platform recommendation; it may reflect student choice, course design, teacher direction, or platform ordering. The reward combines estimated mastery progression, correctness, score, engagement, time cost, prerequisite violations, repetition, and withdrawal, then clips the result to `[-1, 1]`.

Behaviour Cloning learns to reproduce logged actions. Implicit Q-Learning would emphasize logged actions with above-expected return, but discrete IQL still requires a custom implementation. Discrete Conservative Q-Learning estimates long-term action values while penalizing unsupported high values. In the current repository, BC and CQL integration is complete but meaningful fitting has not yet been run; discrete IQL is planned.

<div style="page-break-after: always;"></div>

## Section 2 — Inference: Input and Output of Each Layer

Inference begins with a new learner's observable history. No future correctness, outcome, or withdrawal information may be included.

| Layer | Input | Output |
|---|---|---|
| Preprocessing/tokenization | Raw chronological learner events | Item, action, type, concept, module, source, time, correctness, score, and engagement tensors |
| Sequence embeddings | Categorical tokens and six numerical features | One 64-dimensional embedding per history position |
| Causal Transformer | Up to 127 prior embedded events plus masks | A 64-dimensional chronological state per position |
| Graph lookup and relational encoder | Relevant item/concept graph nodes and typed edges | A 64-dimensional graph context per position |
| Mastery head | Transformer history states | Concept-mastery probabilities: 296 for EdNet or 9 for OULAD |
| Gated fusion | History state, graph context, and mastery context | Final 64-dimensional student state |
| RL observation adapter | Student state, mastery, eight recent features, and one-hot module | 370-dimensional EdNet input or 105-dimensional OULAD input |
| BC policy | Flattened RL observation | One categorical logit/probability per discrete action |
| IQL policy, when implemented | Observation plus learned advantage information | One policy score/probability per discrete action |
| CQL critic | Flattened RL observation and candidate action | Estimated long-term Q-value for the action |
| Eligibility selector | Policy scores/Q-values plus eligible-action indices | Highest-scoring eligible action index |
| Action catalog | Selected action index | Final question, lecture, explanation, VLE activity, or assessment ID |

The fixed Gymnasium action space never changes: `Discrete(20,990)` for EdNet and `Discrete(5,325)` for OULAD. A dynamic mask selects a state-dependent legal subset. OULAD always restricts resources to the current module. Optional rules can require prerequisite mastery or prevent immediate repetition. At selection time, BC applies the mask before choosing the largest logit, while CQL compares Q-values only among eligible candidates.

The offline environment cannot generate a genuine outcome for a newly proposed action. During development it strictly replays the logged action and rejects a different action because no validated student simulator exists. Real deployment would return the selected resource to the application, observe the learner's later interaction, and use that new event to construct the next state.

## Section 3 — Evaluation Metrics

### Predictive and representation metrics

- Next-response ROC-AUC, accuracy, precision, recall, and F1
- Log loss or binary cross-entropy
- Brier score and expected calibration error
- Next-action accuracy
- Validation total loss for checkpoint selection

These metrics test whether the state encoder captures learner behaviour and knowledge. They do not by themselves prove that recommendations improve learning.

### Ranking metrics

- Hit Rate@K: whether the logged relevant item appears in the top `K`
- Recall@K: fraction of relevant items retrieved in the top `K`
- NDCG@K: ranking quality with greater credit for higher positions
- Mean Reciprocal Rank: average reciprocal position of the first relevant item

Ranking must use supported and eligible candidates rather than treating impossible resources as valid recommendations.

### Educational-proxy metrics

- Estimated mastery progression and mastery consistency
- Prerequisite-violation rate
- Concept coverage
- Immediate repetition rate
- Correctness, score, engagement, and withdrawal-related outcomes

These are educational proxies, not direct causal measurements of learning gain. Reward-component results should be reported separately so that a single combined reward does not hide undesirable behaviour.

### Offline-policy and safety metrics

- Behaviour-policy likelihood and divergence from logged behaviour
- Action-support coverage and out-of-distribution action rate
- Fitted Q Evaluation
- Importance sampling or doubly robust evaluation only when their assumptions and behaviour probabilities are supportable
- Reward distribution, episode return, termination, and truncation statistics
- Logged actions forced back into an eligibility mask

Finally, ablations should compare sequence-only, graph-only, graph-plus-sequence, graph-plus-mastery, RL without prerequisite masking, BC, IQL when available, and CQL. Results must be evaluated separately for EdNet and OULAD, and historical offline results must not be presented as proof of causal student improvement.

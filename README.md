# Knowledge-Aware Educational Recommender

A research pipeline for learning educational recommendations from historical learner interactions. It normalizes EdNet-KT3 and OULAD events, builds dataset-specific knowledge graphs and chronological learner sequences, learns a graph-aware student-state representation, and prepares offline reinforcement-learning datasets for Behaviour Cloning (BC) and Conservative Q-Learning (CQL).

EdNet and OULAD use the same pipeline interfaces, but they are processed, trained, and evaluated separately because their learners, resources, actions, and outcomes are unrelated.

## Current status

- Phases 1-5: preprocessing, temporal splitting, graph construction, and sequence packing
- Phases 6-7: student-state model and supervised pretraining
- Phase 8: Gymnasium-compatible offline replay environment
- Phase 9: d3rlpy dataset preparation for discrete BC and CQL
- Discrete IQL is planned; d3rlpy 2.8.1 does not provide it

This is a research implementation. Offline results are educational proxies and must not be treated as proof of causal learning improvement.

## Repository layout

```text
configs/       Reproducible JSON configurations for each phase
datasets/      Dataset instructions and tracked EdNet content metadata
notebooks/     Space for exploratory notebooks
scripts/       Command-line entry points
src/           Python package (`edu_recommender`)
tests/         Unit and integration tests
outputs/       Generated data, graphs, checkpoints, and reports (ignored)
```

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/) (recommended), or pip
- Enough local disk space for EdNet-KT3 (the archive expands to several GB)
- A CUDA-capable GPU is optional; configuration files can be changed to use CPU

## Setup

Clone the repository and install the locked dependencies:

```powershell
git clone https://github.com/SanghaviHarshPankajkumar/recommendation_system.git
cd recommendation_system
uv sync --dev
```

If `uv` is unavailable:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m pip install "pytest>=8,<10"
```

## Dataset setup

Large raw datasets are intentionally excluded from Git because individual source files exceed GitHub's file-size limit and redistribution is governed by the dataset licenses. The small EdNet content metadata files required by the pipeline are included.

Download the datasets from their official sources:

- EdNet: <https://github.com/riiid/ednet>
- OULAD: <https://analyse.kmi.open.ac.uk/open_dataset>

Place them in this structure:

```text
datasets/
├── ednet/
│   ├── EdNet-KT3.zip
│   └── metadata/contents/
│       ├── lectures.csv
│       └── questions.csv
└── oulad/
    └── raw/
        ├── assessments.csv
        ├── courses.csv
        ├── studentAssessment.csv
        ├── studentInfo.csv
        ├── studentRegistration.csv
        ├── studentVle.csv
        └── vle.csv
```

Keep `EdNet-KT3.zip` compressed: the preprocessing code streams user CSV files from the archive. See [`datasets/README.md`](datasets/README.md) for licensing and dataset-specific notes.

## Run the pipeline

The default configs point to the full pipeline outputs. Start with the smoke preprocessing config when validating a new machine:

```powershell
uv run python scripts/run_phase_1_3.py --config configs/phase1_3.json
uv run python scripts/run_phase4_graph.py --config configs/phase4_graph.json
uv run python scripts/run_phase5_sequences.py --config configs/phase5_sequences.json
uv run python scripts/run_phase6_smoke.py --config configs/phase6_model.json
uv run python scripts/run_phase7_pretraining.py --config configs/phase7_pretraining.json
uv run python scripts/run_phase7_pretraining.py --config configs/phase7_pretraining.json --execute
uv run python scripts/run_phase8_environment.py --config configs/phase8_offline_environment.json
uv run python scripts/prepare_phase9_d3rlpy.py --config configs/phase9_d3rlpy.json
```

For complete preprocessing, use:

```powershell
uv run python scripts/run_full_preprocessing.py --config configs/full_preprocessing.json
```

Each phase expects the preceding phase's output. Generated artifacts go under `outputs/` and are not committed.

## Tests

```powershell
uv run pytest
```

## Working with teammates

Create a feature branch before changing code:

```powershell
git switch -c feature/short-description
git add <files>
git commit -m "Describe the change"
git push -u origin feature/short-description
```

Open a pull request into `master`. Do not commit raw datasets, generated outputs, model checkpoints, virtual environments, or secrets. If you add a new dependency, update both `pyproject.toml` and `uv.lock`.

For the detailed architecture, data flow, inference contract, and evaluation metrics, read [`PROJECT_FLOW_GUIDE.md`](PROJECT_FLOW_GUIDE.md).

## Dataset licenses

- EdNet is published under CC BY-NC 4.0 for research/non-commercial use; verify the upstream terms before use or redistribution.
- OULAD has its own upstream terms; review them on the official dataset page.

Only code and small metadata needed to describe or run the project are versioned here. Each teammate is responsible for downloading the raw datasets from the official sources.

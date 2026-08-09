# Dataset setup for Objectives 1 and 2

Both datasets will be used as separate benchmark environments through a shared data interface. They must not be concatenated because their users, item identifiers, event types, and labels are unrelated.

## EdNet

- Interaction release: `ednet/EdNet-KT3.zip`
- Content metadata: `ednet/metadata/contents/`
- Official source: https://github.com/riiid/ednet
- Intended role: primary benchmark for sequential mastery modelling, skill/tag graph construction, and question/lecture/explanation recommendation.
- Storage choice: KT3 remains compressed because it expands to about 4.3 GB and nearly 298,000 files. The preprocessing pipeline should stream user CSV files from the ZIP.
- License stated by the official repository: CC BY-NC 4.0 (research/non-commercial use).

## OULAD

- Archive: `oulad/anonymisedData.zip`
- Extracted tables: `oulad/raw/`
- Official source: https://research.stem.open.ac.uk/ouanalyse/dataset/
- Intended role: secondary benchmark for engagement, assessment progress, retention/time-to-mastery proxies, and withdrawal-risk reward components.

## Modelling rule

Use a common normalized event schema and the same model/reward APIs, but train and evaluate a separate model instance for each dataset. Dataset-specific adapters will map their different schemas into the common representation.

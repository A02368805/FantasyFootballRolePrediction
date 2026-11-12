# Fantasy Football Role Prediction

One unified project for Natural Language Processing and Artificial Neural Networks courses.

**Goal:** predict whether an NFL RB, WR, or TE will have an *increasing*, *stable*, or *decreasing*
offensive role in their next game, using historical player statistics and natural-language features
from NFL play-by-play descriptions.

## Status

Foundation only. Currently working:

- Download of real 2024 weekly NFL player stats (`nflreadpy`, Polars).
- A real-data chart: top 10 RB/WR/TE by total 2024 regular-season targets.
- Offline pytest suite.

**Not implemented yet:** role labels, feature engineering, TF-IDF, any of the five models, and evaluation.
The corresponding modules in `src/` are placeholders.

## Planned models

1. Statistics-only ML baseline
2. TF-IDF NLP-only baseline
3. Combined statistics + TF-IDF baseline
4. PyTorch neural network (structured features)
5. Combined neural network (statistics + NLP)

Evaluation: accuracy, precision, recall, macro F1, confusion matrices, player usage trends, and training
curves, using a **chronological** train/validation/test split (no future-data leakage).

## Setup

Requires Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Apple Silicon note

If `python3` is an Intel (x86_64) build running under Rosetta, standard Polars crashes with a
"Missing required CPU features" warning. `requirements.txt` therefore installs `polars[rtcompat]`,
which works in both cases. Installing a native arm64 Python (python.org installer or Homebrew under
`/opt/homebrew`) is still recommended for speed.

## First run

```bash
python main.py --check-data
```

This downloads 2024 weekly player stats, prints the row count and column names, saves the chart to
`results/graphs/top_targets_2024.png`, and prints the output location. It needs an internet connection.
A copy of the download is saved to `data/raw/` (git-ignored); if a later download fails, that copy of the
same real data is used with a warning. If the download fails and there is no copy, the command prints a
clear error and exits with a non-zero status.

## Tests

```bash
pytest
```

Tests run offline. They use a tiny hand-written fixture only to check aggregation/filtering logic; it is
never used for charts or results.

## Layout

```
data/raw/            downloaded data (git-ignored)
data/processed/      derived datasets (git-ignored)
src/
  data_loader.py         nflreadpy download -> Polars
  preprocessing.py       (placeholder) cleaning, labels, chronological split
  feature_engineering.py (placeholder) stats + TF-IDF features
  models.py              (placeholder) classical baselines
  neural_network.py      (placeholder) PyTorch models
  evaluation.py          (placeholder) metrics
  visualization.py       charts
models/              saved models (git-ignored)
results/graphs/      generated charts
tests/               pytest tests
main.py              command-line entry point
```

## Notes on the chart

- Player names come from `player_display_name` (falls back to `player_name`); positions from `position`
  (falls back to `position_group`). Players are grouped by `player_id` so same-name players are not merged.
- Only RB/WR/TE rows and regular-season games (`season_type == "REG"`) are counted.

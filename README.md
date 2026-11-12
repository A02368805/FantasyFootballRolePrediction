# Fantasy Football Role Prediction

One unified project for Natural Language Processing and Artificial Neural Networks courses.

**Goal:** predict whether an NFL RB, WR, or TE will have an *increasing*, *stable*, or *decreasing*
offensive role in their next game, using historical player statistics and natural-language features
from NFL play-by-play descriptions.

## Status

Working:

- Real-data smoke test (`--check-data`): 2024 weekly stats and a top-targets chart.
- Historical data preparation (`--prepare-data`): 2022-2025 weekly stats + schedules (cached as Parquet),
  cleaning, schedule-aware next-game lookup, rolling statistical features, experimental role-change labels,
  chronological train/validation/test splits, and three graphs.
- Offline pytest suite.

**Not implemented yet:** play-by-play text download/TF-IDF, any of the five models, and evaluation.
`models.py`, `neural_network.py` and `evaluation.py` are placeholders.

## Planned models

1. Statistics-only ML baseline
2. TF-IDF NLP-only baseline
3. Combined statistics + TF-IDF baseline
4. PyTorch neural network (structured features)
5. Combined neural network (statistics + NLP)

Evaluation: accuracy, precision, recall, macro F1, confusion matrices, player usage trends, and training
curves, using a **chronological** split (no future-data leakage).

## Setup

Requires Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` uses `polars[rtcompat]` (a build without AVX requirements so it also runs under Rosetta).
On a native machine plain `polars` works too and is slightly faster.

## Commands

```bash
python main.py --check-data                      # 2024 smoke test + top-targets chart
python main.py --prepare-data                    # 2022-2025 preparation pipeline
python main.py --prepare-data --refresh-data     # same, ignoring cached Parquet (re-download)
pytest                                           # offline tests (no internet needed)
```

### What `--prepare-data` does

1. **Load** weekly player stats (`nflreadpy.load_player_stats([season], summary_level="week")`) and schedules
   (`nflreadpy.load_schedules([season])`) for 2022, 2023, 2024, 2025. Each season is cached in
   `data/raw/player_stats_week_<season>.parquet` and `data/raw/schedules_<season>.parquet`; only missing
   seasons are downloaded. Use `--refresh-data` to force a re-download.
2. **Clean** (`src/preprocessing.py`): regular season only, RB/WR/TE only, validate values, remove duplicate
   player-game records, sort chronologically. `opportunities = carries + targets`.
   * *Required* values (`player_id`, `team`, `season`, `week`, `game_id`, `carries`, `targets`): a row without
     valid values is excluded and counted.
   * *Optional* values (`receptions`, yards, `target_share`, `fantasy_points_ppr`, name): missing or impossible
     values become null and the row is **kept**. Nothing is imputed with 0.
3. **Schedule-aware next game**: each team's next scheduled regular-season game comes from the schedule, so bye
   weeks are skipped and a player's next *appearance* is never assumed to be the next *team game*. The last game
   of a season has no next game. Players are never carried across teams: the next-game lookup requires a record
   for the same player on the same team.
4. **Features** (`src/feature_engineering.py`): current-game `carries`, `targets`, `opportunities`, `target_share`;
   trailing 3-game averages (current game included) of opportunities, carries, targets, receptions, rushing
   yards, receiving yards, and PPR fantasy points. Windows are computed per player and season (seasons are
   independent) and are **strict**: the three records must be the same team's three consecutive scheduled games
   (byes do not break a window; a missed team game or a team change does). Rows without a valid window are
   not model-ready and are counted. Averages of optional stats use the non-null values in the window.
5. **Labels** (see below), only when the next-game outcome is known.
6. **Splits**: Training 2022-2023, Validation 2024, Test 2025, labeled model-ready rows only.
   No scalers or other learned transforms are fitted in this phase. When added later they must be fit on the
   training split only.
7. **Verify** no future information is in the features (see below), save outputs, make graphs, print a summary.

### Experimental role-change labels

These are **experimental labels, not official NFL role classifications.**

```
opportunities = carries + targets
baseline      = average opportunities over the last 3 games (current game included)
change        = next_game_opportunities - baseline
threshold     = max(3.0, baseline * 0.30)

change >=  threshold  -> Increasing
change <= -threshold  -> Decreasing
otherwise             -> Stable
```

`next_game_opportunities` is the player's opportunities in his team's next *scheduled* regular-season game.
If that game has no valid stats record for the player (inactive, did not play, traded away, missing data) or the
season has ended, the label is left **missing**; it is never assumed to be Decreasing. A recorded game with 0
carries and 0 targets is a valid 0 and can be Decreasing. The comparison is done in exact integer arithmetic, so
values exactly on the threshold are classified deterministically.

Limitations:

- Usage changes depend on game script, opponent, injuries and team pace, not only on a "role".
- The `3.0` floor dominates for low-volume players (a TE averaging 3 opportunities needs 3 more to be Increasing),
  while the 30% rule dominates for high-volume RBs.
- The baseline includes the current game; a single outlier game shifts it.
- Strict windows and the next-game requirement exclude traded players and players with injury gaps, so the
  dataset over-represents healthy regulars. Rookies and mid-season arrivals need 3 games first.
- Receivers' opportunities count targets only, not routes or snaps. No snap counts or play-by-play context yet.
- Class balance is skewed toward Stable (roughly two thirds of labels).

### Future-information safety

- `FEATURE_COLUMNS` are the only columns allowed as model inputs. Use `get_model_inputs(df)` to select them.
- Future-derived columns are kept separately as **audit columns** and must never be model inputs:
  `next_game_id`, `next_game_week`, `next_game_opportunities`, `role_change`, `role_threshold`,
  `label_missing_reason`. The label columns are `role_label` and `role_label_id`.
- `--prepare-data` runs a truncation test on the real data: for each season and several cutoff weeks it deletes all
  later games, recomputes the features, and requires identical values for the earlier weeks. It aborts if anything
  differs. Unit tests also confirm that altering future games never changes earlier features, that the detector
  catches a deliberately leaky feature, and that feature columns are disjoint from audit/label columns.

### Output files

```
data/processed/train.parquet                 2022-2023 labeled model-ready rows
data/processed/validation.parquet            2024
data/processed/test.parquet                  2025
data/processed/player_weeks_features.parquet every cleaned player-game with flags and missing-label reasons
data/processed/preparation_report.json       exclusion counts, split seasons, leakage check
results/graphs/observations_by_season.png
results/graphs/label_distribution.png
results/graphs/avg_opportunities_by_position.png
```

Column order in the split files: identifiers, then features, then labels, then audit columns. The identifiers
`player_id`, `game_id`, `season`, `week`, `team` are always present and never null so the data can be joined to
play-by-play descriptions in the NLP phase (`player_id` is the GSIS id; `game_id` matches the
nflverse play-by-play `game_id`).

## Layout

```
data/raw/            downloaded Parquet cache (git-ignored)
data/processed/      derived datasets (git-ignored)
src/
  data_loader.py         nflreadpy download + per-season Parquet cache
  preprocessing.py       cleaning, next-game lookup, labels, chronological splits
  feature_engineering.py rolling features, column groups, leakage checks
  models.py              (placeholder) classical baselines
  neural_network.py      (placeholder) PyTorch models
  evaluation.py          (placeholder) metrics
  visualization.py       charts
models/              saved models (git-ignored)
results/graphs/      generated charts
tests/               pytest tests (synthetic fixtures live in tests/synthetic.py, unit tests only)
main.py              command-line entry point
```

## Tests

`pytest` runs offline. Tests use tiny hand-written fixtures (fake seasons and team codes) to check logic only;
they are never used for graphs, datasets or results.

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
- NLP text features and the first three traditional baselines (`--train-baselines`): play-by-play text windows,
  TF-IDF, a majority-class reference, and statistics-only / text-only / combined logistic regressions, evaluated on
  2024 validation (see "Phase 3" below).
- Offline pytest suite (120 tests).

**Not implemented yet:** the PyTorch neural networks (models 4 and 5) and the locked 2025 test evaluation.
`neural_network.py` is a placeholder.

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
python main.py --train-baselines                 # play-by-play text + 4 baseline models (needs --prepare-data first)
python main.py --train-baselines --refresh-data  # same, re-downloading play-by-play text
python -m pytest -q                              # offline tests (no internet needed)
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
data/processed/text_{train,validation,test}.parquet   play-text windows (keys + text), written by --train-baselines
results/graphs/observations_by_season.png
results/graphs/label_distribution.png
results/graphs/avg_opportunities_by_position.png
```

Column order in the split files: identifiers, then features, then labels, then audit columns. The identifiers
`player_id`, `game_id`, `season`, `week`, `team` are always present and never null so the data can be joined to
play-by-play descriptions in the NLP phase (`player_id` is the GSIS id; `game_id` matches the
nflverse play-by-play `game_id`).

## Phase 3: NLP text features and first baseline models

Command: `python main.py --train-baselines` (run `--prepare-data` first). Everything below is the output of that
command on real 2022-2025 nflverse data; nothing is simulated. Full numbers are in `results/baseline_results.json`.

### Experimental design

| Item | Setting |
|---|---|
| Train | 2022-2023 (5,497 labeled player-weeks) |
| Validation | 2024 (2,796) |
| Test | 2025 (2,889) - **not used** for fitting, tuning, selection or scoring in this phase |
| Target | Increasing / Stable / Decreasing experimental role-change label (see above) |
| Primary metric | macro F1 (also accuracy, macro precision/recall, per-class P/R/F1, confusion matrices) |
| Primary comparison | does adding NLP features to statistics improve on statistics alone (combined vs statistics-only) |
| Seed | 42 |

The test split's text windows are built only so row counts and labels can be checked against the existing split; no
vectorizer is fit or applied to them and no model sees them.

### Text pipeline

1. **Source:** `nflreadpy.load_pbp([season])`, one season at a time. Only regular-season plays and 9 columns are
   kept and cached as `data/raw/pbp_text_<season>.parquet` (about 1 MB per season, instead of about 100 MB for the
   full table).
2. **Player attribution by official id (never by name):** a `run` play belongs to its `rusher_player_id`; a `pass`
   play belongs to its `receiver_player_id` (the target). These ids are the same GSIS ids as `player_id`. Penalty
   "No Play" rows, kneels and plays without an id are excluded. The description text is the play's `desc` field.
3. **Player-game documents:** the cleaned descriptions of one player in one game, joined on `(player_id, game_id)`
   and ordered by `play_id` (chronological).
4. **Strict three-game windows:** each model-ready row's text is the concatenation of the player's documents for the
   same three consecutive team games used by the statistical features: the previous two games and the current,
   completed game. The next game (the one being predicted) and all later games are never included. A row with no
   matching play text in its window keeps `text == ""`; no observation is dropped. This is verified on the real
   data on every run: replacing every document after week `k` with a marker leaves earlier windows byte-identical,
   and no window contains the next game id.
5. **Cleaning:** the leading game clock is removed; jersey numbers and player names are masked (offense and defense,
   with or without a team prefix, including multi-word names such as `Vander Esch` / `St. Brown`, suffixes like `Jr.`,
   `#6 Driskel`, and names in "Injury Update" sentences); text is lowercased. Football language is preserved
   ("up the middle", "left tackle", "pushed ob", "short right", "shotgun", "touchdown", "fumbles"), as are team
   abbreviations and yard lines (`NYJ 25`).
6. **TF-IDF** (scikit-learn, fixed settings, not tuned): unigrams and bigrams, `min_df=5`, `max_features=5000`,
   `sublinear_tf=True`, alphabetic tokens only, no stop-word removal (words like "up" and "out" matter). The
   vectorizer is fit on **training rows only** (inside the pipeline), then used to transform validation. This gave
   a vocabulary of 1,332 terms.

### Verification of player-name masking (real data)

Every one of the 125,861 run/pass plays with an official id is audited after masking on each `--train-baselines` run:

| Check | Plays (of 125,861) |
|---|---|
| leftover jersey token (e.g. `21-X`) | 0 |
| leftover `Initial.Surname` (e.g. `S.Brown`) | 0 |
| leftover `#10 Surname` | 0 |
| the involved player's own surname still in the text | 4 (0.003%) - 1 is a false positive ("downs" vs. Josh Downs), 3 are real (see below) |
| any surname from the league player table (team/city words excluded) | 47 (0.037%) |

A first version of the masking was not good enough, and the audit found real issues that were fixed:
a 2-letter first-name pattern missed names like `Bre.Hall` and `Dari.Williams` (the involved player's own surname survived in about 0.9% of plays in a 60,000-play sample);
`A.St. Brown` and `L.Vander Esch` were only partly masked (the alternation tried `Van` before `Vander`);
bare `T.Hill`, `34-Edmunds`, `#6 Driskel`, and full names in "Injury Update: Jayron Kearse has..." sentences were not
covered. What remains is mostly the ordinary words "Ball" (as in "Loose Ball") and "Rush", which are false positives
of the audit, plus rare back-references such as "Jackson sets record". The 3 real own-surname survivors are one
narrative back-reference ("Jackson sets record..."), one `L. Tunsil 78 Eligible Receiver` (initial + space, no jersey
number) and one replay-review sentence ("TB 17 Gage"); they are left unmasked because widening the patterns to catch
them would start deleting ordinary football words. The audit's own-surname count was previously 1 because the
id-to-name map was built from two separate non-order-stable Polars `unique()` calls that could mispair ids and names;
it is now built from one deterministic call (the masking itself and all model results are unaffected).
Double-initial names such as `J.J.Watt` / `A.J.Brown` are also masked, although nflfastR never writes them that way
(0 occurrences in the 2022-2025 data). Team abbreviations and city/team names are deliberately kept, so team
identity can still be learned from the text (see limitations); the one exception is a team tag that is part of a
player token itself (`PHI-97-J.Hargrave`), which is removed together with the token.

### Text coverage (real data)

- REG run/pass plays with an official player id, 2022-2025: 125,861. Of these, 117,683 belong to cleaned RB/WR/TE
  player-games; 8,178 do not (QB rushes and other positions).
- 18,464 of 20,886 cleaned RB/WR/TE player-games have at least one matched play; 2,422 have none.
- The ids reconcile with the existing stats: the number of matched run plays equals `carries` in 99.55% of
  player-games, and matched pass targets equal `targets` in 98.46%.
- Windows exist for all 13,457 model-ready rows (same count as `--prepare-data`). Train / validation / test row
  counts and labels are unchanged after attaching text (checked).

| Split | Rows | Empty documents | Non-empty | Mean plays per window | Median words per window |
|---|---|---|---|---|---|
| Train 2022-2023 | 5,497 | 124 (2.26%) | 97.74% | 21.0 | 172 |
| Validation 2024 | 2,796 | 94 (3.36%) | 96.64% | 20.5 | 170 |
| Test 2025 | 2,889 | 118 (4.08%) | 95.92% | 20.1 | 166 |

### Models and tuning

All four are scikit-learn pipelines. Imputation (median), `StandardScaler` and TF-IDF are fit on the training rows only.
TF-IDF features stay sparse (`sparse_threshold=1.0`, no dense conversion).

0. `DummyClassifier(strategy="most_frequent")` - majority-class reference (always predicts Stable).
1. Statistics-only: the 11 existing statistical features (`carries, targets, opportunities, target_share` and the
   seven `avg3_*` features) -> `LogisticRegression`.
2. Text-only: TF-IDF of the 3-game play-text window -> `LogisticRegression`.
3. Combined: the 11 statistical features (imputed, scaled) plus TF-IDF -> one `LogisticRegression`.

`LogisticRegression(solver="lbfgs", max_iter=2000, random_state=42)`. For models 1-3 the grid was
`C in {0.01, 0.1, 1, 10}` by `class_weight in {None, "balanced"}` (8 fits per model), fit on 2022-2023 and
scored by **validation macro F1 only**; ties would go to smaller `C`, then `None`. Models stay fit on 2022-2023 (no
refit on train + validation). No player ids, labels, audit columns, next-game statistics or next-game text are model
inputs; this is enforced by `to_model_frame` and tested.

Selected settings:

| Model | C | class_weight |
|---|---|---|
| Majority-class dummy | n/a | n/a |
| Statistics-only LR | 10 | balanced |
| Text-only LR | 0.01 | balanced |
| Combined LR | 0.01 | balanced |

### Results (validation, 2024; train 2022-2023)

| Model | Macro F1 | Accuracy | Macro precision | Macro recall |
|---|---|---|---|---|
| Majority-class dummy | 0.2684 | 0.6738 | 0.2246 | 0.3333 |
| Statistics-only LR | 0.4234 | 0.5182 | 0.4275 | 0.4597 |
| Text-only LR (TF-IDF) | 0.4223 | 0.5812 | 0.4327 | 0.4381 |
| Combined LR (stats + TF-IDF) | **0.4267** | 0.5207 | 0.4336 | 0.4635 |

Per-class precision / recall / F1 (support: Decreasing 438, Stable 1,884, Increasing 474):

| Model | Decreasing P / R / F1 | Stable P / R / F1 | Increasing P / R / F1 |
|---|---|---|---|
| Majority-class dummy | 0.000 / 0.000 / 0.000 | 0.674 / 1.000 / 0.805 | 0.000 / 0.000 / 0.000 |
| Statistics-only LR | 0.273 / 0.548 / 0.365 | 0.775 / 0.578 / 0.662 | 0.234 / 0.253 / 0.243 |
| Text-only LR | 0.271 / 0.438 / 0.335 | 0.743 / 0.722 / 0.732 | 0.284 / 0.154 / 0.200 |
| Combined LR | 0.265 / 0.546 / 0.357 | 0.792 / 0.579 / 0.669 | 0.244 / 0.266 / 0.254 |

Confusion matrices (rows true, columns predicted: Decreasing, Stable, Increasing):

| Model | Decreasing | Stable | Increasing |
|---|---|---|---|
| Statistics-only | 240, 100, 98 | 501, 1089, 294 | 138, 216, 120 |
| Text-only | 192, 203, 43 | 383, 1360, 141 | 133, 268, 73 |
| Combined | 239, 86, 113 | 515, 1091, 278 | 147, 201, 126 |

Paired bootstrap on the 2024 rows (1,000 resamples) of the macro-F1 difference:

| Comparison | Difference | 95% interval | First model better in |
|---|---|---|---|
| Combined - statistics-only (primary) | +0.0034 | [-0.0086, +0.0155] | 72% of resamples |
| Text-only - statistics-only | -0.0010 | [-0.0263, +0.0205] | 44% |
| Combined - text-only | +0.0044 | [-0.0159, +0.0263] | 68% |

### Interpretation

- All three learned models clearly beat the majority-class reference on macro F1 (0.42-0.43 vs 0.27). The
  dummy has the highest accuracy (0.674) only because Stable is 67% of 2024; it never predicts Decreasing or
  Increasing. Accuracy is therefore a poor yardstick here, which is why macro F1 is the primary metric.
- The class-balanced setting was selected for every model, and it matters: without it macro F1 is about 0.29 for
  statistics-only, because the classifiers mostly predict Stable. This costs accuracy (0.52 vs 0.67) but recovers
  the minority classes. For statistics-only, validation macro F1 is flat across `C` (0.419-0.423 with balanced
  weights), so the selection of the edge value `C=10` is not meaningful.
- **Adding NLP did not measurably improve on statistics alone.** The combined model is +0.0034 macro F1 higher, but
  the bootstrap interval includes 0 and the difference is smaller than the noise of this validation set. The
  text-only model reaches about the same macro F1 as statistics-only (-0.0010); it predicts Stable more
  often (higher Stable recall and accuracy, lower Increasing recall). Increasing is the hardest class for every
  model (F1 0.20-0.25).
- The strongest terms in the text-only model are mostly play-type words ("pass", "short", "incomplete" for Stable;
  "guard", "tackle" for Increasing/Decreasing) and, for Increasing, the team abbreviation "chi", which is a sign that
  team identity in the text carries some signal.

### Important caveats

- **Overlap between text and statistics.** The play descriptions come from the same plays used to compute the
  statistical features (`carries`, `targets`, `opportunities` and their 3-game averages). Text and statistics
  therefore share information: the text of a run play is the same event that increments `carries`. Because of this
  overlap, the text-only baseline should not be read as an independent signal, and the meaningful question is the
  primary comparison above: whether adding NLP improves over statistics alone. In this phase it does not do so
  measurably.
- **Numeric yardage is not represented.** The TF-IDF token pattern is alphabetic (`[a-z][a-z']+`), so numbers such
  as "for 12 yards", yard lines and scores are dropped; only words like "yards", "touchdown", "no gain" remain.
  Distance and field-position magnitude are therefore invisible to the text models. This is a deliberate first
  setting, not a final design; numeric features could be added later.
- **Validation is used for both tuning and reporting.** Eight configurations per model were scored on 2024 and the best
  was kept, so the validation numbers are slightly optimistic. The 2025 test split is reserved for a single final
  evaluation of locked settings and has not been looked at.
- **Fixed TF-IDF settings and a shared `C`.** The vectorizer was not tuned, and in the combined model the scaled statistics and
  the L2-normalized TF-IDF block share one regularization strength.
- **Team abbreviations remain in the text** (a deliberate choice), so team identity can leak into the text models.
- Documents contain only the player's own run and target plays: no routes, snaps, quarterback, defense or game
  context, and 2,422 player-games (nearly all of them 0-opportunity games) have no text at all (2.3% to 4.1% of windows
  are fully empty).
- The labels are experimental, class-imbalanced (about two thirds Stable), and noisy; results are preliminary.

### Generated graphs (validation only)

```
results/graphs/baseline_validation_macro_f1.png
results/graphs/baseline_validation_accuracy.png
results/graphs/baseline_validation_confusion_matrices.png
results/graphs/baseline_validation_predicted_label_distribution.png
```

## Layout

```
data/raw/            downloaded Parquet cache (git-ignored)
data/processed/      derived datasets (git-ignored)
src/
  data_loader.py         nflreadpy download + per-season Parquet cache
  preprocessing.py       cleaning, next-game lookup, labels, chronological splits
  feature_engineering.py rolling features, column groups, leakage checks
  text_features.py       play-by-play text: cleaning, name masking, player-game documents, 3-game windows, TF-IDF
  models.py              dummy / stats / text / combined logistic-regression pipelines, validation tuning
  neural_network.py      (placeholder) PyTorch models
  evaluation.py          accuracy, macro P/R/F1, per-class metrics, confusion matrices, paired bootstrap
  visualization.py       charts
models/              saved models, baseline_*.joblib (git-ignored)
results/graphs/      generated charts
results/baseline_results.json   settings, tuning grids, metrics, coverage, audits of the last --train-baselines run
tests/               pytest tests (synthetic fixtures live in tests/synthetic.py, unit tests only)
main.py              command-line entry point
```

## Tests

`pytest` runs offline. Tests use tiny hand-written fixtures (fake seasons and team codes) to check logic only;
they are never used for graphs, datasets or results.

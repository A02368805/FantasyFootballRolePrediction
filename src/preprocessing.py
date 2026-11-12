"""Cleaning, schedule-aware next-game lookup, role-label creation, and
chronological train/validation/test splitting.

ROLE-CHANGE LABELS (experimental, NOT official NFL role classifications)
-----------------------------------------------------------------------
    opportunities = carries + targets
    baseline      = mean opportunities over the player's last 3 games
                    (current game included; see feature_engineering)
    change        = next_game_opportunities - baseline
    threshold     = max(3.0, 0.30 * baseline)

    change >=  threshold -> "Increasing"
    change <= -threshold -> "Decreasing"
    otherwise            -> "Stable"

``next_game_opportunities`` is the player's opportunities in his team's next
SCHEDULED regular-season game (bye weeks skipped, season boundary = no next
game). If that game has no valid stats record for the player (inactive, DNP,
traded away, missing data), the label is left MISSING - it is never assumed to
be "Decreasing". A recorded game with 0 carries and 0 targets is a valid 0.

Limitations: labels depend on game script, injuries, opponent and team volume,
not just role; the 3.0 floor dominates for low-volume players; the baseline
includes the current game; strict history windows and traded players reduce
sample size; there are no snap counts or play-by-play context yet.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from src.feature_engineering import (
    AUDIT_COLUMNS,
    FEATURE_COLUMNS,
    HISTORY_COLUMNS,
    LABEL_COLUMNS,
    METADATA_COLUMNS,
    add_rolling_features,
    assert_feature_columns_safe,
    verify_no_future_information,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

POSITIONS = ["RB", "WR", "TE"]
LABEL_ORDER = ["Decreasing", "Stable", "Increasing"]
LABEL_IDS = {"Decreasing": 0, "Stable": 1, "Increasing": 2}

TRAIN_SEASONS = (2022, 2023)
VALIDATION_SEASONS = (2024,)
TEST_SEASONS = (2025,)

# Required: needed to identify the player-game, place it on the schedule and
# compute opportunities. A row without valid values here is excluded.
REQUIRED_COLUMNS = ["player_id", "position", "team", "season", "week", "game_id", "carries", "targets"]
# Optional: descriptive/secondary stats. Missing values never exclude a row.
OPTIONAL_TEXT_COLUMNS = ["player_display_name"]
OPTIONAL_NUMERIC_COLUMNS = [
    "receptions",
    "rushing_yards",
    "receiving_yards",
    "target_share",
    "fantasy_points_ppr",
]


@dataclass
class PreparationReport:
    """Counts of what was excluded or left missing at each stage."""

    counts: dict[str, int] = field(default_factory=dict)

    def set(self, key: str, value: int) -> None:
        self.counts[key] = int(value)

    def add(self, key: str, value: int) -> None:
        self.counts[key] = self.counts.get(key, 0) + int(value)

    def to_dict(self) -> dict[str, int]:
        return dict(self.counts)


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------


def clean_player_games(stats: pl.DataFrame, report: PreparationReport | None = None) -> pl.DataFrame:
    """Regular-season RB/WR/TE player-games with validated values.

    * Keeps one row per ``(player_id, game_id)`` (duplicates removed, counted).
    * Required values (ids, team, season, week, carries, targets) must be valid;
      otherwise the row is excluded and counted.
    * Optional values (receptions, yards, target_share, fantasy points, name) that
      are missing or invalid become null and the row is KEPT.
    * Sorted chronologically (season, week, player_id). Adds ``opportunities``.
    """
    report = report if report is not None else PreparationReport()
    report.set("raw_rows", stats.height)

    df = stats.filter(pl.col("season_type") == "REG")
    report.set("removed_non_regular_season", stats.height - df.height)

    before = df.height
    df = df.filter(pl.col("position").is_in(POSITIONS))
    report.set("removed_non_rb_wr_te", before - df.height)

    # Keep only useful columns; optional ones missing from the schema become null.
    for col in OPTIONAL_TEXT_COLUMNS:
        if col not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=pl.String).alias(col))
    for col in OPTIONAL_NUMERIC_COLUMNS:
        if col not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias(col))
    df = df.select(REQUIRED_COLUMNS + OPTIONAL_TEXT_COLUMNS + OPTIONAL_NUMERIC_COLUMNS)

    df = df.with_columns(
        [pl.col(c).cast(pl.Int64, strict=False) for c in ("season", "week", "carries", "targets")]
        + [pl.col(c).cast(pl.Float64, strict=False) for c in OPTIONAL_NUMERIC_COLUMNS]
    )

    # Required-value validation.
    valid_required = (
        pl.col("player_id").is_not_null()
        & (pl.col("player_id").cast(pl.String).str.strip_chars() != "")
        & pl.col("team").is_not_null()
        & pl.col("game_id").is_not_null()
        & pl.col("season").is_not_null()
        & pl.col("week").is_not_null()
        & (pl.col("week") >= 1)
        & pl.col("carries").is_not_null()
        & (pl.col("carries") >= 0)
        & pl.col("targets").is_not_null()
        & (pl.col("targets") >= 0)
    )
    before = df.height
    df = df.filter(valid_required)
    report.set("removed_invalid_required_values", before - df.height)

    # Optional-value sanitising: impossible values become null (row is kept).
    # Negative yards / fantasy points are legitimate, so they are left alone.
    invalid_receptions = (pl.col("receptions") < 0).fill_null(False)
    invalid_share = ((pl.col("target_share") < 0) | (pl.col("target_share") > 1)).fill_null(False)
    report.set(
        "optional_values_set_to_null_invalid",
        df.select(invalid_receptions.sum() + invalid_share.sum()).item(),
    )
    df = df.with_columns(
        pl.when(invalid_receptions).then(None).otherwise(pl.col("receptions")).alias("receptions"),
        pl.when(invalid_share).then(None).otherwise(pl.col("target_share")).alias("target_share"),
    )
    any_optional_missing = pl.any_horizontal(
        [pl.col(c).is_null() for c in OPTIONAL_NUMERIC_COLUMNS + OPTIONAL_TEXT_COLUMNS]
    )
    report.set("kept_rows_with_missing_optional_values", df.select(any_optional_missing.sum()).item())

    # Duplicate player-game (and player-week) records.
    df = df.sort(["player_id", "game_id", "season", "week"])
    before = df.height
    df = df.unique(subset=["player_id", "game_id"], keep="first", maintain_order=True)
    report.set("removed_duplicate_player_games", before - df.height)
    before = df.height
    df = df.unique(subset=["player_id", "season", "week"], keep="first", maintain_order=True)
    report.set("removed_duplicate_player_weeks", before - df.height)

    df = df.with_columns((pl.col("carries") + pl.col("targets")).alias("opportunities"))
    df = df.sort(["season", "week", "player_id"])
    report.set("cleaned_rows", df.height)
    return df


# ---------------------------------------------------------------------------
# Schedule-aware next game
# ---------------------------------------------------------------------------


def build_team_schedule(schedules: pl.DataFrame) -> pl.DataFrame:
    """One row per regular-season team-game with its position in the team's season
    (``team_game_number``) and the team's NEXT scheduled game in the same season.

    Bye weeks are skipped naturally (the next row is the next scheduled game);
    the final game of a season has a null next game (season boundary).
    """
    reg = schedules.filter(pl.col("game_type") == "REG").with_columns(
        pl.col("season").cast(pl.Int64), pl.col("week").cast(pl.Int64)
    )
    long = pl.concat(
        [
            reg.select("game_id", "season", "week", pl.col("home_team").alias("team")),
            reg.select("game_id", "season", "week", pl.col("away_team").alias("team")),
        ]
    ).drop_nulls(["team"])
    return (
        long.sort(["season", "team", "week"])
        .with_columns(
            (pl.int_range(pl.len()).over(["season", "team"]) + 1).alias("team_game_number"),
            pl.col("game_id").shift(-1).over(["season", "team"]).alias("next_game_id"),
            pl.col("week").shift(-1).over(["season", "team"]).alias("next_game_week"),
        )
        .select("season", "team", "game_id", "team_game_number", "next_game_id", "next_game_week")
    )


def attach_team_schedule(
    games: pl.DataFrame, schedules: pl.DataFrame, report: PreparationReport | None = None
) -> pl.DataFrame:
    """Add ``team_game_number``, ``next_game_id`` and ``next_game_week`` from the schedule.

    Rows whose ``(season, team, game_id)`` is not on the regular-season schedule
    cannot be placed and are excluded (counted).
    """
    report = report if report is not None else PreparationReport()
    sched = build_team_schedule(schedules)
    joined = games.join(sched, on=["season", "team", "game_id"], how="left")
    unplaced = joined.filter(pl.col("team_game_number").is_null()).height
    report.set("removed_not_on_schedule", unplaced)
    return joined.filter(pl.col("team_game_number").is_not_null()).sort(["season", "week", "player_id"])


def attach_next_game_outcome(df: pl.DataFrame) -> pl.DataFrame:
    """Look up the player's opportunities in his team's next scheduled game.

    The lookup requires a valid record for the same player, the same
    ``next_game_id`` and the same team. If none exists the outcome stays null (it
    is NOT assumed to be 0): this covers injuries/DNPs, traded players, missing
    data and unplayed games. Adds ``next_game_opportunities`` (audit column).
    """
    lookup = df.select(
        "player_id",
        "team",
        pl.col("game_id").alias("next_game_id"),
        pl.col("opportunities").alias("next_game_opportunities"),
    )
    out = df.join(lookup, on=["player_id", "team", "next_game_id"], how="left")
    return out.sort(["season", "week", "player_id"])


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


def assign_role_labels(df: pl.DataFrame) -> pl.DataFrame:
    """Add ``role_label``, ``role_label_id`` and the audit columns ``role_change``,
    ``role_threshold`` and ``label_missing_reason``.

    Requires ``avg3_opportunities`` and ``next_game_opportunities``. The decision
    is made in exact integer arithmetic (everything multiplied by 30) so values
    exactly on the threshold are classified deterministically, free of float error:

        S = 3 * baseline (integer sum of the 3 games), N = next opportunities
        30 * change    = 30*N - 10*S
        30 * threshold = max(90, 3*S)
    """
    s = (pl.col("avg3_opportunities") * 3).round(0).cast(pl.Int64)
    n = pl.col("next_game_opportunities").cast(pl.Int64)
    delta30 = 30 * n - 10 * s
    thr30 = pl.max_horizontal(pl.lit(90, dtype=pl.Int64), 3 * s)
    known = s.is_not_null() & n.is_not_null()

    label = (
        pl.when(~known)
        .then(None)
        .when(delta30 >= thr30)
        .then(pl.lit("Increasing"))
        .when(delta30 <= -thr30)
        .then(pl.lit("Decreasing"))
        .otherwise(pl.lit("Stable"))
    )
    reason = (
        pl.when(~pl.col("is_model_ready"))
        .then(pl.lit("no_three_game_history"))
        .when(pl.col("next_game_id").is_null())
        .then(pl.lit("season_end"))
        .when(pl.col("next_game_opportunities").is_null())
        .then(pl.lit("no_next_game_stats"))
        .otherwise(None)
    )
    return df.with_columns(
        label.alias("role_label"),
        (pl.col("next_game_opportunities") - pl.col("avg3_opportunities")).alias("role_change"),
        pl.when(pl.col("avg3_opportunities").is_null())
        .then(None)
        .otherwise(pl.max_horizontal(pl.lit(3.0), 0.30 * pl.col("avg3_opportunities")))
        .alias("role_threshold"),
        reason.alias("label_missing_reason"),
    ).with_columns(
        pl.col("role_label").replace_strict(LABEL_IDS, default=None, return_dtype=pl.Int64).alias("role_label_id")
    )


# ---------------------------------------------------------------------------
# Chronological splits
# ---------------------------------------------------------------------------

OUTPUT_COLUMNS = METADATA_COLUMNS + FEATURE_COLUMNS + LABEL_COLUMNS + AUDIT_COLUMNS


def chronological_split(
    df: pl.DataFrame,
    train_seasons=TRAIN_SEASONS,
    validation_seasons=VALIDATION_SEASONS,
    test_seasons=TEST_SEASONS,
) -> dict[str, pl.DataFrame]:
    """Split labeled, model-ready rows by season: train < validation < test.

    Only rows with a role label are included. Asserts the seasons are disjoint and
    strictly ordered in time, so no training row is later than a validation row
    and no validation row is later than a test row. Labels never cross a season
    boundary, so no label uses a game from a later split.
    """
    train_s, val_s, test_s = set(train_seasons), set(validation_seasons), set(test_seasons)
    if (train_s & val_s) or (train_s & test_s) or (val_s & test_s):
        raise ValueError("Split seasons must be disjoint.")
    if not (max(train_s) < min(val_s) and max(val_s) < min(test_s)):
        raise ValueError("Splits must be chronological: train < validation < test.")

    usable = df.filter(pl.col("is_model_ready") & pl.col("role_label").is_not_null())
    splits = {
        "train": usable.filter(pl.col("season").is_in(list(train_s))),
        "validation": usable.filter(pl.col("season").is_in(list(val_s))),
        "test": usable.filter(pl.col("season").is_in(list(test_s))),
    }
    # Defensive chronological check on the actual data.
    bounds = {k: (v["season"].min(), v["season"].max()) for k, v in splits.items() if v.height}
    ordered = [bounds[k] for k in ("train", "validation", "test") if k in bounds]
    for earlier, later in zip(ordered, ordered[1:]):
        if earlier[1] >= later[0]:
            raise AssertionError("Chronological split violated.")
    return splits


def label_distribution_table(splits: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Counts and percentages of each label per split (percent of that split)."""
    rows = []
    for split, df in splits.items():
        total = df.height
        counts = {r["role_label"]: r["len"] for r in df.group_by("role_label").len().to_dicts()}
        for label in LABEL_ORDER:
            c = counts.get(label, 0)
            rows.append(
                {
                    "split": split,
                    "label": label,
                    "count": c,
                    "percent": (100.0 * c / total) if total else 0.0,
                }
            )
    return pl.DataFrame(rows, schema={"split": pl.String, "label": pl.String, "count": pl.Int64, "percent": pl.Float64})


def season_label_counts(splits: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Sample counts by season and label across all splits."""
    frames = [df.select("season", "role_label") for df in splits.values() if df.height]
    if not frames:
        return pl.DataFrame(schema={"season": pl.Int64, "role_label": pl.String, "len": pl.UInt32})
    return pl.concat(frames).group_by("season", "role_label").len().sort("season", "role_label")


# ---------------------------------------------------------------------------
# End-to-end preparation
# ---------------------------------------------------------------------------


def prepare_dataset(
    stats: pl.DataFrame,
    schedules: pl.DataFrame,
    verify_leakage: bool = True,
) -> tuple[pl.DataFrame, PreparationReport, dict | None]:
    """Run clean -> schedule -> features -> next-game outcome -> labels.

    Returns ``(full_table, report, leakage_summary)``. ``full_table`` has one row per
    cleaned player-game, including rows that are not model-ready or unlabeled
    (flagged by ``is_model_ready`` / ``label_missing_reason``).
    """
    assert_feature_columns_safe()
    report = PreparationReport()
    games = clean_player_games(stats, report)
    games = attach_team_schedule(games, schedules, report)
    report.set("placed_on_schedule_rows", games.height)

    # Features use only the current and previous games (no next-game columns).
    featured = add_rolling_features(games)
    leakage = verify_no_future_information(games) if verify_leakage else None

    report.set("excluded_insufficient_games_history", featured.filter(pl.col("history_status") == "insufficient_games").height)
    report.set("excluded_gap_or_team_change_history", featured.filter(pl.col("history_status") == "gap_or_team_change").height)
    report.set("model_ready_rows", featured.filter(pl.col("is_model_ready")).height)

    full = assign_role_labels(attach_next_game_outcome(featured))
    ready = full.filter(pl.col("is_model_ready"))
    report.set("model_ready_unlabeled_season_end", ready.filter(pl.col("label_missing_reason") == "season_end").height)
    report.set("model_ready_unlabeled_no_next_game_stats", ready.filter(pl.col("label_missing_reason") == "no_next_game_stats").height)
    report.set("labeled_rows", full.filter(pl.col("role_label").is_not_null()).height)
    return full, report, leakage


def save_processed(
    full: pl.DataFrame,
    splits: dict[str, pl.DataFrame],
    report: PreparationReport,
    out_dir: Path = PROCESSED_DATA_DIR,
    leakage: dict | None = None,
) -> dict[str, Path]:
    """Write splits and the full table to Parquet, plus a JSON preparation report.

    Split files contain identifiers (``player_id, game_id, season, week, team`` ...) for
    the later play-by-play join, then ``FEATURE_COLUMNS``, the label, and clearly
    separate audit columns. No scalers or other learned transforms are fitted here.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, df in splits.items():
        path = out_dir / f"{name}.parquet"
        df.select(OUTPUT_COLUMNS).write_parquet(path)
        paths[name] = path
    full_path = out_dir / "player_weeks_features.parquet"
    full.select(OUTPUT_COLUMNS + HISTORY_COLUMNS).write_parquet(full_path)
    paths["player_weeks_features"] = full_path

    report_path = out_dir / "preparation_report.json"
    report_path.write_text(
        json.dumps(
            {
                "counts": report.to_dict(),
                "split_seasons": {
                    "train": list(TRAIN_SEASONS),
                    "validation": list(VALIDATION_SEASONS),
                    "test": list(TEST_SEASONS),
                },
                "feature_columns": FEATURE_COLUMNS,
                "audit_columns_never_model_inputs": AUDIT_COLUMNS,
                "leakage_check": leakage,
            },
            indent=2,
            default=str,
        )
    )
    paths["preparation_report"] = report_path
    return paths

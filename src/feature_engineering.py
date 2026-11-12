"""Statistical (rolling usage) features.

Everything here uses only information available through the END of the current
game: the current game's own stats plus the previous two games. Features never
look at later games. TF-IDF / text features are a later phase.

Column groups (kept strictly separate):

* ``METADATA_COLUMNS``  identifiers (incl. the keys needed to join play-by-play
  text later: ``player_id``, ``game_id``, ``season``, ``week``, ``team``).
* ``FEATURE_COLUMNS``   the ONLY columns that may be used as model inputs.
* ``LABEL_COLUMNS``     the experimental role-change label (target).
* ``AUDIT_COLUMNS``     future-derived columns (next game, change, threshold).
  They must NEVER be used as model inputs.
"""

from __future__ import annotations

import polars as pl

GROUP_KEYS = ["player_id", "season"]
WINDOW = 3

METADATA_COLUMNS = [
    "player_id",
    "player_display_name",
    "position",
    "team",
    "season",
    "week",
    "game_id",
    "team_game_number",
]

# Current-game stats (known by the end of the game).
CURRENT_GAME_FEATURES = ["carries", "targets", "opportunities", "target_share"]

# Trailing 3-game averages (current game + previous two) of these stats.
ROLLING_SOURCES = {
    "avg3_opportunities": "opportunities",
    "avg3_carries": "carries",
    "avg3_targets": "targets",
    "avg3_receptions": "receptions",
    "avg3_rushing_yards": "rushing_yards",
    "avg3_receiving_yards": "receiving_yards",
    "avg3_fantasy_points_ppr": "fantasy_points_ppr",
}

FEATURE_COLUMNS = CURRENT_GAME_FEATURES + list(ROLLING_SOURCES)

LABEL_COLUMNS = ["role_label", "role_label_id"]

AUDIT_COLUMNS = [
    "next_game_id",
    "next_game_week",
    "next_game_opportunities",
    "role_change",
    "role_threshold",
    "label_missing_reason",
]

HISTORY_COLUMNS = ["history_status", "is_model_ready"]


def assert_feature_columns_safe() -> None:
    """Static guard: model inputs must not overlap audit/label/metadata columns."""
    forbidden = set(AUDIT_COLUMNS) | set(LABEL_COLUMNS) | set(METADATA_COLUMNS)
    overlap = forbidden & set(FEATURE_COLUMNS)
    if overlap:
        raise AssertionError(f"Feature columns overlap forbidden columns: {sorted(overlap)}")
    bad_names = [c for c in FEATURE_COLUMNS if c.startswith("next_") or "label" in c or "role_" in c]
    if bad_names:
        raise AssertionError(f"Feature columns look future/label derived: {bad_names}")


def get_model_inputs(df: pl.DataFrame) -> pl.DataFrame:
    """Return ONLY the predictive feature columns (never audit/label/metadata)."""
    assert_feature_columns_safe()
    return df.select(FEATURE_COLUMNS)


def add_rolling_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add current-game and trailing 3-game features per player-week.

    Required input columns: ``player_id, season, week, team, team_game_number,
    carries, targets, opportunities``. Optional stats (receptions, yards, fantasy
    points, target_share) may be missing/null; they are never imputed with 0.

    Rules:
      * Rolling windows are computed per ``(player_id, season)`` (seasons independent)
        and are trailing only: current game plus the previous two player records.
      * STRICT window: the three records must be the same team's three consecutive
        scheduled games (``team_game_number`` +1 each, same team). Bye weeks are
        skipped by the schedule, so they do not break a window; a missed team game
        or a team change does.
      * Rows without a valid window get null ``avg3_*`` features and
        ``is_model_ready = False``. ``history_status`` records why:
        ``ok`` | ``insufficient_games`` | ``gap_or_team_change``.
      * Optional-stat averages use the non-null values in the window, so a missing
        optional value never discards the player-week; it is null only if all
        three values are missing.
    """
    required = ["player_id", "season", "week", "team", "team_game_number", "carries", "targets", "opportunities"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"add_rolling_features missing columns {missing}")

    work = df.sort(["player_id", "season", "week"])
    # Optional stat columns absent from the dataset become all-null (not zero).
    for col in set(ROLLING_SOURCES.values()) | {"target_share"}:
        if col not in work.columns:
            work = work.with_columns(pl.lit(None, dtype=pl.Float64).alias(col))

    grp = GROUP_KEYS
    idx = pl.int_range(pl.len()).over(grp)
    tgn = pl.col("team_game_number")
    team = pl.col("team")
    consecutive = (
        ((tgn - tgn.shift(1).over(grp)) == 1)
        & ((tgn.shift(1).over(grp) - tgn.shift(2).over(grp)) == 1)
        & (team == team.shift(1).over(grp))
        & (team == team.shift(2).over(grp))
    ).fill_null(False)

    work = work.with_columns(
        pl.when(idx < WINDOW - 1)
        .then(pl.lit("insufficient_games"))
        .when(consecutive)
        .then(pl.lit("ok"))
        .otherwise(pl.lit("gap_or_team_change"))
        .alias("history_status")
    ).with_columns((pl.col("history_status") == "ok").alias("is_model_ready"))

    work = work.with_columns(
        [
            pl.when(pl.col("is_model_ready"))
            .then(pl.col(src).cast(pl.Float64).rolling_mean(window_size=WINDOW, min_samples=1).over(grp))
            .otherwise(None)
            .alias(name)
            for name, src in ROLLING_SOURCES.items()
        ]
    )
    return work.sort(["season", "week", "player_id"])


def verify_no_future_information(
    df: pl.DataFrame, cutoff_weeks: tuple[int, ...] = (5, 9, 13)
) -> dict:
    """Empirical leakage check on real data.

    For each season and cutoff week ``k``: drop every game after week ``k``,
    recompute the features, and require that features for weeks <= ``k`` are
    IDENTICAL to those computed with the full season available. If any feature
    depended on a later game, the values would differ.

    Raises ``AssertionError`` on any mismatch; otherwise returns a summary dict.
    """
    assert_feature_columns_safe()
    full = add_rolling_features(df)
    key = ["player_id", "game_id"]
    checks = 0
    rows_compared = 0
    for season in sorted(df["season"].unique().to_list()):
        for k in cutoff_weeks:
            truncated = add_rolling_features(df.filter(~((pl.col("season") == season) & (pl.col("week") > k))))
            left = truncated.filter((pl.col("season") == season) & (pl.col("week") <= k)).select(
                key + FEATURE_COLUMNS + ["history_status"]
            )
            right = full.filter((pl.col("season") == season) & (pl.col("week") <= k)).select(
                key + FEATURE_COLUMNS + ["history_status"]
            )
            joined = left.join(right, on=key, how="full", suffix="_full")
            if joined.height != left.height or joined.height != right.height:
                raise AssertionError(f"Row sets differ for season {season}, cutoff week {k}")
            for col in FEATURE_COLUMNS + ["history_status"]:
                differs = joined.filter(~pl.col(col).eq_missing(pl.col(f"{col}_full")))
                if differs.height:
                    raise AssertionError(
                        f"Future information leak: '{col}' changed for {differs.height} rows "
                        f"(season {season}, cutoff week {k})"
                    )
            checks += 1
            rows_compared += left.height
    return {
        "checks_run": checks,
        "rows_compared": rows_compared,
        "cutoff_weeks": list(cutoff_weeks),
        "feature_columns": list(FEATURE_COLUMNS),
        "feature_columns_disjoint_from_audit_label_metadata": True,
    }

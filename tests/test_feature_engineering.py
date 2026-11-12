"""Offline tests for rolling features and future-information safety.

All data here is a tiny synthetic fixture (see tests/synthetic.py), used only to
test logic - never as research results.
"""

from __future__ import annotations

import polars as pl
import pytest

import src.feature_engineering as fe
from src.feature_engineering import (
    AUDIT_COLUMNS,
    FEATURE_COLUMNS,
    LABEL_COLUMNS,
    METADATA_COLUMNS,
    add_rolling_features,
    assert_feature_columns_safe,
    get_model_inputs,
    verify_no_future_information,
)
from src.preprocessing import (
    OUTPUT_COLUMNS,
    attach_team_schedule,
    chronological_split,
    clean_player_games,
    prepare_dataset,
    save_processed,
)
from tests.synthetic import make_schedule, make_stats


def featurize(stats: pl.DataFrame, schedule: pl.DataFrame) -> pl.DataFrame:
    games = attach_team_schedule(clean_player_games(stats), schedule)
    return add_rolling_features(games)


def at(df: pl.DataFrame, player: str, week: int, season: int = 2030) -> dict:
    out = df.filter((pl.col("player_id") == player) & (pl.col("week") == week) & (pl.col("season") == season))
    assert out.height == 1
    return out.to_dicts()[0]


# ------------------------------------------------------------- rolling averages


def test_rolling_three_game_averages_include_current_game_only_past():
    schedule = make_schedule([1, 2, 3, 4, 5])
    games = {1: (2, 0), 2: (4, 2), 3: (6, 3), 4: (10, 0), 5: (0, 9)}
    stats = make_stats(
        "p",
        games,
        receptions={1: 0.0, 2: 1.0, 3: 2.0, 4: 3.0, 5: 4.0},
        rushing_yards={1: 10.0, 2: 20.0, 3: 30.0, 4: 40.0, 5: 50.0},
        receiving_yards={1: 0.0, 2: 10.0, 3: 20.0, 4: 30.0, 5: 40.0},
        fantasy_points_ppr={1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0, 5: 5.0},
    )
    df = featurize(stats, schedule)

    w3 = at(df, "p", 3)  # weeks 1, 2, 3
    assert w3["is_model_ready"] is True
    assert w3["avg3_carries"] == pytest.approx((2 + 4 + 6) / 3)
    assert w3["avg3_targets"] == pytest.approx((0 + 2 + 3) / 3)
    assert w3["avg3_opportunities"] == pytest.approx((2 + 6 + 9) / 3)
    assert w3["avg3_receptions"] == pytest.approx(1.0)
    assert w3["avg3_rushing_yards"] == pytest.approx(20.0)
    assert w3["avg3_receiving_yards"] == pytest.approx(10.0)
    assert w3["avg3_fantasy_points_ppr"] == pytest.approx(2.0)

    w4 = at(df, "p", 4)  # weeks 2, 3, 4: week 1 drops out
    assert w4["avg3_carries"] == pytest.approx((4 + 6 + 10) / 3)
    w5 = at(df, "p", 5)  # weeks 3, 4, 5
    assert w5["avg3_opportunities"] == pytest.approx((9 + 10 + 9) / 3)


def test_current_game_features_and_opportunities():
    df = featurize(make_stats("p", {1: (3, 4), 2: (1, 1), 3: (2, 2)}), make_schedule([1, 2, 3]))
    w3 = at(df, "p", 3)
    assert (w3["carries"], w3["targets"], w3["opportunities"]) == (2, 2, 4)


def test_three_completed_games_of_history_required():
    df = featurize(make_stats("p", {1: (3, 4), 2: (1, 1), 3: (2, 2)}), make_schedule([1, 2, 3]))
    assert at(df, "p", 1)["is_model_ready"] is False
    assert at(df, "p", 2)["is_model_ready"] is False
    assert at(df, "p", 1)["avg3_opportunities"] is None
    assert at(df, "p", 2)["avg3_carries"] is None
    assert at(df, "p", 3)["is_model_ready"] is True


def test_missing_optional_values_never_drop_rows_or_get_zero_filled():
    schedule = make_schedule([1, 2, 3, 4])
    stats = make_stats(
        "p",
        {1: (1, 1), 2: (1, 1), 3: (1, 1), 4: (1, 1)},
        target_share={1: None, 2: None, 3: None, 4: None},
        receptions={1: 3.0, 2: None, 3: 5.0, 4: None},
        fantasy_points_ppr={1: None, 2: None, 3: None, 4: None},
    )
    df = featurize(stats, schedule)
    assert df.height == 4
    w3 = at(df, "p", 3)
    assert w3["is_model_ready"] is True
    assert w3["target_share"] is None
    assert w3["avg3_receptions"] == pytest.approx(4.0)  # mean of the non-null values (3, 5), not zero-filled
    assert w3["avg3_fantasy_points_ppr"] is None  # all missing -> stays missing
    assert at(df, "p", 4)["avg3_receptions"] == pytest.approx(5.0)


def test_players_are_windowed_independently():
    schedule = make_schedule([1, 2, 3])
    stats = pl.concat([make_stats("a", {1: (1, 0), 2: (1, 0), 3: (1, 0)}), make_stats("b", {1: (9, 0), 2: (9, 0), 3: (9, 0)})])
    df = featurize(stats, schedule)
    assert at(df, "a", 3)["avg3_carries"] == pytest.approx(1.0)
    assert at(df, "b", 3)["avg3_carries"] == pytest.approx(9.0)


# --------------------------------------------------------------- leakage safety


def _prepared_synthetic():
    schedule = pl.concat([make_schedule(list(range(1, 11)), season=2030), make_schedule(list(range(1, 11)), season=2031)])
    stats = pl.concat(
        [
            make_stats("a", {w: (w % 5 + 1, w % 3) for w in range(1, 11)}, season=2030),
            make_stats("a", {w: (w % 4, w % 6 + 1) for w in range(1, 11)}, season=2031),
            make_stats("b", {w: (w * 2, 1) for w in range(1, 11) if w != 4}, season=2030, position="WR"),
        ]
    )
    return stats, schedule


def test_changing_future_games_does_not_change_earlier_features():
    stats, schedule = _prepared_synthetic()
    games = attach_team_schedule(clean_player_games(stats), schedule)
    base = add_rolling_features(games)

    cutoff = 6
    perturbed = games.with_columns(
        [
            pl.when((pl.col("season") == 2030) & (pl.col("week") > cutoff)).then(pl.col(c) + 1000).otherwise(pl.col(c)).alias(c)
            for c in ("carries", "targets", "opportunities", "receptions", "rushing_yards", "receiving_yards", "fantasy_points_ppr")
        ]
    )
    changed = add_rolling_features(perturbed)

    key = ["player_id", "game_id"]
    left = base.filter((pl.col("season") == 2030) & (pl.col("week") <= cutoff)).select(key + FEATURE_COLUMNS).sort(key)
    right = changed.filter((pl.col("season") == 2030) & (pl.col("week") <= cutoff)).select(key + FEATURE_COLUMNS).sort(key)
    assert left.equals(right)
    # sanity: later rows really did change, so the test is able to detect leakage
    later_a = base.filter((pl.col("season") == 2030) & (pl.col("week") == 9)).select(FEATURE_COLUMNS).sort(FEATURE_COLUMNS)
    later_b = changed.filter((pl.col("season") == 2030) & (pl.col("week") == 9)).select(FEATURE_COLUMNS).sort(FEATURE_COLUMNS)
    assert not later_a.equals(later_b)


def test_truncation_check_passes_for_correct_features():
    stats, schedule = _prepared_synthetic()
    games = attach_team_schedule(clean_player_games(stats), schedule)
    summary = verify_no_future_information(games, cutoff_weeks=(4, 7))
    assert summary["checks_run"] == 4  # 2 seasons x 2 cutoffs
    assert summary["feature_columns_disjoint_from_audit_label_metadata"] is True


def test_truncation_check_detects_a_leaky_feature(monkeypatch):
    """The detector must fail if a feature peeks at the next game."""
    stats, schedule = _prepared_synthetic()
    games = attach_team_schedule(clean_player_games(stats), schedule)
    original = fe.add_rolling_features

    def leaky(df):
        out = original(df)
        peek = out.sort(["player_id", "season", "week"]).with_columns(
            pl.col("opportunities").shift(-1).over(["player_id", "season"]).alias("avg3_opportunities")
        )
        return peek.sort(["season", "week", "player_id"])

    monkeypatch.setattr(fe, "add_rolling_features", leaky)
    with pytest.raises(AssertionError, match="leak"):
        verify_no_future_information(games, cutoff_weeks=(4,))


def test_feature_columns_are_disjoint_from_audit_label_and_metadata():
    assert_feature_columns_safe()
    assert not set(FEATURE_COLUMNS) & set(AUDIT_COLUMNS)
    assert not set(FEATURE_COLUMNS) & set(LABEL_COLUMNS)
    assert not set(FEATURE_COLUMNS) & set(METADATA_COLUMNS)
    assert not any(c.startswith("next_") for c in FEATURE_COLUMNS)


def test_feature_guard_rejects_overlap(monkeypatch):
    monkeypatch.setattr(fe, "FEATURE_COLUMNS", FEATURE_COLUMNS + ["next_game_opportunities"])
    with pytest.raises(AssertionError):
        fe.assert_feature_columns_safe()


def test_model_inputs_exclude_future_derived_and_label_columns():
    stats, schedule = _prepared_synthetic()
    full, _, _ = prepare_dataset(stats, schedule, verify_leakage=False)
    inputs = get_model_inputs(full)
    assert inputs.columns == FEATURE_COLUMNS
    for col in AUDIT_COLUMNS + LABEL_COLUMNS:
        assert col not in inputs.columns
    # the future-derived audit columns are present in the table, but separate
    for col in AUDIT_COLUMNS + LABEL_COLUMNS:
        assert col in full.columns


def test_next_game_stats_change_labels_but_not_features():
    stats, schedule = _prepared_synthetic()
    full, _, _ = prepare_dataset(stats, schedule, verify_leakage=False)
    altered_stats = stats.with_columns(
        pl.when((pl.col("player_id") == "a") & (pl.col("season") == 2030) & (pl.col("week") == 8))
        .then(pl.col("carries") + 40)
        .otherwise(pl.col("carries"))
        .alias("carries")
    )
    altered, _, _ = prepare_dataset(altered_stats, schedule, verify_leakage=False)
    wk7_before = at(full, "a", 7)
    wk7_after = at(altered, "a", 7)
    assert all(wk7_before[c] == wk7_after[c] for c in FEATURE_COLUMNS)  # features unchanged
    assert wk7_before["next_game_opportunities"] != wk7_after["next_game_opportunities"]  # audit changed
    assert wk7_after["role_label"] == "Increasing"


# ----------------------------------------------- identifiers for the NLP join


def test_saved_datasets_keep_join_keys_and_separate_audit_columns(tmp_path):
    stats, schedule = _prepared_synthetic()
    full, report, leakage = prepare_dataset(stats, schedule, verify_leakage=False)
    splits = chronological_split(full, train_seasons=(2030,), validation_seasons=(2031,), test_seasons=(2032,))
    paths = save_processed(full, splits, report, out_dir=tmp_path, leakage=leakage)

    for name in ("train", "validation", "test", "player_weeks_features"):
        df = pl.read_parquet(paths[name])
        for key in ("player_id", "game_id", "season", "week", "team"):
            assert key in df.columns
        if df.height:
            assert df.select(pl.col(["player_id", "game_id", "season", "week", "team"]).null_count()).row(0) == (0,) * 5
    train = pl.read_parquet(paths["train"])
    assert train.columns == OUTPUT_COLUMNS
    assert set(train["season"]) <= {2030}
    assert paths["preparation_report"].exists()

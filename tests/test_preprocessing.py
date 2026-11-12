"""Offline tests for cleaning, next-game lookup, labels, and splits.

All data here is a tiny synthetic fixture (see tests/synthetic.py), used only to
test logic - never as research results.
"""

from __future__ import annotations

import polars as pl
import pytest

from src.preprocessing import (
    PreparationReport,
    assign_role_labels,
    build_team_schedule,
    chronological_split,
    clean_player_games,
    label_distribution_table,
    prepare_dataset,
)
from tests.synthetic import game_id, make_schedule, make_stats


def run(stats: pl.DataFrame, schedule: pl.DataFrame):
    full, report, _ = prepare_dataset(stats, schedule, verify_leakage=False)
    return full, report


def row(full: pl.DataFrame, player: str, week: int, season: int = 2030) -> dict:
    out = full.filter((pl.col("player_id") == player) & (pl.col("week") == week) & (pl.col("season") == season))
    assert out.height == 1
    return out.to_dicts()[0]


# ---------------------------------------------------------------- opportunities


def test_opportunities_is_carries_plus_targets():
    stats = make_stats("p", {1: (10, 3), 2: (0, 0), 3: (4, 7)})
    cleaned = clean_player_games(stats)
    assert cleaned.sort("week")["opportunities"].to_list() == [13, 0, 11]


# ------------------------------------------------------------------- thresholds


def _label(baseline_sum: int, nxt: int | None) -> str | None:
    """Label for a baseline given as the integer sum of 3 games and next opportunities."""
    df = pl.DataFrame(
        {
            "avg3_opportunities": [baseline_sum / 3],
            "next_game_opportunities": [nxt],
            "is_model_ready": [True],
            "next_game_id": ["g"],
        },
        schema_overrides={"next_game_opportunities": pl.Int64},
    )
    return assign_role_labels(df)["role_label"][0]


@pytest.mark.parametrize(
    "baseline_sum,nxt,expected",
    [
        # baseline 10 -> threshold max(3, 3.0) = 3.0, exact boundaries inclusive
        (30, 13, "Increasing"),
        (30, 7, "Decreasing"),
        (30, 12, "Stable"),
        (30, 8, "Stable"),
        # baseline 20 -> threshold 6
        (60, 26, "Increasing"),
        (60, 14, "Decreasing"),
        (60, 25, "Stable"),
        # baseline 30 -> threshold 9, exact boundary
        (90, 39, "Increasing"),
        (90, 21, "Decreasing"),
        (90, 38, "Stable"),
        # low baseline 5 -> floor of 3.0 applies (0.3*5 = 1.5)
        (15, 8, "Increasing"),
        (15, 2, "Decreasing"),
        (15, 7, "Stable"),
        # non-integer baseline 31/3: threshold 3.1
        (31, 14, "Increasing"),
        (31, 13, "Stable"),
    ],
)
def test_threshold_rule(baseline_sum, nxt, expected):
    assert _label(baseline_sum, nxt) == expected


@pytest.mark.parametrize(
    "baseline_sum,nxt,expected",
    [
        (0, 0, "Stable"),  # zero baseline, zero next: no change
        (0, 2, "Stable"),  # below floor of 3
        (0, 3, "Increasing"),  # change == floor
        (30, 0, "Decreasing"),  # a recorded zero-opportunity game is a valid 0
        (3, 0, "Stable"),  # baseline 1, next 0: change -1 within floor
        (9, 0, "Decreasing"),  # baseline 3, next 0: change -3 == floor
    ],
)
def test_zero_opportunity_edge_cases(baseline_sum, nxt, expected):
    assert _label(baseline_sum, nxt) == expected


def test_unknown_future_outcome_gets_no_label():
    assert _label(30, None) is None


# --------------------------------------------------------------------- bye week


def test_bye_week_is_skipped_for_next_game_and_window():
    # team plays weeks 1-4 then 6-8 (bye in week 5)
    schedule = make_schedule([1, 2, 3, 4, 6, 7, 8])
    stats = make_stats("p", {1: (5, 0), 2: (5, 0), 3: (5, 0), 4: (10, 0), 6: (20, 0), 7: (5, 0), 8: (5, 0)})
    full, _ = run(stats, schedule)

    week4 = row(full, "p", 4)
    assert week4["next_game_id"] == game_id(6)
    assert week4["next_game_week"] == 6
    assert week4["next_game_opportunities"] == 20  # week 6, not an assumed week 5

    # the window across the bye is still 3 consecutive TEAM games (weeks 3, 4, 6)
    week6 = row(full, "p", 6)
    assert week6["is_model_ready"] is True
    assert week6["avg3_opportunities"] == pytest.approx((5 + 10 + 20) / 3)


# ------------------------------------------------------------------ missed games


def test_missed_future_game_leaves_label_missing_not_decreasing():
    # team plays weeks 1-6; player has no record for week 4
    schedule = make_schedule([1, 2, 3, 4, 5, 6])
    stats = make_stats("p", {1: (5, 0), 2: (5, 0), 3: (5, 0), 5: (5, 0), 6: (5, 0)})
    full, _ = run(stats, schedule)

    week3 = row(full, "p", 3)
    assert week3["is_model_ready"] is True
    assert week3["next_game_id"] == game_id(4)
    assert week3["next_game_opportunities"] is None
    assert week3["role_label"] is None
    assert week3["label_missing_reason"] == "no_next_game_stats"


def test_missed_game_in_history_window_excludes_row():
    schedule = make_schedule([1, 2, 3, 4, 5, 6, 7])
    stats = make_stats("p", {1: (5, 0), 2: (5, 0), 3: (5, 0), 5: (5, 0), 6: (5, 0), 7: (5, 0)})
    full, report = run(stats, schedule)

    # windows (2,3,5) and (3,5,6) skip team game 4 -> not model-ready
    assert row(full, "p", 5)["history_status"] == "gap_or_team_change"
    assert row(full, "p", 6)["history_status"] == "gap_or_team_change"
    assert row(full, "p", 5)["avg3_opportunities"] is None
    # (5,6,7) are consecutive again
    assert row(full, "p", 7)["is_model_ready"] is True
    assert report.counts["excluded_gap_or_team_change_history"] == 2


def test_season_boundary_has_no_next_game_and_seasons_are_independent():
    schedule = pl.concat([make_schedule([1, 2, 3, 4], season=2030), make_schedule([1, 2, 3, 4], season=2031)])
    stats = pl.concat(
        [
            make_stats("p", {1: (5, 0), 2: (5, 0), 3: (5, 0), 4: (5, 0)}, season=2030),
            make_stats("p", {1: (50, 0), 2: (50, 0), 3: (50, 0), 4: (50, 0)}, season=2031),
        ]
    )
    full, _ = run(stats, schedule)

    last_2030 = row(full, "p", 4, 2030)
    assert last_2030["next_game_id"] is None
    assert last_2030["role_label"] is None
    assert last_2030["label_missing_reason"] == "season_end"

    # first weeks of 2031 do not borrow 2030 history
    assert row(full, "p", 1, 2031)["history_status"] == "insufficient_games"
    assert row(full, "p", 3, 2031)["avg3_opportunities"] == pytest.approx(50.0)


def test_traded_player_is_handled_conservatively():
    schedule = pl.concat([make_schedule([1, 2, 3, 4, 5, 6], team="AAA"), make_schedule([1, 2, 3, 4, 5, 6], team="CCC")])
    stats = pl.concat(
        [
            make_stats("p", {1: (5, 0), 2: (5, 0), 3: (5, 0)}, team="AAA"),
            make_stats("p", {4: (5, 0), 5: (5, 0), 6: (5, 0)}, team="CCC"),
        ]
    )
    full, _ = run(stats, schedule)

    # last AAA game: his next AAA game has no record for him -> missing, not Decreasing
    wk3 = row(full, "p", 3)
    assert wk3["next_game_id"] == game_id(4, team="AAA")
    assert wk3["role_label"] is None
    # first CCC games: window spans a team change -> not model-ready
    assert row(full, "p", 4)["history_status"] == "gap_or_team_change"
    assert row(full, "p", 5)["history_status"] == "gap_or_team_change"
    # three consecutive CCC games -> ready again
    assert row(full, "p", 6)["is_model_ready"] is True


def test_build_team_schedule_next_game_and_bye():
    sched = build_team_schedule(make_schedule([1, 2, 4]))
    s = sched.filter(pl.col("team") == "AAA").sort("team_game_number")  # opponent ZZZ is also listed
    assert s["team_game_number"].to_list() == [1, 2, 3]
    assert s["next_game_week"].to_list() == [2, 4, None]


# -------------------------------------------------------------------- cleaning


def test_cleaning_filters_dedupes_and_counts():
    good = make_stats("good", {1: (5, 2)})
    qb = make_stats("qb", {1: (1, 0)}, position="QB")
    post = make_stats("post", {1: (1, 0)}, season_type="POST")
    bad_carries = make_stats("bad", {1: (-1, 2)})
    dup = pl.concat([make_stats("dup", {1: (3, 3)}), make_stats("dup", {1: (3, 3)})])
    report = PreparationReport()
    cleaned = clean_player_games(pl.concat([good, qb, post, bad_carries, dup]), report)

    assert sorted(cleaned["player_id"].to_list()) == ["dup", "good"]
    assert report.counts["removed_non_regular_season"] == 1
    assert report.counts["removed_non_rb_wr_te"] == 1
    assert report.counts["removed_invalid_required_values"] == 1
    assert report.counts["removed_duplicate_player_games"] == 1


def test_missing_optional_values_do_not_discard_rows():
    stats = make_stats(
        "p",
        {1: (5, 2), 2: (5, 2)},
        target_share={1: None, 2: 0.2},
        receptions={1: None, 2: 2.0},
        fantasy_points_ppr={1: None, 2: None},
    )
    report = PreparationReport()
    cleaned = clean_player_games(stats, report)
    assert cleaned.height == 2
    assert report.counts["removed_invalid_required_values"] == 0
    assert report.counts["kept_rows_with_missing_optional_values"] == 2


def test_invalid_optional_values_become_null_but_rows_kept():
    stats = make_stats("p", {1: (5, 2), 2: (5, 2)}, target_share={1: 1.7, 2: 0.2}, receptions={1: -3.0, 2: 1.0})
    report = PreparationReport()
    cleaned = clean_player_games(stats, report).sort("week")
    assert cleaned.height == 2
    assert cleaned["target_share"].to_list() == [None, 0.2]
    assert cleaned["receptions"].to_list() == [None, 1.0]
    assert report.counts["optional_values_set_to_null_invalid"] == 2


def test_negative_yards_are_valid():
    stats = make_stats("p", {1: (5, 2)}, rushing_yards=-4.0, fantasy_points_ppr=-1.0)
    cleaned = clean_player_games(stats)
    assert cleaned["rushing_yards"].to_list() == [-4.0]


def test_records_are_ordered_chronologically():
    schedule = pl.concat([make_schedule([1, 2], season=2030), make_schedule([1, 2], season=2031)])
    stats = pl.concat(
        [
            make_stats("b", {2: (1, 1), 1: (1, 1)}, season=2031),
            make_stats("a", {2: (1, 1), 1: (1, 1)}, season=2030),
        ]
    ).sample(fraction=1.0, shuffle=True, seed=3)
    cleaned = clean_player_games(stats)
    keys = list(zip(cleaned["season"].to_list(), cleaned["week"].to_list()))
    assert keys == sorted(keys)


def test_not_on_schedule_rows_are_excluded_and_counted():
    schedule = make_schedule([1, 2])
    stats = make_stats("p", {1: (1, 1), 2: (1, 1), 3: (1, 1)})  # week 3 not scheduled
    _, report = run(stats, schedule)
    assert report.counts["removed_not_on_schedule"] == 1


# --------------------------------------------------------------------- splitting


def _fake_full(seasons_labels: list[tuple[int, int, str | None, bool]]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {"season": s, "week": w, "role_label": lab, "is_model_ready": ready, "player_id": "p"}
            for s, w, lab, ready in seasons_labels
        ],
        schema={"season": pl.Int64, "week": pl.Int64, "role_label": pl.String, "is_model_ready": pl.Boolean, "player_id": pl.String},
    )


def test_chronological_split_assigns_seasons_and_keeps_order():
    df = _fake_full(
        [(s, w, "Stable", True) for s in (2022, 2023, 2024, 2025) for w in (3, 10)]
        + [(2023, 5, None, True), (2024, 6, "Stable", False)]  # unlabeled / not ready are excluded
    )
    splits = chronological_split(df)
    assert set(splits["train"]["season"]) == {2022, 2023}
    assert set(splits["validation"]["season"]) == {2024}
    assert set(splits["test"]["season"]) == {2025}
    assert splits["train"].height == 4 and splits["validation"].height == 2 and splits["test"].height == 2
    assert splits["train"]["season"].max() < splits["validation"]["season"].min()
    assert splits["validation"]["season"].max() < splits["test"]["season"].min()


def test_chronological_split_rejects_overlap_or_wrong_order():
    df = _fake_full([(2022, 3, "Stable", True)])
    with pytest.raises(ValueError):
        chronological_split(df, train_seasons=(2022, 2023), validation_seasons=(2023,), test_seasons=(2025,))
    with pytest.raises(ValueError):
        chronological_split(df, train_seasons=(2024,), validation_seasons=(2023,), test_seasons=(2025,))


def test_label_distribution_table_counts_and_percentages():
    splits = {
        "train": _fake_full([(2022, 3, "Stable", True)] * 6 + [(2022, 4, "Increasing", True)] * 3 + [(2022, 5, "Decreasing", True)]),
        "validation": _fake_full([(2024, 3, "Stable", True)]),
        "test": _fake_full([]),
    }
    table = label_distribution_table(splits)
    train = {r["label"]: r for r in table.filter(pl.col("split") == "train").to_dicts()}
    assert train["Stable"]["count"] == 6 and train["Stable"]["percent"] == pytest.approx(60.0)
    assert train["Increasing"]["percent"] == pytest.approx(30.0)
    assert train["Decreasing"]["percent"] == pytest.approx(10.0)
    assert table.filter(pl.col("split") == "test")["percent"].sum() == 0.0

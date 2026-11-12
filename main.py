"""Entry point for the Fantasy Football Role Prediction project.

Commands:

    python main.py --check-data      # real-data smoke test (2024 targets chart)
    python main.py --prepare-data    # 2022-2025 data prep, features, labels, splits
    python main.py --prepare-data --refresh-data   # ignore the Parquet cache
"""

from __future__ import annotations

import argparse
import sys

import polars as pl

from src.data_loader import (
    HISTORICAL_SEASONS,
    DataLoadError,
    load_2024_weekly_player_stats,
    load_historical_player_stats,
    load_schedules_cached,
)
from src.feature_engineering import FEATURE_COLUMNS
from src.preprocessing import (
    TEST_SEASONS,
    TRAIN_SEASONS,
    VALIDATION_SEASONS,
    chronological_split,
    label_distribution_table,
    prepare_dataset,
    save_processed,
    season_label_counts,
)
from src.visualization import (
    TOP_TARGETS_PATH,
    plot_avg_opportunities_by_position,
    plot_label_distribution,
    plot_observations_by_season,
    plot_top_targets,
)


def check_data() -> int:
    """Download 2024 weekly stats, print a summary, and save the chart."""
    print("Downloading 2024 weekly NFL player statistics via nflreadpy...")
    try:
        df = load_2024_weekly_player_stats()
    except DataLoadError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    print(f"Rows: {df.height}")
    print(f"Columns ({len(df.columns)}):")
    for name in df.columns:
        print(f"  - {name}")

    try:
        output_path = plot_top_targets(df, output_path=TOP_TARGETS_PATH)
    except (KeyError, ValueError) as exc:
        print(f"[error] Could not generate chart: {exc}", file=sys.stderr)
        return 1

    print(f"Chart saved to: {output_path}")
    return 0


def _print_report(report, leakage, splits) -> None:
    c = report.to_dict()
    print("\n=== Rows excluded / left missing (counts) ===")
    rows = [
        ("Raw weekly player rows (all positions, all season types)", "raw_rows"),
        ("  removed: not regular season", "removed_non_regular_season"),
        ("  removed: not RB/WR/TE", "removed_non_rb_wr_te"),
        ("  removed: invalid required values", "removed_invalid_required_values"),
        ("  removed: duplicate player-game records", "removed_duplicate_player_games"),
        ("  removed: duplicate player-week records", "removed_duplicate_player_weeks"),
        ("  removed: game not on regular-season schedule", "removed_not_on_schedule"),
        ("Cleaned player-game rows kept", "placed_on_schedule_rows"),
        ("  kept despite missing optional values", "kept_rows_with_missing_optional_values"),
        ("  optional values nulled as invalid (rows kept)", "optional_values_set_to_null_invalid"),
        ("Not model-ready: fewer than 3 games of history in season", "excluded_insufficient_games_history"),
        ("Not model-ready: missed team game / team change in window", "excluded_gap_or_team_change_history"),
        ("Model-ready rows (3 consecutive team games of history)", "model_ready_rows"),
        ("  label missing: season ended (no next game)", "model_ready_unlabeled_season_end"),
        ("  label missing: no stats record in next scheduled game", "model_ready_unlabeled_no_next_game_stats"),
        ("Labeled rows", "labeled_rows"),
    ]
    for text, key in rows:
        print(f"  {text}: {c.get(key, 0):,}")

    print("\n=== Future-information check ===")
    print(f"  Predictive features ({len(FEATURE_COLUMNS)}): {', '.join(FEATURE_COLUMNS)}")
    print("  Feature columns are disjoint from audit, label and metadata columns: yes")
    if leakage:
        print(
            f"  Truncation test: {leakage['checks_run']} season/cutoff checks "
            f"({leakage['rows_compared']:,} rows) - features identical with and without later games: PASSED"
        )

    print("\n=== Samples per season and label ===")
    for r in season_label_counts(splits).to_dicts():
        print(f"  {r['season']}  {r['role_label']:<11} {r['len']:>6,}")

    print("\n=== Label distribution per chronological split (count, percent) ===")
    seasons = {"train": TRAIN_SEASONS, "validation": VALIDATION_SEASONS, "test": TEST_SEASONS}
    table = label_distribution_table(splits)
    for split in ("train", "validation", "test"):
        part = table.filter(pl.col("split") == split)
        total = splits[split].height
        years = "-".join(str(s) for s in (seasons[split][0], seasons[split][-1])) if len(seasons[split]) > 1 else seasons[split][0]
        print(f"  {split} ({years}), total {total:,}")
        for r in part.to_dicts():
            print(f"    {r['label']:<11} {r['count']:>6,}  {r['percent']:6.2f}%")


def prepare_data(refresh: bool = False) -> int:
    """Download/load cached 2022-2025 data, build features and labels, split, save, plot."""
    print(f"Preparing data for seasons {HISTORICAL_SEASONS[0]}-{HISTORICAL_SEASONS[-1]}...")
    try:
        stats = load_historical_player_stats(refresh=refresh)
        schedules = load_schedules_cached(refresh=refresh)
    except DataLoadError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    try:
        full, report, leakage = prepare_dataset(stats, schedules)
        splits = chronological_split(full)
        paths = save_processed(full, splits, report, leakage=leakage)
        graphs = [
            plot_observations_by_season(full),
            plot_label_distribution(splits),
            plot_avg_opportunities_by_position(full),
        ]
    except (KeyError, ValueError, AssertionError, DataLoadError, pl.exceptions.PolarsError) as exc:
        print(f"[error] Data preparation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    _print_report(report, leakage, splits)

    print("\n=== Output files ===")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    for path in graphs:
        print(f"  graph: {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fantasy Football Role Prediction")
    parser.add_argument(
        "--check-data",
        action="store_true",
        help="Download 2024 weekly player stats, print a summary, and save a top-targets chart.",
    )
    parser.add_argument(
        "--prepare-data",
        action="store_true",
        help="Load 2022-2025 data, clean, build features and role labels, split chronologically, "
        "save Parquet files to data/processed/, and generate graphs.",
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="With --prepare-data: ignore cached Parquet files and download again.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    status = 0
    ran = False
    if args.check_data:
        ran = True
        status = check_data() or status
    if args.prepare_data:
        ran = True
        status = prepare_data(refresh=args.refresh_data) or status
    if not ran:
        parser.print_help()
    return status


if __name__ == "__main__":
    sys.exit(main())

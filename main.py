"""Entry point for the Fantasy Football Role Prediction project.

Currently supports only a real-data smoke test:

    python main.py --check-data
"""

from __future__ import annotations

import argparse
import sys

from src.data_loader import DataLoadError, load_2024_weekly_player_stats
from src.visualization import TOP_TARGETS_PATH, plot_top_targets


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fantasy Football Role Prediction")
    parser.add_argument(
        "--check-data",
        action="store_true",
        help="Download 2024 weekly player stats, print a summary, and save a top-targets chart.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.check_data:
        return check_data()
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

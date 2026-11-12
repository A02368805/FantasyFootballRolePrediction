"""Plotting utilities (real-data charts only)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: we only save files
import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402
import seaborn as sns  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRAPHS_DIR = PROJECT_ROOT / "results" / "graphs"
TOP_TARGETS_PATH = GRAPHS_DIR / "top_targets_2024.png"

FANTASY_POSITIONS = ["RB", "WR", "TE"]

# Candidate column names, in order of preference. nflreadpy has changed some
# column names across versions, so we pick whichever actually exists.
NAME_CANDIDATES = ["player_display_name", "player_name", "name"]
POSITION_CANDIDATES = ["position", "position_group"]
ID_CANDIDATES = ["player_id", "gsis_id"]
TARGETS_CANDIDATES = ["targets"]
SEASON_TYPE_CANDIDATES = ["season_type"]


def find_column(df: pl.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    """Return the first candidate column present in ``df``."""
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise KeyError(
            f"None of the expected columns {candidates} were found. "
            f"Available columns: {df.columns}"
        )
    return None


def compute_top_targets(
    df: pl.DataFrame,
    n: int = 10,
    positions: list[str] | None = None,
    regular_season_only: bool = True,
) -> pl.DataFrame:
    """Total targets per player, top ``n``.

    Returns columns: ``player``, ``position`` (may be null if the dataset has no
    position column), ``targets``. Rows are filtered to ``positions`` (default
    RB/WR/TE) when a position column exists, and to regular-season games when a
    ``season_type`` column exists and ``regular_season_only`` is True.
    """
    positions = positions or FANTASY_POSITIONS
    name_col = find_column(df, NAME_CANDIDATES)
    targets_col = find_column(df, TARGETS_CANDIDATES)
    pos_col = find_column(df, POSITION_CANDIDATES, required=False)
    id_col = find_column(df, ID_CANDIDATES, required=False)
    season_type_col = find_column(df, SEASON_TYPE_CANDIDATES, required=False)

    work = df
    if regular_season_only and season_type_col is not None:
        work = work.filter(pl.col(season_type_col) == "REG")
    if pos_col is not None:
        work = work.filter(pl.col(pos_col).is_in(positions))
    work = work.filter(pl.col(name_col).is_not_null())

    # Group by player id when available so two players sharing a name are not merged.
    group_cols = [c for c in (id_col, name_col, pos_col) if c is not None]
    totals = (
        work.group_by(group_cols)
        .agg(pl.col(targets_col).fill_null(0).sum().alias("targets"))
        .sort("targets", descending=True)
        .head(n)
    )

    return totals.select(
        pl.col(name_col).alias("player"),
        (pl.col(pos_col) if pos_col is not None else pl.lit(None, dtype=pl.String)).alias("position"),
        pl.col("targets"),
    )


def plot_top_targets(
    df: pl.DataFrame,
    output_path: Path = TOP_TARGETS_PATH,
    n: int = 10,
    season: int = 2024,
    regular_season_only: bool = True,
) -> Path:
    """Save a horizontal bar chart of the top ``n`` players by total targets."""
    top = compute_top_targets(df, n=n, regular_season_only=regular_season_only)
    if top.height == 0:
        raise ValueError("No rows left after filtering; cannot plot top targets.")

    has_position = find_column(df, POSITION_CANDIDATES, required=False) is not None
    labels = [
        f"{p} ({pos})" if pos is not None else p
        for p, pos in zip(top["player"].to_list(), top["position"].to_list())
    ]
    values = top["targets"].to_list()

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=values, y=labels, orient="h", color="#1f77b4", ax=ax)
    for i, v in enumerate(values):
        ax.text(v, i, f" {v:g}", va="center")

    scope = "RB/WR/TE" if has_position else "all players"
    season_scope = "regular season" if regular_season_only else "all games"
    ax.set_title(f"Top {len(values)} {scope} by total targets, {season} ({season_scope})")
    ax.set_xlabel("Total targets")
    ax.set_ylabel("Player")
    ax.set_xlim(0, max(values) * 1.12)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path

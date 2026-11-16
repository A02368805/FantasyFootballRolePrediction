"""Plotting utilities (real-data charts only)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: we only save files
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
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


# ---------------------------------------------------------------------------
# Data-preparation graphs (real processed data only)
# ---------------------------------------------------------------------------

OBSERVATIONS_BY_SEASON_PATH = GRAPHS_DIR / "observations_by_season.png"
LABEL_DISTRIBUTION_PATH = GRAPHS_DIR / "label_distribution.png"
AVG_OPPORTUNITIES_BY_POSITION_PATH = GRAPHS_DIR / "avg_opportunities_by_position.png"

LABEL_COLORS = {"Decreasing": "#d62728", "Stable": "#7f7f7f", "Increasing": "#2ca02c"}


def _save(fig, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_observations_by_season(full: pl.DataFrame, output_path: Path = OBSERVATIONS_BY_SEASON_PATH) -> Path:
    """Player-week observations per season at each stage: cleaned, model-ready
    (3 consecutive team games of history), and labeled."""
    stages = (
        full.group_by("season")
        .agg(
            pl.len().alias("Cleaned player-weeks"),
            pl.col("is_model_ready").sum().alias("Model-ready (3-game history)"),
            pl.col("role_label").is_not_null().sum().alias("Labeled"),
        )
        .sort("season")
    )
    seasons = [str(s) for s in stages["season"].to_list()]
    series = ["Cleaned player-weeks", "Model-ready (3-game history)", "Labeled"]
    colors = ["#9ecae1", "#4292c6", "#08519c"]

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.27
    for i, (name, color) in enumerate(zip(series, colors)):
        values = stages[name].to_list()
        xs = [j + (i - 1) * width for j in range(len(seasons))]
        bars = ax.bar(xs, values, width=width, label=name, color=color)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(seasons)))
    ax.set_xticklabels(seasons)
    ax.set_xlabel("Season")
    ax.set_ylabel("Number of RB/WR/TE player-week observations")
    ax.set_title("Player-week observations by season (regular season)")
    ax.margins(y=0.15)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    return _save(fig, output_path)


def plot_label_distribution(
    splits: dict[str, pl.DataFrame], output_path: Path = LABEL_DISTRIBUTION_PATH
) -> Path:
    """Counts of Decreasing / Stable / Increasing labels for each chronological split."""
    order = ["Decreasing", "Stable", "Increasing"]
    names = [n for n, df in splits.items() if df.height]
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.27
    for i, label in enumerate(order):
        counts = []
        for n in names:
            df = splits[n]
            counts.append(df.filter(pl.col("role_label") == label).height)
        xs = [j + (i - 1) * width for j in range(len(names))]
        bars = ax.bar(xs, counts, width=width, label=label, color=LABEL_COLORS[label])
        for bar, c, n in zip(bars, counts, names):
            pct = 100.0 * c / splits[n].height
            ax.text(bar.get_x() + bar.get_width() / 2, c, f"{c:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(names)))
    def _season_range(n: str) -> str:
        seasons = sorted(set(splits[n]["season"].to_list()))
        return str(seasons[0]) if len(seasons) == 1 else f"{seasons[0]}-{seasons[-1]}"

    ax.set_xticklabels([f"{n}\n({_season_range(n)})" for n in names])
    ax.set_xlabel("Chronological split (season)")
    ax.set_ylabel("Number of labeled player-weeks")
    ax.set_title("Experimental role-change label distribution by split")
    ax.legend(title="Label")
    ax.margins(y=0.15)
    return _save(fig, output_path)


def plot_avg_opportunities_by_position(
    full: pl.DataFrame, output_path: Path = AVG_OPPORTUNITIES_BY_POSITION_PATH
) -> Path:
    """Average offensive opportunities (carries + targets) per player-game by position,
    stacked into carries and targets. Uses all cleaned player-games."""
    agg = (
        full.group_by("position")
        .agg(pl.col("carries").mean().alias("carries"), pl.col("targets").mean().alias("targets"), pl.len().alias("n"))
        .sort("position")
    )
    positions = agg["position"].to_list()
    carries = agg["carries"].to_list()
    targets = agg["targets"].to_list()
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(positions, carries, label="Carries", color="#1f77b4")
    ax.bar(positions, targets, bottom=carries, label="Targets", color="#ff7f0e")
    for i, (c, t, n) in enumerate(zip(carries, targets, agg["n"].to_list())):
        ax.text(i, c + t, f"{c + t:.2f}\n(n={n:,})", ha="center", va="bottom", fontsize=9)
    seasons = sorted(full["season"].unique().to_list())
    ax.set_xlabel("Position")
    ax.set_ylabel("Average opportunities per player-game (carries + targets)")
    ax.set_title(f"Average offensive opportunities by position, {seasons[0]}-{seasons[-1]}")
    ax.legend()
    ax.margins(y=0.15)
    return _save(fig, output_path)


# ---------------------------------------------------------------------------
# Baseline model graphs (real validation results only; 2025 test is not used)
# ---------------------------------------------------------------------------

BASELINE_MACRO_F1_PATH = GRAPHS_DIR / "baseline_validation_macro_f1.png"
BASELINE_ACCURACY_PATH = GRAPHS_DIR / "baseline_validation_accuracy.png"
BASELINE_CONFUSION_PATH = GRAPHS_DIR / "baseline_validation_confusion_matrices.png"
BASELINE_PREDICTED_DIST_PATH = GRAPHS_DIR / "baseline_validation_predicted_label_distribution.png"

_MODEL_COLORS = ["#9e9e9e", "#1f77b4", "#ff7f0e", "#2ca02c"]


def plot_model_metric_bars(
    results: dict[str, dict], metric: str, ylabel: str, title: str, output_path: Path
) -> Path:
    """Bar chart of one overall metric (``macro_f1`` or ``accuracy``) across models."""
    names = list(results)
    values = [results[n][metric] for n in names]
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(names)), values, color=_MODEL_COLORS[: len(names)])
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=9)
    ax.set_xlabel("Model")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, min(1.0, max(values) * 1.2 + 0.02))
    return _save(fig, output_path)


def plot_confusion_matrices(
    results: dict[str, dict], output_path: Path = BASELINE_CONFUSION_PATH, split_name: str = "Validation (2024)"
) -> Path:
    """One confusion-matrix panel per model: counts annotated, colour = row-normalized recall."""
    names = list(results)
    cols = 2
    rows = (len(names) + cols - 1) // cols
    sns.set_theme(style="white")
    fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 5.2 * rows))
    axes = list(axes.flat) if hasattr(axes, "flat") else [axes]
    for ax, name in zip(axes, names):
        m = results[name]
        cm = np.array(m["confusion_matrix"], dtype=float)
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        labels = m["labels"]
        sns.heatmap(norm, annot=False, cmap="Blues", vmin=0, vmax=1, cbar=False, ax=ax,
                    xticklabels=labels, yticklabels=labels)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j + 0.5, i + 0.5, f"{int(cm[i, j])}\n({norm[i, j]:.0%})", ha="center", va="center",
                        color="white" if norm[i, j] > 0.5 else "black", fontsize=10)
        ax.set_yticklabels(labels, rotation=0, va="center")
        ax.set_title(f"{name}\nmacro F1 {m['macro_f1']:.3f}", fontsize=11)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
    for ax in axes[len(names):]:
        ax.axis("off")
    fig.suptitle(f"Confusion matrices, {split_name} (counts; percent of true class)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, output_path)


def plot_predicted_label_distribution(
    results: dict[str, dict], output_path: Path = BASELINE_PREDICTED_DIST_PATH
) -> Path:
    """Counts of each class predicted by each model, next to the true label counts."""
    names = list(results)
    labels = results[names[0]]["labels"]
    series = [("Actual (true labels)", [results[names[0]]["true_counts"][l] for l in labels], "#333333")]
    for n, color in zip(names, _MODEL_COLORS):
        series.append((n, [results[n]["predicted_counts"][l] for l in labels], color))
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(11, 6))
    width = 0.8 / len(series)
    for i, (name, counts, color) in enumerate(series):
        xs = [j + (i - (len(series) - 1) / 2) * width for j in range(len(labels))]
        bars = ax.bar(xs, counts, width=width, label=name, color=color)
        for bar, c in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, c, f"{c:,}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Class (true label for 'Actual', predicted label for models)")
    ax.set_ylabel("Number of validation player-weeks")
    ax.set_title("Label distribution by predicted class, Validation (2024)")
    ax.legend(fontsize=8)
    ax.margins(y=0.12)
    return _save(fig, output_path)

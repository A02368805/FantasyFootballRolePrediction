"""Data loading utilities.

Downloads real NFL data via ``nflreadpy`` and returns Polars DataFrames.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_SEASONS: tuple[int, ...] = (2024,)


class DataLoadError(RuntimeError):
    """Raised when NFL data cannot be downloaded or read."""


def raw_cache_path(seasons: Sequence[int], raw_dir: Path = RAW_DATA_DIR) -> Path:
    """Path of the local copy of the weekly player stats for ``seasons``."""
    label = "_".join(str(s) for s in seasons)
    return raw_dir / f"player_stats_week_{label}.parquet"


def load_weekly_player_stats(
    seasons: Sequence[int] = DEFAULT_SEASONS,
    save_raw: bool = True,
    raw_dir: Path = RAW_DATA_DIR,
) -> pl.DataFrame:
    """Download weekly NFL player statistics as a Polars DataFrame.

    Calls ``nflreadpy.load_player_stats(seasons, summary_level="week")``.

    If ``save_raw`` is True, a copy of a successful download is written to
    ``data/raw/`` (git-ignored). If the download fails and a previously saved
    copy of the *same real data* exists, that copy is used instead and a
    warning is printed. If neither is available, ``DataLoadError`` is raised.
    """
    seasons = list(seasons)
    cache_path = raw_cache_path(seasons, raw_dir)

    try:
        import nflreadpy as nfl
    except ImportError as exc:
        raise DataLoadError(
            "nflreadpy is not installed. Run: pip install -r requirements.txt"
        ) from exc

    try:
        df = nfl.load_player_stats(seasons, summary_level="week")
    except Exception as exc:  # network errors, HTTP errors, parse errors, etc.
        if cache_path.exists():
            print(
                f"[warning] Download failed ({type(exc).__name__}: {exc}). "
                f"Using previously downloaded copy: {cache_path}"
            )
            try:
                return pl.read_parquet(cache_path)
            except Exception as cache_exc:
                raise DataLoadError(
                    f"Download failed and the cached file {cache_path} could not be read: "
                    f"{cache_exc}"
                ) from exc
        raise DataLoadError(
            f"Could not download weekly player stats for seasons {seasons}: "
            f"{type(exc).__name__}: {exc}. Check your internet connection and try again."
        ) from exc

    if not isinstance(df, pl.DataFrame):
        # Defensive: nflreadpy returns Polars by default, but normalize if not.
        df = pl.from_pandas(df)

    if df.height == 0:
        raise DataLoadError(f"Download succeeded but returned no rows for seasons {seasons}.")

    if save_raw:
        try:
            raw_dir.mkdir(parents=True, exist_ok=True)
            df.write_parquet(cache_path)
        except OSError as exc:
            print(f"[warning] Could not save raw data to {cache_path}: {exc}")

    return df


def load_2024_weekly_player_stats(**kwargs) -> pl.DataFrame:
    """Convenience wrapper: 2024 weekly player stats."""
    return load_weekly_player_stats([2024], **kwargs)


# ---------------------------------------------------------------------------
# Historical (multi-season) loading with per-season Parquet caching
# ---------------------------------------------------------------------------

HISTORICAL_SEASONS: tuple[int, ...] = (2022, 2023, 2024, 2025)

# Column names verified against the real nflreadpy 0.1.5 schemas. Aliases are
# accepted so a future nflverse rename (e.g. ``recent_team``) fails gracefully.
COLUMN_ALIASES: dict[str, list[str]] = {
    "team": ["team", "recent_team"],
    "player_display_name": ["player_display_name", "player_name"],
    "position": ["position", "position_group"],
}

STATS_REQUIRED_COLUMNS = [
    "player_id",
    "position",
    "team",
    "season",
    "week",
    "season_type",
    "game_id",
    "carries",
    "targets",
]
SCHEDULE_REQUIRED_COLUMNS = ["game_id", "season", "week", "game_type", "home_team", "away_team"]


def schedule_cache_path(season: int, raw_dir: Path = RAW_DATA_DIR) -> Path:
    """Path of the cached schedule for one season."""
    return raw_dir / f"schedules_{season}.parquet"


def validate_columns(df: pl.DataFrame, required: Sequence[str], name: str) -> None:
    """Raise ``DataLoadError`` listing the real columns if any required one is missing."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataLoadError(
            f"{name} is missing required columns {missing}. Available columns: {df.columns}"
        )


def standardize_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Rename known alias columns (see ``COLUMN_ALIASES``) to their canonical names."""
    renames = {}
    for canonical, candidates in COLUMN_ALIASES.items():
        if canonical in df.columns:
            continue
        for candidate in candidates:
            if candidate in df.columns:
                renames[candidate] = canonical
                break
    return df.rename(renames) if renames else df


def _read_cached(path: Path) -> pl.DataFrame:
    try:
        return pl.read_parquet(path)
    except Exception as exc:
        raise DataLoadError(f"Cached file {path} could not be read: {exc}. Delete it to re-download.") from exc


def load_historical_player_stats(
    seasons: Sequence[int] = HISTORICAL_SEASONS,
    raw_dir: Path = RAW_DATA_DIR,
    refresh: bool = False,
) -> pl.DataFrame:
    """Weekly player stats for several seasons, cached per season as Parquet.

    Each season is read from ``data/raw/player_stats_week_<season>.parquet`` when it
    exists (unless ``refresh``); otherwise it is downloaded with
    ``nflreadpy.load_player_stats([season], summary_level="week")``. Returns the
    unfiltered seasons concatenated (all positions and season types).
    """
    frames = []
    for season in seasons:
        path = raw_cache_path([season], raw_dir)
        if path.exists() and not refresh:
            print(f"[cache] player stats {season}: {path.name}")
            frames.append(_read_cached(path))
        else:
            print(f"[download] player stats {season}")
            frames.append(load_weekly_player_stats([season], save_raw=True, raw_dir=raw_dir))
    df = standardize_columns(pl.concat(frames, how="diagonal_relaxed"))
    validate_columns(df, STATS_REQUIRED_COLUMNS, "Weekly player stats")
    return df


def load_schedules_cached(
    seasons: Sequence[int] = HISTORICAL_SEASONS,
    raw_dir: Path = RAW_DATA_DIR,
    refresh: bool = False,
) -> pl.DataFrame:
    """NFL schedules for several seasons, cached per season as Parquet.

    Uses ``nflreadpy.load_schedules([season])`` for seasons that are not cached.
    """
    try:
        import nflreadpy as nfl
    except ImportError as exc:
        raise DataLoadError("nflreadpy is not installed. Run: pip install -r requirements.txt") from exc

    frames = []
    for season in seasons:
        path = schedule_cache_path(season, raw_dir)
        if path.exists() and not refresh:
            print(f"[cache] schedule {season}: {path.name}")
            frames.append(_read_cached(path))
            continue
        print(f"[download] schedule {season}")
        try:
            sched = nfl.load_schedules([season])
        except Exception as exc:
            if path.exists():
                print(f"[warning] Schedule download failed ({exc}); using cached {path}")
                frames.append(_read_cached(path))
                continue
            raise DataLoadError(
                f"Could not download schedule for {season}: {type(exc).__name__}: {exc}. "
                "Check your internet connection and try again."
            ) from exc
        if not isinstance(sched, pl.DataFrame):
            sched = pl.from_pandas(sched)
        if sched.height == 0:
            raise DataLoadError(f"Schedule download for {season} returned no rows.")
        try:
            raw_dir.mkdir(parents=True, exist_ok=True)
            sched.write_parquet(path)
        except OSError as exc:
            print(f"[warning] Could not save schedule to {path}: {exc}")
        frames.append(sched)

    df = pl.concat(frames, how="diagonal_relaxed")
    validate_columns(df, SCHEDULE_REQUIRED_COLUMNS, "Schedules")
    return df


# ---------------------------------------------------------------------------
# Play-by-play text (slim, per-season Parquet cache)
# ---------------------------------------------------------------------------

# Only the columns needed to build player-game text documents. The full pbp table
# has ~370 columns and ~100 MB per season; we never store it.
PBP_COLUMNS = [
    "game_id",
    "play_id",
    "season",
    "week",
    "season_type",
    "play_type",
    "desc",
    "rusher_player_id",
    "receiver_player_id",
]


def pbp_cache_path(season: int, raw_dir: Path = RAW_DATA_DIR) -> Path:
    """Path of the cached slim play-by-play text table for one season."""
    return raw_dir / f"pbp_text_{season}.parquet"


def load_pbp_text_cached(
    seasons: Sequence[int] = HISTORICAL_SEASONS,
    raw_dir: Path = RAW_DATA_DIR,
    refresh: bool = False,
) -> pl.DataFrame:
    """Regular-season play-by-play (slim columns) for several seasons.

    Each season is downloaded separately with ``nflreadpy.load_pbp([season])``,
    reduced to ``PBP_COLUMNS`` and REG games only, and cached as
    ``data/raw/pbp_text_<season>.parquet``.
    """
    try:
        import nflreadpy as nfl
    except ImportError as exc:
        raise DataLoadError("nflreadpy is not installed. Run: pip install -r requirements.txt") from exc

    frames = []
    for season in seasons:
        path = pbp_cache_path(season, raw_dir)
        if path.exists() and not refresh:
            print(f"[cache] play-by-play text {season}: {path.name}")
            frames.append(_read_cached(path))
            continue
        print(f"[download] play-by-play {season} (large; only text columns are kept)")
        try:
            pbp = nfl.load_pbp([season])
        except Exception as exc:
            if path.exists():
                print(f"[warning] pbp download failed ({exc}); using cached {path}")
                frames.append(_read_cached(path))
                continue
            raise DataLoadError(
                f"Could not download play-by-play for {season}: {type(exc).__name__}: {exc}. "
                "Check your internet connection and try again."
            ) from exc
        if not isinstance(pbp, pl.DataFrame):
            pbp = pl.from_pandas(pbp)
        validate_columns(pbp, PBP_COLUMNS, f"Play-by-play {season}")
        slim = pbp.select(PBP_COLUMNS).filter(pl.col("season_type") == "REG")
        del pbp
        if slim.height == 0:
            raise DataLoadError(f"Play-by-play for {season} returned no regular-season rows.")
        try:
            raw_dir.mkdir(parents=True, exist_ok=True)
            slim.write_parquet(path)
        except OSError as exc:
            print(f"[warning] Could not save pbp text to {path}: {exc}")
        frames.append(slim)

    return pl.concat(frames, how="diagonal_relaxed")

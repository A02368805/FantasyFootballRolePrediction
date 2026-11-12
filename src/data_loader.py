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

"""Offline tests for per-season Parquet caching (nflreadpy is mocked).

The fake frames below are fixtures only; they are not NFL data.
"""

from __future__ import annotations

import nflreadpy
import polars as pl
import pytest

from src import data_loader


def fake_stats(season: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "player_id": ["x"],
            "position": ["RB"],
            "team": ["AAA"],
            "season": [season],
            "week": [1],
            "season_type": ["REG"],
            "game_id": [f"{season}_01_ZZZ_AAA"],
            "carries": [1],
            "targets": [1],
        }
    )


def fake_schedule(season: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_id": [f"{season}_01_ZZZ_AAA"],
            "season": [season],
            "week": [1],
            "game_type": ["REG"],
            "home_team": ["AAA"],
            "away_team": ["ZZZ"],
        }
    )


def test_stats_are_downloaded_once_then_read_from_parquet(monkeypatch, tmp_path):
    calls = []

    def fake_load(seasons, summary_level="week"):
        calls.append(list(seasons))
        assert summary_level == "week"
        return fake_stats(seasons[0])

    monkeypatch.setattr(nflreadpy, "load_player_stats", fake_load)
    first = data_loader.load_historical_player_stats([2030, 2031], raw_dir=tmp_path)
    assert calls == [[2030], [2031]]
    assert first.height == 2
    assert (tmp_path / "player_stats_week_2030.parquet").exists()

    def must_not_download(*args, **kwargs):
        raise AssertionError("should have used the Parquet cache")

    monkeypatch.setattr(nflreadpy, "load_player_stats", must_not_download)
    second = data_loader.load_historical_player_stats([2030, 2031], raw_dir=tmp_path)
    assert second.height == 2


def test_only_missing_seasons_are_downloaded(monkeypatch, tmp_path):
    fake_stats(2030).write_parquet(tmp_path / "player_stats_week_2030.parquet")
    calls = []

    def fake_load(seasons, summary_level="week"):
        calls.append(list(seasons))
        return fake_stats(seasons[0])

    monkeypatch.setattr(nflreadpy, "load_player_stats", fake_load)
    data_loader.load_historical_player_stats([2030, 2031], raw_dir=tmp_path)
    assert calls == [[2031]]


def test_schedules_are_cached_per_season(monkeypatch, tmp_path):
    calls = []

    def fake_load(seasons):
        calls.append(list(seasons))
        return fake_schedule(seasons[0])

    monkeypatch.setattr(nflreadpy, "load_schedules", fake_load)
    data_loader.load_schedules_cached([2030, 2031], raw_dir=tmp_path)
    assert calls == [[2030], [2031]]
    assert (tmp_path / "schedules_2031.parquet").exists()

    monkeypatch.setattr(nflreadpy, "load_schedules", lambda s: (_ for _ in ()).throw(AssertionError("cache miss")))
    assert data_loader.load_schedules_cached([2030, 2031], raw_dir=tmp_path).height == 2


def test_schedule_download_failure_gives_clear_error(monkeypatch, tmp_path):
    def boom(seasons):
        raise ConnectionError("offline")

    monkeypatch.setattr(nflreadpy, "load_schedules", boom)
    with pytest.raises(data_loader.DataLoadError, match="Could not download schedule"):
        data_loader.load_schedules_cached([2030], raw_dir=tmp_path)


def test_validate_columns_lists_available_columns():
    df = pl.DataFrame({"a": [1]})
    with pytest.raises(data_loader.DataLoadError, match="Available columns"):
        data_loader.validate_columns(df, ["a", "b"], "Test frame")


def test_standardize_columns_maps_aliases():
    df = pl.DataFrame({"recent_team": ["AAA"], "player_name": ["x"], "position_group": ["WR"]})
    out = data_loader.standardize_columns(df)
    assert set(out.columns) == {"team", "player_display_name", "position"}

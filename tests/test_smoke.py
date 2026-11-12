"""Offline tests. No internet required.

NOTE: ``synthetic_stats`` below is a tiny hand-written fixture used ONLY to test
the aggregation/filtering logic. It is not research data and is never used by
main.py or any chart.
"""

from __future__ import annotations

import matplotlib
import polars as pl
import pytest

matplotlib.use("Agg")

import main as main_module  # noqa: E402
from src import data_loader  # noqa: E402
from src.visualization import compute_top_targets, find_column, plot_top_targets  # noqa: E402


@pytest.fixture
def synthetic_stats() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "player_id": ["a", "a", "b", "c", "d", "e"],
            "player_display_name": ["Alice WR", "Alice WR", "Bob TE", "Cy QB", "Dee RB", "Eve WR"],
            "position": ["WR", "WR", "TE", "QB", "RB", "WR"],
            "season_type": ["REG", "REG", "REG", "REG", "REG", "POST"],
            "week": [1, 2, 1, 1, 1, 19],
            "targets": [5, 7, 10, 99, 3, 50],
        }
    )


def test_find_column_picks_first_available():
    df = pl.DataFrame({"player_name": ["x"], "position_group": ["WR"]})
    assert find_column(df, ["player_display_name", "player_name"]) == "player_name"
    assert find_column(df, ["position", "position_group"]) == "position_group"


def test_find_column_missing_raises():
    df = pl.DataFrame({"foo": [1]})
    with pytest.raises(KeyError):
        find_column(df, ["targets"])


def test_compute_top_targets_filters_and_sums(synthetic_stats):
    top = compute_top_targets(synthetic_stats, n=10)
    assert top["player"].to_list() == ["Alice WR", "Bob TE", "Dee RB"]  # no QB, no POST game
    assert top["targets"].to_list() == [12, 10, 3]


def test_compute_top_targets_respects_n(synthetic_stats):
    assert compute_top_targets(synthetic_stats, n=2).height == 2


def test_compute_top_targets_without_position_column(synthetic_stats):
    top = compute_top_targets(synthetic_stats.drop("position"), n=1)
    assert top["player"].to_list() == ["Cy QB"]
    assert top["position"].to_list() == [None]


def test_plot_top_targets_writes_file(synthetic_stats, tmp_path):
    out = plot_top_targets(synthetic_stats, output_path=tmp_path / "chart.png")
    assert out.exists() and out.stat().st_size > 0


def test_loader_raises_clear_error_when_download_fails(monkeypatch, tmp_path):
    import nflreadpy

    def boom(*args, **kwargs):
        raise ConnectionError("no network")

    monkeypatch.setattr(nflreadpy, "load_player_stats", boom)
    with pytest.raises(data_loader.DataLoadError, match="Could not download"):
        data_loader.load_weekly_player_stats([2024], raw_dir=tmp_path)


def test_main_check_data_returns_nonzero_on_load_failure(monkeypatch, capsys):
    def fail():
        raise data_loader.DataLoadError("offline")

    monkeypatch.setattr(main_module, "load_2024_weekly_player_stats", fail)
    assert main_module.main(["--check-data"]) == 1
    assert "offline" in capsys.readouterr().err

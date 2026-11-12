"""Tiny hand-written fixtures for unit tests ONLY.

These are not research data and are never used by main.py, the processed
datasets, or any graph. Seasons use fake years (2030+) and fake team codes so
they can never be mistaken for real NFL results.
"""

from __future__ import annotations

import polars as pl


def make_schedule(weeks: list[int], team: str = "AAA", season: int = 2030, opponent: str = "ZZZ") -> pl.DataFrame:
    """Regular-season schedule rows for ``team``. Weeks not listed are byes."""
    rows = [
        {
            "game_id": f"{season}_{wk:02d}_{opponent}_{team}",
            "season": season,
            "week": wk,
            "game_type": "REG",
            "home_team": team,
            "away_team": opponent,
        }
        for wk in weeks
    ]
    return pl.DataFrame(rows)


def game_id(week: int, team: str = "AAA", season: int = 2030, opponent: str = "ZZZ") -> str:
    return f"{season}_{week:02d}_{opponent}_{team}"


def make_stats(
    player_id: str,
    games: dict[int, tuple[int, int]],
    team: str = "AAA",
    season: int = 2030,
    position: str = "RB",
    opponent: str = "ZZZ",
    season_type: str = "REG",
    **optional,
) -> pl.DataFrame:
    """Player-week stat rows. ``games`` maps week -> (carries, targets).

    ``optional`` values (receptions, rushing_yards, receiving_yards, target_share,
    fantasy_points_ppr) are constants unless given as ``{week: value}`` dicts.
    """
    defaults = {
        "receptions": 1.0,
        "rushing_yards": 10.0,
        "receiving_yards": 5.0,
        "target_share": 0.1,
        "fantasy_points_ppr": 8.0,
    }
    defaults.update(optional)

    def val(name, wk):
        v = defaults[name]
        return v.get(wk) if isinstance(v, dict) else v

    rows = []
    for wk, (carries, targets) in sorted(games.items()):
        rows.append(
            {
                "player_id": player_id,
                "player_display_name": f"Player {player_id}",
                "position": position,
                "team": team,
                "season": season,
                "week": wk,
                "season_type": season_type,
                "game_id": game_id(wk, team, season, opponent),
                "carries": carries,
                "targets": targets,
                **{name: val(name, wk) for name in defaults},
            }
        )
    return pl.DataFrame(
        rows,
        schema_overrides={
            "receptions": pl.Float64,
            "rushing_yards": pl.Float64,
            "receiving_yards": pl.Float64,
            "target_share": pl.Float64,
            "fantasy_points_ppr": pl.Float64,
        },
    )

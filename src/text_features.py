"""NLP text features from NFL play-by-play descriptions.

Pipeline
--------
1. ``build_player_plays``: plays in which a player is the official rusher
   (``rusher_player_id`` on ``run`` plays) or receiver/target
   (``receiver_player_id`` on ``pass`` plays). Matching uses the official ids,
   never player names inside the text.
2. ``build_player_game_documents``: one document per ``(player_id, game_id)``,
   the cleaned play descriptions in ``play_id`` (chronological) order.
3. ``add_window_documents``: for each model-ready row, the text of the STRICT
   three-game window (current completed game plus the previous two consecutive
   team games). The game being predicted (the next game) and anything later is
   never included: the window game ids come only from the player's current and
   earlier rows.
4. ``build_tfidf_vectorizer``: fixed-setting TF-IDF (fit on training rows only
   by the caller / the modelling pipeline).

Cleaning masks player identity (jersey number + player name tokens) but keeps
football language, team abbreviations and yard lines. The TF-IDF token pattern is
alphabetic only, so numeric yardage ("for 12 yards") is NOT represented.

Overlap note: the same plays also produce ``carries``/``targets`` in the
statistical features, so text and statistics are not independent sources.
"""

from __future__ import annotations

import re
from collections import Counter

import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer

TEXT_COLUMN = "text"
KEYS = ["player_id", "game_id"]
WINDOW_COLUMNS = [
    "text",
    "text_empty",
    "window_n_plays",
    "window_n_rush_plays",
    "window_n_target_plays",
    "window_games_with_text",
]

# ---------------------------------------------------------------------------
# Description cleaning and player-identity masking
# ---------------------------------------------------------------------------

# Surnames may have a prefix ("St. Brown", "Vander Esch", "O'Neill") or a suffix ("Jr.").
# Longer prefixes must come first in the alternation (``Vander`` before ``Van``).
_SURNAME = r"(?:(?:Vander|Van|Von|St|Mc|Mac|De|Di|Du|La|Le|O|D)['.]?\s?)?[A-Za-z][A-Za-z'\-]*"
_SUFFIX = r"(?:\s+(?:Jr|Sr|II|III|IV)\b\.?)?"
_TEAM_PREFIX = r"(?:\b[A-Z]{2,3}-)?"  # e.g. "LV-21-A.Robertson"

_NAME_PATTERNS = [
    # 83-G.Dortch, 20-Bre.Hall, LV-21-A.Robertson, 14-A.St. Brown, 55-L.Vander Esch, 11-M.Parsons Jr.
    # An optional SECOND single-letter initial covers "99-J.J.Watt" / "11-A.J.Brown". It must be one
    # capital letter so a sentence-final short surname ("4-D.Hall. Penalty") cannot swallow the next word.
    re.compile(_TEAM_PREFIX + r"\b\d{1,2}-[A-Za-z]{1,4}\.(?:[A-Z]\.)?" + r"\s?" + _SURNAME + _SUFFIX),
    # bare initials + surname without a jersey number: "T.Hill at quarterback", "Key block: S.Brown"
    re.compile(r"\b[A-Z][A-Za-z]{0,3}\.(?:[A-Z]\.)?[A-Z][A-Za-z'\-]+" + _SUFFIX),
    # jersey number + surname only: "34-Edmunds"
    re.compile(_TEAM_PREFIX + r"\b\d{1,2}-[A-Z][A-Za-z'\-]+"),
    # "#6 Driskel in at QB", "#10 Berrios to #17 Wilson", "#14 M White", "LV#3 J Stidham"
    re.compile(r"#\d{1,2}\s+(?:[A-Z]\.?\s+)?[A-Z][A-Za-z'\-]+"),
    # full names in injury-update sentences: "Injury Update: Jayron Kearse has a knee injury"
    re.compile(r"(?<=Injury Update: )(?:[A-Z][A-Za-z.'\-]+\s+){1,3}(?=(?:has|was|is|will|returned)\b)"),
]
_LEADING_CLOCK = re.compile(r"^\s*\(\d{0,2}:\d{2}\)\s*")
_NON_TEXT = re.compile(r"[^\w\s'\-]")
_SPACES = re.compile(r"\s+")


def mask_player_names(desc: str) -> str:
    """Remove jersey-number / player-name tokens (offense and defense) from a description."""
    for pattern in _NAME_PATTERNS:
        desc = pattern.sub(" ", desc)
    return desc


def clean_description(desc: str | None) -> str:
    """Mask player identity, drop the leading game clock, lowercase, and tidy spacing.

    Football terms ("up the middle", "pushed ob", "touchdown", "short right") and
    team abbreviations / yard lines are preserved. Digits are kept in the cleaned
    string (the TF-IDF tokenizer later ignores them).
    """
    if not desc:
        return ""
    text = _LEADING_CLOCK.sub("", desc)
    text = mask_player_names(text)
    text = _NON_TEXT.sub(" ", text).lower()
    return _SPACES.sub(" ", text).strip()


# ---------------------------------------------------------------------------
# Plays -> player-game documents
# ---------------------------------------------------------------------------


def build_player_plays(pbp: pl.DataFrame) -> pl.DataFrame:
    """Regular-season plays attributed to a player by OFFICIAL id.

    * ``run``  plays -> ``rusher_player_id``   (kind ``rush``)
    * ``pass`` plays -> ``receiver_player_id`` (kind ``target``)

    Penalty-nullified (``no_play``), kneel and other plays are excluded.
    Columns: ``player_id, game_id, play_id, kind, desc``.
    """
    pbp = pbp.filter(pl.col("season_type") == "REG")
    rush = pbp.filter((pl.col("play_type") == "run") & pl.col("rusher_player_id").is_not_null()).select(
        pl.col("rusher_player_id").alias("player_id"), "game_id", "play_id", pl.lit("rush").alias("kind"), "desc"
    )
    target = pbp.filter((pl.col("play_type") == "pass") & pl.col("receiver_player_id").is_not_null()).select(
        pl.col("receiver_player_id").alias("player_id"), "game_id", "play_id", pl.lit("target").alias("kind"), "desc"
    )
    return pl.concat([rush, target]).filter(pl.col("desc").is_not_null())


def build_player_game_documents(
    plays: pl.DataFrame, valid_keys: pl.DataFrame | None = None
) -> tuple[pl.DataFrame, dict]:
    """One document per ``(player_id, game_id)``: cleaned descriptions in ``play_id`` order.

    If ``valid_keys`` (columns ``player_id, game_id``) is given, only those
    player-games are kept (e.g. the cleaned RB/WR/TE player-games); the number of
    plays dropped because their player-game is not in that set is reported.

    Returns ``(documents, info)``; documents have ``player_id, game_id, doc,
    n_rush, n_target, n_plays``.
    """
    ordered = plays.sort(["game_id", "play_id", "player_id"])
    cleaned = ordered.with_columns(
        pl.col("desc").map_elements(clean_description, return_dtype=pl.String).alias("clean")
    )
    docs = cleaned.group_by(KEYS, maintain_order=True).agg(
        pl.col("clean").filter(pl.col("clean") != "").str.join(" ").alias("doc"),
        (pl.col("kind") == "rush").sum().alias("n_rush"),
        (pl.col("kind") == "target").sum().alias("n_target"),
        pl.len().alias("n_plays"),
    )
    docs = docs.with_columns(pl.col("doc").fill_null(""))
    info = {"plays_with_official_player_id": plays.height, "player_games_with_plays_all_players": docs.height}
    if valid_keys is not None:
        kept = docs.join(valid_keys.select(KEYS).unique(), on=KEYS, how="semi")
        info["plays_not_in_cleaned_player_games"] = int(docs["n_plays"].sum() - kept["n_plays"].sum())
        docs = kept
    info["plays_matched_to_player_games"] = int(docs["n_plays"].sum())
    info["player_games_with_text"] = docs.height
    return docs, info


def window_game_ids(full: pl.DataFrame) -> pl.DataFrame:
    """For each model-ready row, the three game ids of its STRICT history window.

    ``g_m2``, ``g_m1`` are the player's two previous records in the same season and
    ``g_0`` is the current game. Only valid when ``is_model_ready`` (the existing
    strict rule: three consecutive scheduled games of the same team). The next game
    is deliberately not used.
    """
    grp = ["player_id", "season"]
    work = (
        full.select("player_id", "game_id", "season", "week", "team", "team_game_number", "is_model_ready")
        .sort(["player_id", "season", "week"])
        .with_columns(
            pl.col("game_id").shift(2).over(grp).alias("g_m2"),
            pl.col("game_id").shift(1).over(grp).alias("g_m1"),
            pl.col("game_id").alias("g_0"),
            pl.col("team_game_number").shift(2).over(grp).alias("tgn_m2"),
            pl.col("team_game_number").shift(1).over(grp).alias("tgn_m1"),
            pl.col("team").shift(2).over(grp).alias("team_m2"),
            pl.col("team").shift(1).over(grp).alias("team_m1"),
        )
        .filter(pl.col("is_model_ready"))
    )
    bad = work.filter(
        (pl.col("tgn_m1") != pl.col("team_game_number") - 1)
        | (pl.col("tgn_m2") != pl.col("team_game_number") - 2)
        | (pl.col("team_m1") != pl.col("team"))
        | (pl.col("team_m2") != pl.col("team"))
        | pl.col("g_m1").is_null()
        | pl.col("g_m2").is_null()
    )
    if bad.height:
        raise AssertionError(f"{bad.height} model-ready rows do not have a strict consecutive 3-game window")
    return work.select("player_id", "game_id", "season", "week", "g_m2", "g_m1", "g_0")


def add_window_documents(full: pl.DataFrame, docs: pl.DataFrame) -> pl.DataFrame:
    """Text of each model-ready row's strict 3-game window (current + previous two games).

    Games without any matching play contribute nothing; a row whose whole window
    has no matching play keeps ``text == ""`` (``text_empty`` True) and is NOT dropped.
    One output row per model-ready row.
    """
    ids = window_game_ids(full)
    parts = []
    for tag, col in (("m2", "g_m2"), ("m1", "g_m1"), ("0", "g_0")):
        d = docs.select(
            "player_id",
            pl.col("game_id").alias(col),
            pl.col("doc").alias(f"doc_{tag}"),
            pl.col("n_rush").alias(f"rush_{tag}"),
            pl.col("n_target").alias(f"target_{tag}"),
        )
        parts.append((tag, col, d))
    out = ids
    for _, col, d in parts:
        out = out.join(d, on=["player_id", col], how="left", maintain_order="left")
    tags = ["m2", "m1", "0"]
    docs_cols = [pl.col(f"doc_{t}").fill_null("") for t in tags]
    out = out.with_columns(
        pl.concat_str(docs_cols, separator=" ").str.replace_all(r"\s+", " ").str.strip_chars().alias("text"),
        sum(pl.col(f"rush_{t}").fill_null(0) for t in tags).alias("window_n_rush_plays"),
        sum(pl.col(f"target_{t}").fill_null(0) for t in tags).alias("window_n_target_plays"),
        sum((pl.col(f"doc_{t}").fill_null("") != "").cast(pl.Int64) for t in tags).alias("window_games_with_text"),
    ).with_columns(
        (pl.col("window_n_rush_plays") + pl.col("window_n_target_plays")).alias("window_n_plays"),
        (pl.col("text") == "").alias("text_empty"),
    )
    if out.height != ids.height:
        raise AssertionError("Window join changed the number of rows")
    return out.select(KEYS + ["season", "week"] + WINDOW_COLUMNS)


def attach_text(split_df: pl.DataFrame, windows: pl.DataFrame) -> pl.DataFrame:
    """Attach window text to a processed split, preserving rows, order and labels.

    Raises if the join would change the row count, reorder rows, or if any split
    row has no window (every model-ready split row must have one).
    """
    extra = windows.select(KEYS + WINDOW_COLUMNS)
    if extra.select(pl.struct(KEYS).is_duplicated().any()).item():
        raise AssertionError("Duplicate (player_id, game_id) keys in text windows")
    joined = split_df.join(extra, on=KEYS, how="left", maintain_order="left")
    if joined.height != split_df.height:
        raise AssertionError("attach_text changed the number of rows")
    if joined.select(KEYS).rows() != split_df.select(KEYS).rows():
        raise AssertionError("attach_text changed the row order")
    missing = joined.filter(pl.col("text").is_null()).height
    if missing:
        raise AssertionError(f"{missing} split rows have no text window")
    if "role_label" in split_df.columns and joined["role_label"].to_list() != split_df["role_label"].to_list():
        raise AssertionError("attach_text changed labels")
    return joined


# ---------------------------------------------------------------------------
# Future-text safety
# ---------------------------------------------------------------------------


def verify_no_future_text(
    full: pl.DataFrame, docs: pl.DataFrame, cutoff_weeks: tuple[int, ...] = (5, 9, 13)
) -> dict:
    """Empirical check that a row's text never depends on later games.

    Static check: no window game is the row's ``next_game_id`` and no window game is
    later than the current game. Dynamic check: for each season and cutoff week
    ``k`` every document from a game after week ``k`` is replaced with a marker; the
    windows for weeks <= ``k`` must be byte-identical and must not contain the marker.
    Raises ``AssertionError`` otherwise.
    """
    ids = window_game_ids(full)
    if "next_game_id" in full.columns:
        nxt = full.select("player_id", "game_id", "next_game_id")
        joined = ids.join(nxt, on=KEYS, how="left")
        for col in ("g_m2", "g_m1", "g_0"):
            if joined.filter(pl.col(col) == pl.col("next_game_id")).height:
                raise AssertionError(f"Window game {col} equals the next (predicted) game")
    base = add_window_documents(full, docs)
    game_week = full.select("game_id", "season", "week").unique("game_id")
    docs_w = docs.join(game_week, on="game_id", how="left")
    checks = rows = 0
    for season in sorted(full["season"].unique().to_list()):
        for k in cutoff_weeks:
            future = (pl.col("season") == season) & (pl.col("week") > k)
            marked = docs_w.with_columns(
                pl.when(future).then(pl.lit("zzfuturemarker")).otherwise(pl.col("doc")).alias("doc")
            ).drop("season", "week")
            alt = add_window_documents(full, marked)
            sel = (pl.col("season") == season) & (pl.col("week") <= k)
            a = base.filter(sel).select(KEYS + WINDOW_COLUMNS).sort(KEYS)
            b = alt.filter(sel).select(KEYS + WINDOW_COLUMNS).sort(KEYS)
            if not a.equals(b):
                raise AssertionError(f"Future text leak: windows changed (season {season}, cutoff week {k})")
            if b.filter(pl.col("text").str.contains("zzfuturemarker")).height:
                raise AssertionError(f"Future text marker found in a window (season {season}, cutoff week {k})")
            checks += 1
            rows += a.height
    return {"checks_run": checks, "rows_compared": rows, "cutoff_weeks": list(cutoff_weeks)}


# ---------------------------------------------------------------------------
# Coverage and masking audits (real data reporting)
# ---------------------------------------------------------------------------


def text_coverage_report(
    docs: pl.DataFrame, info: dict, full: pl.DataFrame, split_texts: dict[str, pl.DataFrame]
) -> dict:
    """Coverage of play text over player-games and over each split's windows."""
    pg = full.select(KEYS + ["carries", "targets"]).join(docs, on=KEYS, how="left").with_columns(
        pl.col("n_rush").fill_null(0), pl.col("n_target").fill_null(0), pl.col("n_plays").fill_null(0)
    )
    report: dict = dict(info)
    report["cleaned_player_games"] = pg.height
    report["player_games_with_text"] = pg.filter(pl.col("n_plays") > 0).height
    report["player_games_without_text"] = pg.filter(pl.col("n_plays") == 0).height
    report["carries_equal_run_plays_rate"] = float((pg["carries"] == pg["n_rush"]).mean())
    report["targets_equal_pass_plays_rate"] = float((pg["targets"] == pg["n_target"]).mean())
    splits = {}
    for name, df in split_texts.items():
        n = df.height
        empty = int(df["text_empty"].sum())
        splits[name] = {
            "rows": n,
            "empty_documents": empty,
            "empty_documents_pct": 100.0 * empty / n if n else 0.0,
            "non_empty_pct": 100.0 * (n - empty) / n if n else 0.0,
            "mean_plays_per_window": float(df["window_n_plays"].mean()) if n else 0.0,
            "median_words_per_window": float(df["text"].str.count_matches(r"\S+").median()) if n else 0.0,
            "windows_with_text_in_1_game": int((df["window_games_with_text"] == 1).sum()),
            "windows_with_text_in_2_games": int((df["window_games_with_text"] == 2).sum()),
            "windows_with_text_in_3_games": int((df["window_games_with_text"] == 3).sum()),
        }
    report["splits"] = splits
    return report


# Team / city words that legitimately remain in cleaned text and share spellings with surnames.
_PLACE_WORDS = {
    "New", "York", "Green", "Bay", "Washington", "Dallas", "Cleveland", "Houston", "Tampa", "Kansas", "City",
    "Las", "Vegas", "Los", "Angeles", "San", "Francisco", "Diego", "Denver", "Miami", "Chicago", "Detroit",
    "Minnesota", "Atlanta", "Carolina", "Arizona", "Seattle", "Buffalo", "Baltimore", "Cincinnati", "Pittsburgh",
    "Indianapolis", "Jacksonville", "Tennessee", "Philadelphia", "England", "Orleans", "Jets", "Giants",
    "Patriots", "Eagles", "Bears", "Lions", "Packers", "Vikings", "Cowboys", "Commanders", "Falcons",
    "Panthers", "Saints", "Buccaneers", "Cardinals", "Rams", "Seahawks", "Niners", "Bills", "Dolphins",
    "Ravens", "Bengals", "Browns", "Steelers", "Texans", "Colts", "Jaguars", "Titans", "Broncos", "Chiefs",
    "Raiders", "Chargers",
}


def name_masking_audit(
    descs: list[str],
    player_ids: list[str] | None = None,
    names_by_id: dict[str, str] | None = None,
    league_names: list[str] | None = None,
) -> dict:
    """Measure how many recognizable player-name patterns survive ``clean_description``'s masking.

    Works on the masked (but not lowercased) text and reports the share of descriptions with:
      * a leftover jersey token (``21-X``),
      * a leftover ``Initial.Surname`` pattern (no space, e.g. ``S.Brown``),
      * a leftover ``#10 Surname`` pattern,
      * the involved player's own surname (from official ids -> display names),
      * any league surname (from ``league_names``, excluding team/city words).
    """
    jersey = re.compile(r"\b\d{1,2}-[A-Za-z]")
    initial = re.compile(r"\b[A-Za-z]{1,4}\.[A-Z][a-z]")
    hashnum = re.compile(r"#\d{1,2}\s+[A-Z]")
    token = re.compile(r"\b[A-Z][a-z'\-]+\b")
    suffixes = {"Jr", "Sr", "II", "III", "IV"}

    def surname(full: str | None) -> str | None:
        if not full:
            return None
        words = full.replace(".", "").split()
        if not words:
            return None
        return words[-2] if (words[-1] in suffixes and len(words) > 1) else words[-1]

    league = set()
    for n in league_names or []:
        s = surname(n)
        if s and len(s) > 3 and s not in _PLACE_WORDS:
            league.add(s)

    n = len(descs)
    counts = Counter()
    leftovers: Counter = Counter()
    examples: list[str] = []
    for i, desc in enumerate(descs):
        masked = mask_player_names(_LEADING_CLOCK.sub("", desc))
        j, ini, h = bool(jersey.search(masked)), bool(initial.search(masked)), bool(hashnum.search(masked))
        own = False
        if player_ids is not None and names_by_id is not None:
            s = surname(names_by_id.get(player_ids[i]))
            own = bool(s and len(s) > 2 and re.search(r"\b" + re.escape(s) + r"\b", masked, flags=re.I))
        hits = [t for t in token.findall(masked) if t in league]
        counts["leftover_jersey_token"] += j
        counts["leftover_initial_surname"] += ini
        counts["leftover_hash_number_name"] += h
        counts["involved_player_surname_survives"] += own
        counts["any_league_surname_token"] += bool(hits)
        leftovers.update(hits)
        if (j or ini or h or own or hits) and len(examples) < 5:
            examples.append(desc)
    result = {"descriptions_audited": n}
    for key in (
        "leftover_jersey_token",
        "leftover_initial_surname",
        "leftover_hash_number_name",
        "involved_player_surname_survives",
        "any_league_surname_token",
    ):
        result[key] = {"count": int(counts[key]), "pct": 100.0 * counts[key] / n if n else 0.0}
    result["top_leftover_surname_tokens"] = leftovers.most_common(10)
    result["example_descriptions_with_residuals"] = examples
    return result


# ---------------------------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------------------------

TFIDF_PARAMS = {
    "ngram_range": (1, 2),
    "min_df": 5,
    "max_features": 5000,
    "sublinear_tf": True,
    "lowercase": True,
    "token_pattern": r"(?u)\b[a-z][a-z']+\b",  # alphabetic tokens only: numeric yardage is dropped
    "stop_words": None,  # keep words like "up" / "out": "up the middle", "ran ob"
    "dtype": "float64",
}


def build_tfidf_vectorizer(**overrides) -> TfidfVectorizer:
    """TF-IDF with the fixed project settings (fit on TRAINING text only by the caller)."""
    params = {**TFIDF_PARAMS, **overrides}
    import numpy as np

    params["dtype"] = getattr(np, params["dtype"]) if isinstance(params["dtype"], str) else params["dtype"]
    return TfidfVectorizer(**params)

"""Offline tests for play-by-play text features. Synthetic fixtures only (see tests/synthetic.py)."""

from __future__ import annotations

import polars as pl
import pytest

import src.text_features as tf
from src.preprocessing import prepare_dataset
from src.text_features import (
    add_window_documents,
    attach_text,
    build_player_game_documents,
    build_player_plays,
    build_tfidf_vectorizer,
    clean_description,
    name_masking_audit,
    text_coverage_report,
    verify_no_future_text,
    window_game_ids,
)
from tests.synthetic import game_id, make_pbp, make_schedule, make_stats, marker_docs


def full_table(weeks_played, schedule_weeks=None, player="p", opp=5):
    schedule_weeks = schedule_weeks or list(range(1, 9))
    stats = make_stats(player, {w: (opp, 0) for w in weeks_played})
    full, _, _ = prepare_dataset(stats, make_schedule(schedule_weeks), verify_leakage=False)
    return full


# ------------------------------------------------------------ player-game joins


def test_plays_are_attributed_by_official_id_not_by_name_in_text():
    pbp = make_pbp(
        [
            # rushed by A; description mentions B's name
            {"game_id": "g1", "play_id": 1, "rusher_player_id": "A", "desc": "(1:00) 30-B.Other up the middle for 2 yards"},
            # B is the receiver; description mentions A's name in the text
            {"game_id": "g1", "play_id": 2, "play_type": "pass", "receiver_player_id": "B", "desc": "(2:00) 1-Q.Back pass short right to 10-A.Name"},
        ]
    )
    plays = build_player_plays(pbp)
    by_player = {r["player_id"]: r for r in plays.to_dicts()}
    assert set(by_player) == {"A", "B"}
    # A is credited with the play where A is the official rusher, even though only B's name is in that text
    assert by_player["A"]["kind"] == "rush" and by_player["A"]["play_id"] == 1.0
    assert "B.Other" in by_player["A"]["desc"]
    # B is credited with the play where B is the official receiver, even though A's name is in that text
    assert by_player["B"]["kind"] == "target" and by_player["B"]["play_id"] == 2.0
    assert plays.height == 2


def test_excluded_play_types_and_post_season():
    pbp = make_pbp(
        [
            {"game_id": "g1", "play_id": 1, "rusher_player_id": "A"},
            {"game_id": "g1", "play_id": 2, "play_type": "no_play", "rusher_player_id": "A"},
            {"game_id": "g1", "play_id": 3, "play_type": "qb_kneel", "rusher_player_id": "A"},
            {"game_id": "g1", "play_id": 4, "play_type": "run", "rusher_player_id": None},
            {"game_id": "g1", "play_id": 5, "season_type": "POST", "rusher_player_id": "A"},
        ]
    )
    assert build_player_plays(pbp)["play_id"].to_list() == [1.0]


def test_documents_are_chronological_and_join_on_player_and_game():
    pbp = make_pbp(
        [
            {"game_id": "g1", "play_id": 30, "rusher_player_id": "A", "desc": "(1:00) 22-A.Back third play"},
            {"game_id": "g1", "play_id": 10, "rusher_player_id": "A", "desc": "(14:00) 22-A.Back first play"},
            {"game_id": "g1", "play_id": 20, "play_type": "pass", "receiver_player_id": "A", "desc": "(8:00) 1-Q.Qb second play to 22-A.Back"},
            {"game_id": "g2", "play_id": 5, "rusher_player_id": "A", "desc": "(14:00) 22-A.Back other game"},
            {"game_id": "g1", "play_id": 11, "rusher_player_id": "B", "desc": "(13:00) 33-B.Other b play"},
        ]
    ).sample(fraction=1.0, shuffle=True, seed=1)
    docs, info = build_player_game_documents(build_player_plays(pbp))
    d = {(r["player_id"], r["game_id"]): r for r in docs.to_dicts()}
    words = d[("A", "g1")]["doc"].split()
    assert words.index("first") < words.index("second") < words.index("third")
    assert d[("A", "g1")]["n_rush"] == 2 and d[("A", "g1")]["n_target"] == 1
    assert "other" in d[("A", "g2")]["doc"] and "first" not in d[("A", "g2")]["doc"]
    assert "b" in d[("B", "g1")]["doc"].split()


def test_valid_keys_filter_reports_unmatched_plays():
    pbp = make_pbp(
        [
            {"game_id": "g1", "play_id": 1, "rusher_player_id": "A"},
            {"game_id": "g1", "play_id": 2, "rusher_player_id": "QB"},  # not an RB/WR/TE player-game
        ]
    )
    keys = pl.DataFrame({"player_id": ["A"], "game_id": ["g1"]})
    docs, info = build_player_game_documents(build_player_plays(pbp), valid_keys=keys)
    assert docs["player_id"].to_list() == ["A"]
    assert info["plays_not_in_cleaned_player_games"] == 1
    assert info["plays_matched_to_player_games"] == 1


# --------------------------------------------------------------- strict windows


def test_window_contains_exactly_current_and_previous_two_games_in_order():
    full = full_table(range(1, 9))
    docs = marker_docs("p", list(range(1, 9)))
    w = add_window_documents(full, docs).filter(pl.col("week") == 5).to_dicts()[0]
    words = w["text"].split()
    for included in ("markerw3", "markerw4", "markerw5"):
        assert included in words
    for excluded in ("markerw1", "markerw2", "markerw6", "markerw7", "markerw8"):
        assert excluded not in words
    assert words.index("markerw3") < words.index("markerw4") < words.index("markerw5")
    assert w["window_games_with_text"] == 3


def test_window_never_includes_the_predicted_next_game():
    full = full_table(range(1, 9))
    docs = marker_docs("p", list(range(1, 9)))
    windows = add_window_documents(full, docs)
    nxt = full.select("player_id", "game_id", "next_game_id", "week")
    for r in windows.join(nxt, on=["player_id", "game_id"]).to_dicts():
        if r["next_game_id"] is not None:
            next_week = int(r["next_game_id"].split("_")[1])
            assert f"markerw{next_week}" not in r["text"].split()


def test_window_skips_bye_week_using_team_schedule():
    full = full_table([1, 2, 3, 4, 6, 7], schedule_weeks=[1, 2, 3, 4, 6, 7])  # bye in week 5
    docs = marker_docs("p", [1, 2, 3, 4, 6, 7])
    w = add_window_documents(full, docs).filter(pl.col("week") == 6)["text"][0].split()
    assert {"markerw3", "markerw4", "markerw6"} <= set(w)
    assert "markerw2" not in w


def test_row_with_missed_team_game_has_no_window():
    full = full_table([1, 2, 3, 5, 6, 7], schedule_weeks=list(range(1, 9)))  # missed week 4
    docs = marker_docs("p", [1, 2, 3, 5, 6, 7])
    windows = add_window_documents(full, docs)
    weeks = sorted(windows["week"].to_list())
    assert weeks == [3, 7]  # weeks 5 and 6 span the missed game, so they are not model-ready
    assert "markerw5" in windows.filter(pl.col("week") == 7)["text"][0].split()
    assert "markerw3" not in windows.filter(pl.col("week") == 7)["text"][0].split()


def test_window_row_count_equals_model_ready_rows():
    full = full_table(range(1, 9))
    windows = add_window_documents(full, marker_docs("p", list(range(1, 9))))
    assert windows.height == full.filter(pl.col("is_model_ready")).height
    assert window_game_ids(full).height == windows.height


# ------------------------------------------------------------------ missingness


def test_empty_documents_are_preserved_not_dropped():
    full = full_table(range(1, 7))
    no_docs = marker_docs("p", [1]).clear()  # player never matched any play
    windows = add_window_documents(full, no_docs)
    assert windows.height == full.filter(pl.col("is_model_ready")).height
    assert windows["text"].to_list() == [""] * windows.height
    assert windows["text_empty"].all()
    assert windows["window_n_plays"].to_list() == [0] * windows.height


def test_partially_empty_window_counts_games_with_text():
    full = full_table(range(1, 7))
    docs = marker_docs("p", [2])  # only week 2 has text
    w = add_window_documents(full, docs).filter(pl.col("week") == 4).to_dicts()[0]  # weeks 2,3,4
    assert w["text"] == "up the middle markerw2"
    assert w["window_games_with_text"] == 1 and w["text_empty"] is False
    w5 = add_window_documents(full, docs).filter(pl.col("week") == 5).to_dicts()[0]  # weeks 3,4,5
    assert w5["text_empty"] is True


def test_coverage_report_counts_empty_documents():
    full = full_table(range(1, 7))
    docs = marker_docs("p", [2])
    windows = add_window_documents(full, docs)
    report = text_coverage_report(docs, {"plays_matched_to_player_games": 1}, full, {"train": windows})
    s = report["splits"]["train"]
    assert s["rows"] == windows.height
    assert s["empty_documents"] == int(windows["text_empty"].sum())
    assert s["empty_documents"] + round(s["non_empty_pct"] * s["rows"] / 100) == s["rows"]
    assert report["player_games_with_text"] == 1


# ------------------------------------------------------------ future-text safety


def test_changing_future_game_text_never_changes_earlier_windows():
    full = full_table(range(1, 9))
    docs = marker_docs("p", list(range(1, 9)))
    base = add_window_documents(full, docs)
    changed_docs = docs.with_columns(
        pl.when(pl.col("game_id").is_in([game_id(7), game_id(8)])).then(pl.lit("leaked words")).otherwise(pl.col("doc")).alias("doc")
    )
    changed = add_window_documents(full, changed_docs)
    for wk in (3, 4, 5):
        assert base.filter(pl.col("week") == wk)["text"][0] == changed.filter(pl.col("week") == wk)["text"][0]
    # sanity: the week-8 window does change (so the test could detect leakage)
    assert base.filter(pl.col("week") == 8)["text"][0] != changed.filter(pl.col("week") == 8)["text"][0]


def test_verify_no_future_text_passes_for_correct_windows():
    full = full_table(range(1, 9))
    summary = verify_no_future_text(full, marker_docs("p", list(range(1, 9))), cutoff_weeks=(4, 6))
    assert summary["checks_run"] == 2


def test_verify_no_future_text_detects_a_leaky_window(monkeypatch):
    full = full_table(range(1, 9))
    docs = marker_docs("p", list(range(1, 9)))
    original = tf.add_window_documents

    def leaky(full_, docs_):
        out = original(full_, docs_)
        base_cols = out.columns
        nxt = full_.select("player_id", "game_id", "next_game_id")
        nxt_doc = docs_.select("player_id", pl.col("game_id").alias("next_game_id"), pl.col("doc").alias("nd"))
        leaky_out = out.join(nxt, on=["player_id", "game_id"], how="left").join(
            nxt_doc, on=["player_id", "next_game_id"], how="left"
        )
        # deliberately append the NEXT game's text to the window
        return leaky_out.with_columns(
            (pl.col("text") + " " + pl.col("nd").fill_null("")).alias("text")
        ).select(base_cols)

    monkeypatch.setattr(tf, "add_window_documents", leaky)
    with pytest.raises(AssertionError):
        verify_no_future_text(full, docs, cutoff_weeks=(4,))


# -------------------------------------------------------------------- alignment


def _split_from_full(full):
    return full.filter(pl.col("is_model_ready")).with_columns(pl.lit("Stable").alias("role_label"))


def test_attach_text_preserves_rows_order_and_labels():
    full = full_table(range(1, 9))
    windows = add_window_documents(full, marker_docs("p", list(range(1, 9))))
    split = _split_from_full(full).sort("week", descending=True)  # deliberately not in default order
    out = attach_text(split, windows)
    assert out.height == split.height
    assert out["week"].to_list() == split["week"].to_list()
    assert out["role_label"].to_list() == split["role_label"].to_list()
    assert out.filter(pl.col("week") == 5)["text"][0].count("markerw") == 3


def test_attach_text_raises_when_a_split_row_has_no_window():
    full = full_table(range(1, 9))
    windows = add_window_documents(full, marker_docs("p", list(range(1, 9)))).filter(pl.col("week") != 5)
    with pytest.raises(AssertionError, match="no text window"):
        attach_text(_split_from_full(full), windows)


def test_attach_text_raises_on_duplicate_window_keys():
    full = full_table(range(1, 9))
    windows = add_window_documents(full, marker_docs("p", list(range(1, 9))))
    with pytest.raises(AssertionError, match="Duplicate"):
        attach_text(_split_from_full(full), pl.concat([windows, windows.head(1)]))


# --------------------------------------------------------------- name masking


@pytest.mark.parametrize(
    "raw,gone",
    [
        ("(11:52) 83-G.Dortch up the middle to SF 49 for 5 yards (57-D.Greenlaw, 20-A.Thomas).", ["dortch", "greenlaw", "thomas", "83", "57"]),
        ("(15:00) 20-Bre.Hall left tackle pushed ob at LV 14 for 11 yards (1-M.Epps; 21-A.Robertson).", ["hall", "epps", "robertson", "bre"]),
        ("(10:05) 16-J.Goff pass deep right to 14-A.St. Brown pushed ob at LV 16 for 24 yards.", ["goff", "brown", "st"]),
        ("(11:20) 26-S.Barkley right tackle to DAL 45 for 6 yards (99-C.Golston; 55-L.Vander Esch).", ["barkley", "golston", "esch", "vander"]),
        ("(:02) 7-D.Penny Jr. up the middle for 2 yards (LV-21-A.Robertson).", ["penny", "robertson"]),
        ("(4:36) (Shotgun) 6-J.Driskel right end to IND 19 for -1 yards (99-D.Buckner). #6 Driskel in at QB for HOU.", ["driskel", "buckner"]),
        ("(4:44) T.Hill at quarterback. 41-A.Kamara right guard to NO 40. Coverage by 34-Edmunds.", ["hill", "kamara", "edmunds"]),
        ("(:19) 20-T.Pollard up the middle. Injury Update: Jayron Kearse has a knee injury; his return is questionable", ["pollard", "jayron", "kearse"]),
        ("(1:00) 99-J.J.Watt sacks 7-B.Roethlisberger for -7 yards.", ["watt", "roethlisberger"]),
        ("(2:00) 11-A.J.Brown pass deep left to 84-K.Allison.", ["brown", "allison"]),
        ("(3:10) S.Brown up the middle for 3 yards (T.Hill).", ["brown", "hill"]),
        ("(3:30) #6 Driskel up the middle for 3 yards.", ["driskel"]),
        ("(3:40) 88-C.Beasley pass to 11-J.Smith-Schuster (21-P.Surtain II).", ["beasley", "smith", "schuster", "surtain"]),
    ],
)
def test_player_names_and_jersey_numbers_are_masked(raw, gone):
    cleaned = clean_description(raw)
    tokens = set(cleaned.replace("-", " ").split())
    for g in gone:
        assert g not in tokens, f"{g!r} survived in {cleaned!r}"


def test_football_terms_teams_and_yard_lines_are_preserved():
    raw = "(13:54) (Shotgun) 1-T.Tagovailoa pass short right to 17-J.Waddle pushed ob at NYJ 25 for 27 yards, TOUCHDOWN. up the middle"
    cleaned = clean_description(raw)
    for term in ("shotgun", "pass short right", "pushed ob", "nyj 25", "touchdown", "up the middle", "27 yards"):
        assert term in cleaned
    assert not cleaned.startswith("13") and "(" not in cleaned  # clock and brackets are dropped


def test_masking_does_not_swallow_following_football_words_or_locations():
    # a short surname ending a sentence must not consume the next word (second-initial pattern guard)
    cleaned = clean_description("(5:00) 4-D.Hall. Penalty on 70-J.Doe, false start. Green Bay Packers challenged at KC 30.")
    assert "hall" not in cleaned.split() and "doe" not in cleaned.split()
    for kept in ("penalty", "false start", "green bay packers", "challenged", "kc 30"):
        assert kept in cleaned
    cleaned = clean_description("(2:10) 17-J.Allen scrambles left end to BUF 49 for 11 yards (PHI-97-J.Hargrave); LV-78-K.Miller holding.")
    for kept in ("scrambles", "left end", "buf 49", "11 yards", "holding"):
        assert kept in cleaned
    assert "phi" not in cleaned.split() and "lv" not in cleaned.split()  # team tag inside a player token goes with the token


def test_clean_description_handles_missing_values():
    assert clean_description(None) == "" and clean_description("") == ""


def test_name_masking_audit_reports_residuals_on_synthetic_text():
    descs = ["(1:00) 22-A.Back up the middle for 3 yards (50-C.Def).", "(2:00) 22-A.Back up the middle. Walker was injured"]
    audit = name_masking_audit(descs, ["x", "x"], {"x": "Ann Walker"}, ["Ann Walker"])
    assert audit["descriptions_audited"] == 2
    assert audit["leftover_jersey_token"]["count"] == 0
    assert audit["involved_player_surname_survives"]["count"] == 1  # "Walker" in the second text


# ------------------------------------------------------------------- TF-IDF


def test_tfidf_ignores_numeric_yardage_and_keeps_football_words():
    vec = build_tfidf_vectorizer(min_df=1)
    vec.fit(["up the middle for 12 yards", "pass short right for 5 yards touchdown", "pushed ob for 27 yards"])
    vocab = set(vec.vocabulary_)
    assert not any(any(ch.isdigit() for ch in t) for t in vocab)  # numeric yardage is removed
    assert {"yards", "touchdown", "pushed ob", "up the", "the middle"} <= vocab  # 1-2 grams
    assert "up" in vocab  # no stop-word removal

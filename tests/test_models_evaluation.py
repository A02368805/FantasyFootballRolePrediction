"""Offline tests for baseline models and evaluation metrics.

All data is synthetic (random numbers with a fixed seed) and exists only to test
logic; it is never used for research results.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from scipy import sparse

import src.data_loader as data_loader
import nflreadpy
from src.evaluation import LABELS, compute_metrics, select_best
from src.feature_engineering import AUDIT_COLUMNS, FEATURE_COLUMNS
from src.models import (
    STAT_FEATURES,
    build_pipeline,
    n_tfidf_features,
    to_model_frame,
    tune_on_validation,
)

VOCAB = {
    "Decreasing": "tackle short slant",
    "Stable": "pass middle screen",
    "Increasing": "guard sweep toss",
}


def synthetic_split(n: int, seed: int, extra_text: str = "", extra_every: int = 0) -> pl.DataFrame:
    """Random labeled rows: stats loosely related to the label, text built from label words."""
    rng = np.random.default_rng(seed)
    labels = rng.choice(LABELS, size=n, p=[0.2, 0.6, 0.2])
    data = {c: rng.normal(size=n) for c in FEATURE_COLUMNS}
    shift = {"Decreasing": -1.0, "Stable": 0.0, "Increasing": 1.0}
    data["avg3_opportunities"] = np.array([shift[l] for l in labels]) + rng.normal(scale=0.7, size=n)
    texts = []
    for i, lab in enumerate(labels):
        words = VOCAB[lab].split()
        t = " ".join(rng.choice(words, size=6)) + " " + " ".join(rng.choice(["run", "pass", "right", "left"], size=4))
        if extra_every and i % extra_every == 0:
            t += " " + extra_text
        texts.append(t)
    return pl.DataFrame({**data, "text": texts, "role_label": labels})


@pytest.fixture(scope="module")
def train_df():
    return synthetic_split(200, seed=0, extra_text="trainonlyterm", extra_every=10)


@pytest.fixture(scope="module")
def valid_df():
    return synthetic_split(80, seed=1, extra_text="validonlyterm", extra_every=2)


# ----------------------------------------------------------------------- metrics


def test_metrics_match_hand_calculation():
    y_true = ["Decreasing"] * 3 + ["Stable"] * 2 + ["Increasing"]
    y_pred = ["Decreasing", "Decreasing", "Stable", "Stable", "Increasing", "Increasing"]
    m = compute_metrics(y_true, y_pred)
    assert m["accuracy"] == pytest.approx(4 / 6)
    d, s, i = (m["per_class"][k] for k in LABELS)
    assert (d["precision"], d["recall"], d["f1"]) == pytest.approx((1.0, 2 / 3, 0.8))
    assert (s["precision"], s["recall"], s["f1"]) == pytest.approx((0.5, 0.5, 0.5))
    assert (i["precision"], i["recall"], i["f1"]) == pytest.approx((0.5, 1.0, 2 / 3))
    assert m["macro_precision"] == pytest.approx((1.0 + 0.5 + 0.5) / 3)
    assert m["macro_recall"] == pytest.approx((2 / 3 + 0.5 + 1.0) / 3)
    assert m["macro_f1"] == pytest.approx((0.8 + 0.5 + 2 / 3) / 3)
    assert m["confusion_matrix"] == [[2, 1, 0], [0, 1, 1], [0, 0, 1]]
    assert [d["support"], s["support"], i["support"]] == [3, 2, 1]
    assert m["predicted_counts"] == {"Decreasing": 2, "Stable": 2, "Increasing": 2}


def test_metrics_for_a_never_predicted_class_use_zero_division():
    y_true = ["Decreasing", "Stable", "Stable", "Increasing"]
    m = compute_metrics(y_true, ["Stable"] * 4)
    assert m["per_class"]["Decreasing"]["precision"] == 0.0
    assert m["per_class"]["Stable"]["recall"] == 1.0
    assert m["per_class"]["Stable"]["precision"] == pytest.approx(0.5)
    assert m["macro_f1"] == pytest.approx((0 + 2 * 0.5 * 1.0 / 1.5 + 0) / 3)


def test_select_best_uses_macro_f1_then_deterministic_tie_break():
    grid = [
        {"C": 1.0, "class_weight": None, "macro_f1": 0.40},
        {"C": 0.1, "class_weight": "balanced", "macro_f1": 0.45},
        {"C": 0.01, "class_weight": "balanced", "macro_f1": 0.45},
        {"C": 0.01, "class_weight": None, "macro_f1": 0.45},
        {"C": 10.0, "class_weight": None, "macro_f1": 0.30},
    ]
    best = select_best(grid)
    assert best["macro_f1"] == 0.45 and best["C"] == 0.01 and best["class_weight"] is None


# ----------------------------------------------------------------- model inputs


def test_model_frame_contains_only_predictive_inputs(train_df):
    df = train_df.with_columns(
        pl.lit("00-0000001").alias("player_id"),
        pl.lit("g").alias("game_id"),
        pl.lit(7.0).alias("next_game_opportunities"),
        pl.lit(1.0).alias("role_change"),
    )
    X = to_model_frame(df)
    assert list(X.columns) == STAT_FEATURES + ["text"]
    assert len(STAT_FEATURES) == 11
    for forbidden in AUDIT_COLUMNS + ["player_id", "game_id", "role_label", "role_label_id"]:
        assert forbidden not in X.columns


# ------------------------------------------------------ fit on training data only


def test_tfidf_vocabulary_is_fit_on_training_text_only(train_df, valid_df):
    pipe = build_pipeline("text", C=1.0).fit(to_model_frame(train_df), train_df["role_label"].to_numpy())
    vocab = pipe.named_steps["prep"].named_transformers_["tfidf"].vocabulary_
    assert "trainonlyterm" in vocab  # appears in 20 training documents (min_df=5)
    assert "validonlyterm" not in vocab  # appears only in validation documents
    # transforming validation does not change the fitted vocabulary / idf
    idf_before = pipe.named_steps["prep"].named_transformers_["tfidf"].idf_.copy()
    pipe.predict(to_model_frame(valid_df))
    vec = pipe.named_steps["prep"].named_transformers_["tfidf"]
    assert np.array_equal(vec.idf_, idf_before) and "validonlyterm" not in vec.vocabulary_


def test_fitted_text_model_is_independent_of_validation_data(train_df, valid_df):
    X_tr, y_tr = to_model_frame(train_df), train_df["role_label"].to_numpy()
    a = build_pipeline("combined", C=1.0).fit(X_tr, y_tr)
    other_valid = synthetic_split(80, seed=99, extra_text="different words entirely", extra_every=1)
    a.predict(to_model_frame(other_valid))
    b = build_pipeline("combined", C=1.0).fit(X_tr, y_tr)
    a.predict(to_model_frame(valid_df))
    assert np.allclose(a.named_steps["clf"].coef_, b.named_steps["clf"].coef_)


def test_imputer_and_scaler_statistics_come_from_training_rows_only(train_df, valid_df):
    train = train_df.with_columns(pl.when(pl.arange(0, pl.len()) % 7 == 0).then(None).otherwise(pl.col("target_share")).alias("target_share"))
    pipe = build_pipeline("stats", C=1.0).fit(to_model_frame(train), train["role_label"].to_numpy())
    block = pipe.named_steps["prep"].named_transformers_["stats"]
    train_median = float(train["target_share"].drop_nulls().median())
    idx = STAT_FEATURES.index("target_share")
    assert block.named_steps["impute"].statistics_[idx] == pytest.approx(train_median)
    imputed = block.named_steps["impute"].transform(to_model_frame(train)[STAT_FEATURES])
    assert block.named_steps["scale"].mean_[idx] == pytest.approx(imputed[:, idx].mean())
    pipe.predict(to_model_frame(valid_df))  # transforming validation leaves the statistics unchanged
    assert block.named_steps["impute"].statistics_[idx] == pytest.approx(train_median)


@pytest.mark.filterwarnings("ignore:Skipping features without any observed values")
def test_missing_stat_values_are_handled_by_the_pipeline(train_df):
    X = to_model_frame(train_df.with_columns(pl.lit(None, dtype=pl.Float64).alias("target_share")))
    pipe = build_pipeline("stats").fit(X.iloc[:150], train_df["role_label"].to_numpy()[:150])
    assert len(pipe.predict(X.iloc[150:])) == 50


# ------------------------------------------------------------- sparse features


@pytest.mark.parametrize("kind", ["text", "combined"])
def test_text_features_stay_sparse(kind, train_df):
    X, y = to_model_frame(train_df), train_df["role_label"].to_numpy()
    pipe = build_pipeline(kind).fit(X, y)
    out = pipe.named_steps["prep"].transform(X)
    assert sparse.issparse(out)
    assert out.shape[1] == (n_tfidf_features(pipe) + (len(STAT_FEATURES) if kind == "combined" else 0))


# ------------------------------------------------------------ dummy / reference


def test_dummy_predicts_the_training_majority_class(train_df, valid_df):
    y = train_df["role_label"].to_numpy()
    majority = max(set(y), key=list(y).count)
    pipe = build_pipeline("dummy").fit(to_model_frame(train_df), y)
    assert set(pipe.predict(to_model_frame(valid_df))) == {majority}


# ---------------------------------------------------- alignment, reproducibility


def test_train_and_validation_frames_align_with_their_rows(train_df, valid_df):
    for df in (train_df, valid_df):
        X = to_model_frame(df)
        assert len(X) == df.height
        assert X["text"].tolist() == df["text"].to_list()
        assert np.allclose(X["avg3_opportunities"].to_numpy(), df["avg3_opportunities"].to_numpy())


def test_tuning_is_reproducible_with_fixed_seed(train_df, valid_df):
    results = []
    for _ in range(2):
        pipe, best, grid = tune_on_validation("combined", train_df, valid_df)
        results.append((pipe.predict(to_model_frame(valid_df)), pipe.named_steps["clf"].coef_, best, grid))
    assert np.array_equal(results[0][0], results[1][0])
    assert np.allclose(results[0][1], results[1][1])
    assert results[0][2] == results[1][2] and results[0][3] == results[1][3]


def test_tuning_grid_is_scored_on_validation_only(train_df, valid_df):
    pipe, best, grid = tune_on_validation("stats", train_df, valid_df)
    assert len(grid) == 8  # 4 C values x 2 class_weight options
    y_valid = valid_df["role_label"].to_numpy()
    m = compute_metrics(y_valid, pipe.predict(to_model_frame(valid_df)))
    assert m["macro_f1"] == pytest.approx(best["macro_f1"])
    assert best["macro_f1"] == max(e["macro_f1"] for e in grid)


def test_unknown_model_kind_raises():
    with pytest.raises(ValueError):
        build_pipeline("forest")


# ----------------------------------------------------------- pbp loader (mocked)


def _fake_pbp(season: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_id": ["g1", "g1", "g2"],
            "play_id": [1.0, 2.0, 3.0],
            "season": [season] * 3,
            "week": [1, 1, 19],
            "season_type": ["REG", "REG", "POST"],
            "play_type": ["run", "pass", "run"],
            "desc": ["a", "b", "c"],
            "rusher_player_id": ["x", None, "x"],
            "receiver_player_id": [None, "y", None],
            "epa": [0.1, 0.2, 0.3],  # extra column that must NOT be cached
        }
    )


def test_pbp_loader_keeps_slim_reg_columns_and_caches(monkeypatch, tmp_path):
    calls = []

    def fake_load_pbp(seasons):
        calls.append(list(seasons))
        return _fake_pbp(seasons[0])

    monkeypatch.setattr(nflreadpy, "load_pbp", fake_load_pbp)
    first = data_loader.load_pbp_text_cached([2030, 2031], raw_dir=tmp_path)
    assert calls == [[2030], [2031]]  # one season at a time
    assert first.columns == data_loader.PBP_COLUMNS
    assert first.height == 4 and set(first["season_type"]) == {"REG"}
    assert (tmp_path / "pbp_text_2030.parquet").exists()

    monkeypatch.setattr(nflreadpy, "load_pbp", lambda s: (_ for _ in ()).throw(AssertionError("cache miss")))
    assert data_loader.load_pbp_text_cached([2030, 2031], raw_dir=tmp_path).height == 4


def test_pbp_loader_clear_error_on_failed_download(monkeypatch, tmp_path):
    monkeypatch.setattr(nflreadpy, "load_pbp", lambda s: (_ for _ in ()).throw(ConnectionError("offline")))
    with pytest.raises(data_loader.DataLoadError, match="Could not download play-by-play"):
        data_loader.load_pbp_text_cached([2030], raw_dir=tmp_path)


# ------------------------------------------------------------ paired bootstrap


def test_paired_bootstrap_matches_point_difference_and_is_reproducible():
    from src.evaluation import paired_bootstrap_macro_f1

    rng = np.random.default_rng(3)
    y = rng.choice(LABELS, size=300, p=[0.2, 0.6, 0.2])
    good = np.where(rng.random(300) < 0.8, y, rng.choice(LABELS, size=300))
    bad = np.where(rng.random(300) < 0.3, y, rng.choice(LABELS, size=300))
    r1 = paired_bootstrap_macro_f1(y, good, bad, n_boot=300, seed=7)
    r2 = paired_bootstrap_macro_f1(y, good, bad, n_boot=300, seed=7)
    assert r1 == r2
    expected = compute_metrics(y, good)["macro_f1"] - compute_metrics(y, bad)["macro_f1"]
    assert r1["point_difference"] == pytest.approx(expected)
    assert r1["ci95_low"] > 0 and r1["share_resamples_a_better"] > 0.95


def test_paired_bootstrap_identical_predictions_have_zero_difference():
    from src.evaluation import paired_bootstrap_macro_f1

    y = np.array(["Stable", "Increasing", "Decreasing", "Stable"] * 20)
    r = paired_bootstrap_macro_f1(y, y, y, n_boot=50)
    assert r["point_difference"] == 0 and r["ci95_low"] == 0 and r["ci95_high"] == 0

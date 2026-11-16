"""Classical ML baselines: dummy, statistics-only, text-only and combined.

All four are scikit-learn pipelines. Imputation, scaling and TF-IDF are fit on the
TRAINING rows only (the whole pipeline is fit on train; validation/test are only
transformed). TF-IDF features stay sparse end to end (``sparse_threshold=1.0``,
no ``.toarray()``).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation import compute_metrics, select_best
from src.feature_engineering import FEATURE_COLUMNS, assert_feature_columns_safe
from src.text_features import TEXT_COLUMN, build_tfidf_vectorizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

RANDOM_SEED = 42
STAT_FEATURES = list(FEATURE_COLUMNS)  # the existing 11 statistical features
C_GRID = [0.01, 0.1, 1.0, 10.0]
CLASS_WEIGHT_GRID = [None, "balanced"]
MAX_ITER = 2000

MODEL_KINDS = ["dummy", "stats", "text", "combined"]
MODEL_NAMES = {
    "dummy": "Majority-class dummy",
    "stats": "Statistics-only LR",
    "text": "Text-only LR (TF-IDF)",
    "combined": "Combined LR (stats+TF-IDF)",
}


def to_model_frame(df: pl.DataFrame) -> pd.DataFrame:
    """Predictive inputs only: the 11 statistical features plus the ``text`` window.

    Identifiers, labels, audit and future-derived columns are never included.
    """
    assert_feature_columns_safe()
    data = {c: df[c].cast(pl.Float64).to_numpy() for c in STAT_FEATURES}
    data[TEXT_COLUMN] = np.asarray(df[TEXT_COLUMN].to_list(), dtype=object)
    return pd.DataFrame(data)


def _stats_block() -> Pipeline:
    return Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )


def _classifier(C: float, class_weight) -> LogisticRegression:
    return LogisticRegression(
        C=C, class_weight=class_weight, max_iter=MAX_ITER, solver="lbfgs", random_state=RANDOM_SEED
    )


def build_pipeline(kind: str, C: float = 1.0, class_weight=None) -> Pipeline:
    """Build one baseline pipeline (unfitted)."""
    if kind == "dummy":
        return Pipeline([("clf", DummyClassifier(strategy="most_frequent"))])
    if kind == "stats":
        prep = ColumnTransformer([("stats", _stats_block(), STAT_FEATURES)], sparse_threshold=1.0)
    elif kind == "text":
        prep = ColumnTransformer([("tfidf", build_tfidf_vectorizer(), TEXT_COLUMN)], sparse_threshold=1.0)
    elif kind == "combined":
        prep = ColumnTransformer(
            [("stats", _stats_block(), STAT_FEATURES), ("tfidf", build_tfidf_vectorizer(), TEXT_COLUMN)],
            sparse_threshold=1.0,
        )
    else:
        raise ValueError(f"Unknown model kind {kind!r}; expected one of {MODEL_KINDS}")
    return Pipeline([("prep", prep), ("clf", _classifier(C, class_weight))])


def tune_on_validation(
    kind: str, train: pl.DataFrame, valid: pl.DataFrame, label_col: str = "role_label"
) -> tuple[Pipeline, dict, list[dict]]:
    """Fit on ``train``; choose the configuration using ``valid`` macro F1 only.

    Returns ``(best_fitted_pipeline, best_entry, all_grid_entries)``. The pipeline
    stays fit on the training rows only (no refit on train+validation). The dummy
    has no hyperparameters. The test split is never an input.
    """
    X_train, y_train = to_model_frame(train), train[label_col].to_numpy()
    X_valid, y_valid = to_model_frame(valid), valid[label_col].to_numpy()

    grid = [(1.0, None)] if kind == "dummy" else [(C, cw) for C in C_GRID for cw in CLASS_WEIGHT_GRID]
    entries: list[dict] = []
    fitted: dict[tuple, Pipeline] = {}
    for C, cw in grid:
        pipe = build_pipeline(kind, C, cw).fit(X_train, y_train)
        m = compute_metrics(y_valid, pipe.predict(X_valid))
        entries.append(
            {
                "C": C,
                "class_weight": cw,
                "macro_f1": m["macro_f1"],
                "accuracy": m["accuracy"],
            }
        )
        fitted[(C, cw)] = pipe
    best = select_best(entries)
    return fitted[(best["C"], best["class_weight"])], best, entries


def n_tfidf_features(pipe: Pipeline) -> int | None:
    """Number of TF-IDF features of a fitted text/combined pipeline (else None)."""
    prep = pipe.named_steps.get("prep")
    if prep is None:
        return None
    for name, trans, _ in prep.transformers_:
        if name == "tfidf":
            return len(trans.vocabulary_)
    return None


def save_model(pipe: Pipeline, name: str, models_dir: Path = MODELS_DIR) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"baseline_{name}.joblib"
    joblib.dump(pipe, path)
    return path


def top_terms(pipe: Pipeline, n: int = 8) -> dict[str, list[str]]:
    """Highest-weight TF-IDF terms per class for a fitted text-only pipeline (interpretation aid)."""
    vec = pipe.named_steps["prep"].named_transformers_["tfidf"]
    clf = pipe.named_steps["clf"]
    vocab = np.array(vec.get_feature_names_out())
    out = {}
    for i, cls in enumerate(clf.classes_):
        out[str(cls)] = vocab[np.argsort(clf.coef_[i])[::-1][:n]].tolist()
    return out

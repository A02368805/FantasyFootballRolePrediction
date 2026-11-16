"""Entry point for the Fantasy Football Role Prediction project.

Commands:

    python main.py --check-data      # real-data smoke test (2024 targets chart)
    python main.py --prepare-data    # 2022-2025 data prep, features, labels, splits
    python main.py --prepare-data --refresh-data   # ignore the Parquet cache
    python main.py --train-baselines # NLP text features + 4 baseline models (train 2022-23, validate 2024)
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import polars as pl

from src.data_loader import (
    HISTORICAL_SEASONS,
    DataLoadError,
    load_2024_weekly_player_stats,
    load_historical_player_stats,
    load_pbp_text_cached,
    load_schedules_cached,
)
from src.evaluation import compute_metrics, format_metrics_table, paired_bootstrap_macro_f1
from src.feature_engineering import FEATURE_COLUMNS
from src.models import (
    MODEL_KINDS,
    MODEL_NAMES,
    RANDOM_SEED,
    STAT_FEATURES,
    n_tfidf_features,
    save_model,
    to_model_frame,
    top_terms,
    tune_on_validation,
)
from src.preprocessing import (
    PROCESSED_DATA_DIR,
    TEST_SEASONS,
    TRAIN_SEASONS,
    VALIDATION_SEASONS,
    chronological_split,
    label_distribution_table,
    prepare_dataset,
    save_processed,
    season_label_counts,
)
from src.text_features import (
    KEYS,
    TFIDF_PARAMS,
    add_window_documents,
    attach_text,
    build_player_game_documents,
    build_player_plays,
    name_masking_audit,
    text_coverage_report,
    verify_no_future_text,
)
from src.visualization import (
    BASELINE_ACCURACY_PATH,
    BASELINE_CONFUSION_PATH,
    BASELINE_MACRO_F1_PATH,
    BASELINE_PREDICTED_DIST_PATH,
    GRAPHS_DIR,
    TOP_TARGETS_PATH,
    plot_avg_opportunities_by_position,
    plot_confusion_matrices,
    plot_label_distribution,
    plot_model_metric_bars,
    plot_observations_by_season,
    plot_predicted_label_distribution,
    plot_top_targets,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def check_data() -> int:
    """Download 2024 weekly stats, print a summary, and save the chart."""
    print("Downloading 2024 weekly NFL player statistics via nflreadpy...")
    try:
        df = load_2024_weekly_player_stats()
    except DataLoadError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    print(f"Rows: {df.height}")
    print(f"Columns ({len(df.columns)}):")
    for name in df.columns:
        print(f"  - {name}")

    try:
        output_path = plot_top_targets(df, output_path=TOP_TARGETS_PATH)
    except (KeyError, ValueError) as exc:
        print(f"[error] Could not generate chart: {exc}", file=sys.stderr)
        return 1

    print(f"Chart saved to: {output_path}")
    return 0


def _print_report(report, leakage, splits) -> None:
    c = report.to_dict()
    print("\n=== Rows excluded / left missing (counts) ===")
    rows = [
        ("Raw weekly player rows (all positions, all season types)", "raw_rows"),
        ("  removed: not regular season", "removed_non_regular_season"),
        ("  removed: not RB/WR/TE", "removed_non_rb_wr_te"),
        ("  removed: invalid required values", "removed_invalid_required_values"),
        ("  removed: duplicate player-game records", "removed_duplicate_player_games"),
        ("  removed: duplicate player-week records", "removed_duplicate_player_weeks"),
        ("  removed: game not on regular-season schedule", "removed_not_on_schedule"),
        ("Cleaned player-game rows kept", "placed_on_schedule_rows"),
        ("  kept despite missing optional values", "kept_rows_with_missing_optional_values"),
        ("  optional values nulled as invalid (rows kept)", "optional_values_set_to_null_invalid"),
        ("Not model-ready: fewer than 3 games of history in season", "excluded_insufficient_games_history"),
        ("Not model-ready: missed team game / team change in window", "excluded_gap_or_team_change_history"),
        ("Model-ready rows (3 consecutive team games of history)", "model_ready_rows"),
        ("  label missing: season ended (no next game)", "model_ready_unlabeled_season_end"),
        ("  label missing: no stats record in next scheduled game", "model_ready_unlabeled_no_next_game_stats"),
        ("Labeled rows", "labeled_rows"),
    ]
    for text, key in rows:
        print(f"  {text}: {c.get(key, 0):,}")

    print("\n=== Future-information check ===")
    print(f"  Predictive features ({len(FEATURE_COLUMNS)}): {', '.join(FEATURE_COLUMNS)}")
    print("  Feature columns are disjoint from audit, label and metadata columns: yes")
    if leakage:
        print(
            f"  Truncation test: {leakage['checks_run']} season/cutoff checks "
            f"({leakage['rows_compared']:,} rows) - features identical with and without later games: PASSED"
        )

    print("\n=== Samples per season and label ===")
    for r in season_label_counts(splits).to_dicts():
        print(f"  {r['season']}  {r['role_label']:<11} {r['len']:>6,}")

    print("\n=== Label distribution per chronological split (count, percent) ===")
    seasons = {"train": TRAIN_SEASONS, "validation": VALIDATION_SEASONS, "test": TEST_SEASONS}
    table = label_distribution_table(splits)
    for split in ("train", "validation", "test"):
        part = table.filter(pl.col("split") == split)
        total = splits[split].height
        years = "-".join(str(s) for s in (seasons[split][0], seasons[split][-1])) if len(seasons[split]) > 1 else seasons[split][0]
        print(f"  {split} ({years}), total {total:,}")
        for r in part.to_dicts():
            print(f"    {r['label']:<11} {r['count']:>6,}  {r['percent']:6.2f}%")


def prepare_data(refresh: bool = False) -> int:
    """Download/load cached 2022-2025 data, build features and labels, split, save, plot."""
    print(f"Preparing data for seasons {HISTORICAL_SEASONS[0]}-{HISTORICAL_SEASONS[-1]}...")
    try:
        stats = load_historical_player_stats(refresh=refresh)
        schedules = load_schedules_cached(refresh=refresh)
    except DataLoadError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    try:
        full, report, leakage = prepare_dataset(stats, schedules)
        splits = chronological_split(full)
        paths = save_processed(full, splits, report, leakage=leakage)
        graphs = [
            plot_observations_by_season(full),
            plot_label_distribution(splits),
            plot_avg_opportunities_by_position(full),
        ]
    except (KeyError, ValueError, AssertionError, DataLoadError, pl.exceptions.PolarsError) as exc:
        print(f"[error] Data preparation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    _print_report(report, leakage, splits)

    print("\n=== Output files ===")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    for path in graphs:
        print(f"  graph: {path}")
    return 0


def _print_per_class(name: str, m: dict) -> None:
    print(f"  {name}")
    print(f"    {'class':<12}{'precision':>10}{'recall':>10}{'F1':>10}{'support':>9}")
    for lab, c in m["per_class"].items():
        print(f"    {lab:<12}{c['precision']:>10.4f}{c['recall']:>10.4f}{c['f1']:>10.4f}{c['support']:>9,}")
    labels = m["labels"]
    print("    confusion matrix (rows = true, cols = predicted):")
    print("      " + " " * 12 + "".join(f"{l:>12}" for l in labels))
    for lab, row in zip(labels, m["confusion_matrix"]):
        print("      " + f"{lab:<12}" + "".join(f"{v:>12,}" for v in row))


def train_baselines(refresh: bool = False) -> int:
    """Build play-by-play text features and train/evaluate the first four baseline models.

    Train on 2022-2023, tune and evaluate on 2024 validation. The 2025 test split is
    only used for a row-count alignment check; it is never vectorized, scored, or used
    for model selection.
    """
    processed = PROCESSED_DATA_DIR
    needed = ["player_weeks_features", "train", "validation", "test", "preparation_report"]
    files = {n: processed / (f"{n}.json" if n == "preparation_report" else f"{n}.parquet") for n in needed}
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        print(
            "[error] Processed data not found. Run `python main.py --prepare-data` first.\n  Missing: "
            + ", ".join(missing),
            file=sys.stderr,
        )
        return 1

    try:
        full = pl.read_parquet(files["player_weeks_features"])
        split_dfs = {n: pl.read_parquet(files[n]) for n in ("train", "validation", "test")}
        prep_report = json.loads(files["preparation_report"].read_text())
        print("Loading play-by-play text (REG, slim columns, cached per season)...")
        pbp = load_pbp_text_cached(refresh=refresh)
        stats_for_names = load_historical_player_stats()
    except (DataLoadError, OSError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    try:
        # ---- Part 1: text documents and strict 3-game windows -------------------
        plays = build_player_plays(pbp)
        docs, doc_info = build_player_game_documents(plays, valid_keys=full.select(KEYS))
        windows = add_window_documents(full, docs)
        text_leakage = verify_no_future_text(full, docs)

        expected_rows = prep_report["counts"]["model_ready_rows"]
        if windows.height != expected_rows:
            raise AssertionError(f"Text windows ({windows.height}) != model-ready rows ({expected_rows})")
        split_text = {n: attach_text(df, windows) for n, df in split_dfs.items()}
        for n in split_dfs:
            if split_text[n].height != split_dfs[n].height:
                raise AssertionError(f"Split {n} row count changed after attaching text")
        for n, df in split_text.items():
            df.select(KEYS + ["season", "week", "text", "text_empty", "window_n_plays", "window_games_with_text"]).write_parquet(
                processed / f"text_{n}.parquet"
            )

        coverage = text_coverage_report(docs, doc_info, full, split_text)

        # ---- Name masking audit on real descriptions ----------------------------
        sample = plays  # audit every run/pass play with an official player id
        # One deterministic call: Polars' unique() does not guarantee row order, so ids and names
        # must come from the same frame or they can be mispaired.
        id_names = stats_for_names.select("player_id", "player_display_name").unique(
            subset="player_id", keep="first", maintain_order=True
        )
        name_map = dict(zip(id_names["player_id"].to_list(), id_names["player_display_name"].to_list()))
        masking = name_masking_audit(
            sample["desc"].to_list(),
            player_ids=sample["player_id"].to_list(),
            names_by_id=name_map,
            league_names=list(name_map.values()),
        )
    except (AssertionError, KeyError, ValueError, pl.exceptions.PolarsError) as exc:
        print(f"[error] Text feature construction failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    train, valid = split_text["train"], split_text["validation"]
    print("\n=== Text coverage ===")
    print(f"  REG run/pass plays with an official player id: {doc_info['plays_with_official_player_id']:,}")
    print(f"  ...matched to cleaned RB/WR/TE player-games: {doc_info['plays_matched_to_player_games']:,}")
    print(f"  ...not matched (e.g. QB rushes, other positions): {doc_info.get('plays_not_in_cleaned_player_games', 0):,}")
    print(
        f"  Cleaned player-games with >=1 play: {coverage['player_games_with_text']:,} of "
        f"{coverage['cleaned_player_games']:,} ({coverage['player_games_without_text']:,} have no play text)"
    )
    print(
        f"  Agreement with stats: carries == run plays {coverage['carries_equal_run_plays_rate']:.2%}, "
        f"targets == pass plays {coverage['targets_equal_pass_plays_rate']:.2%}"
    )
    print(f"  Windows built for all model-ready rows: {windows.height:,} (matches --prepare-data report)")
    for n in ("train", "validation", "test"):
        s = coverage["splits"][n]
        print(
            f"  {n:<10} rows {s['rows']:>6,} | empty documents {s['empty_documents']:>5,} "
            f"({s['empty_documents_pct']:.2f}%) | non-empty {s['non_empty_pct']:.2f}% | "
            f"mean plays/window {s['mean_plays_per_window']:.1f} | median words {s['median_words_per_window']:.0f}"
        )
    print("  Row counts and labels match the existing splits (alignment checked).")
    print(
        f"  No-future-text check: {text_leakage['checks_run']} season/cutoff checks "
        f"({text_leakage['rows_compared']:,} rows): PASSED; no window includes the next game."
    )

    print("\n=== Player-name masking audit (real descriptions) ===")
    print(f"  Descriptions audited: {masking['descriptions_audited']:,}")
    labels_nice = {
        "leftover_jersey_token": "leftover jersey token (e.g. '21-X')",
        "leftover_initial_surname": "leftover Initial.Surname (e.g. 'S.Brown')",
        "leftover_hash_number_name": "leftover '#10 Surname'",
        "involved_player_surname_survives": "involved player's own surname survives",
        "any_league_surname_token": "any league surname token (excl. team/city words)",
    }
    for key, text in labels_nice.items():
        print(f"  {text}: {masking[key]['count']:,} ({masking[key]['pct']:.3f}%)")
    print(f"  Most common leftover surname-like tokens: {masking['top_leftover_surname_tokens']}")

    # ---- Parts 2-4: models ---------------------------------------------------
    print(
        f"\n=== Training (2022-2023: {train.height:,} rows) and tuning on validation "
        f"(2024: {valid.height:,} rows); 2025 test untouched ==="
    )
    y_valid = valid["role_label"].to_numpy()
    X_valid = to_model_frame(valid)
    fitted, selected, grids, results = {}, {}, {}, {}
    for kind in MODEL_KINDS:
        name = MODEL_NAMES[kind]
        pipe, best, entries = tune_on_validation(kind, train, valid)
        fitted[kind], selected[kind], grids[kind] = pipe, best, entries
        results[name] = compute_metrics(y_valid, pipe.predict(X_valid))
        extra = n_tfidf_features(pipe)
        print(
            f"  {name}: selected C={best['C']}, class_weight={best['class_weight']}"
            + (f", TF-IDF vocabulary {extra:,}" if extra else "")
            + f" -> validation macro F1 {best['macro_f1']:.4f}"
        )

    print("\n=== Validation results (2024) ===")
    print(format_metrics_table(results))
    print("\n=== Per-class performance and confusion matrices (validation) ===")
    for name, m in results.items():
        _print_per_class(name, m)

    stats_f1 = results[MODEL_NAMES["stats"]]["macro_f1"]
    comb_f1 = results[MODEL_NAMES["combined"]]["macro_f1"]
    print(
        f"\nPrimary comparison (adding NLP to statistics): combined macro F1 {comb_f1:.4f} vs "
        f"statistics-only {stats_f1:.4f} (difference {comb_f1 - stats_f1:+.4f}), validation 2024 only."
    )
    preds = {k: fitted[k].predict(X_valid) for k in ("stats", "text", "combined")}
    bootstrap = {
        "combined_minus_stats": paired_bootstrap_macro_f1(y_valid, preds["combined"], preds["stats"], seed=RANDOM_SEED),
        "text_minus_stats": paired_bootstrap_macro_f1(y_valid, preds["text"], preds["stats"], seed=RANDOM_SEED),
        "combined_minus_text": paired_bootstrap_macro_f1(y_valid, preds["combined"], preds["text"], seed=RANDOM_SEED),
    }
    print("\nPaired bootstrap of validation macro-F1 differences (1,000 resamples of the 2024 rows):")
    for key, b in bootstrap.items():
        print(
            f"  {key}: {b['point_difference']:+.4f}, 95% CI [{b['ci95_low']:+.4f}, {b['ci95_high']:+.4f}], "
            f"first model better in {b['share_resamples_a_better']:.0%} of resamples"
        )
    terms = top_terms(fitted["text"])
    print("\nTop TF-IDF terms by class (text-only model):")
    for lab, words in terms.items():
        print(f"  {lab:<11} {', '.join(words)}")

    # ---- Save models, results, graphs ---------------------------------------
    model_paths = {k: save_model(p, k) for k, p in fitted.items()}
    import numpy, sklearn

    results_doc = {
        "experiment": "Phase 3 baselines: train 2022-2023, tune/evaluate on 2024 validation; 2025 test untouched",
        "random_seed": RANDOM_SEED,
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "numpy": numpy.__version__,
            "polars": pl.__version__,
        },
        "statistical_features": STAT_FEATURES,
        "tfidf_params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in TFIDF_PARAMS.items()},
        "tuning_grid": {"C": [0.01, 0.1, 1.0, 10.0], "class_weight": [None, "balanced"], "selection_metric": "validation macro F1"},
        "sizes": {n: df.height for n, df in split_text.items()},
        "selected": {MODEL_NAMES[k]: v for k, v in selected.items()},
        "grid_results": {MODEL_NAMES[k]: v for k, v in grids.items()},
        "validation_metrics": results,
        "validation_macro_f1_paired_bootstrap": bootstrap,
        "text_coverage": coverage,
        "name_masking_audit": masking,
        "no_future_text_check": text_leakage,
        "test_set_used_for_selection_or_scoring": False,
        "saved_models": {k: str(p) for k, p in model_paths.items()},
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "baseline_results.json"
    results_path.write_text(json.dumps(results_doc, indent=2, default=str))

    graphs = [
        plot_model_metric_bars(
            results, "macro_f1", "Macro F1", "Validation macro F1 by model (train 2022-2023, validate 2024)", BASELINE_MACRO_F1_PATH
        ),
        plot_model_metric_bars(
            results, "accuracy", "Accuracy", "Validation accuracy by model (train 2022-2023, validate 2024)", BASELINE_ACCURACY_PATH
        ),
        plot_confusion_matrices(results, BASELINE_CONFUSION_PATH),
        plot_predicted_label_distribution(results, BASELINE_PREDICTED_DIST_PATH),
    ]

    print("\n=== Output files ===")
    for k, p in model_paths.items():
        print(f"  model ({k}): {p}")
    print(f"  results: {results_path}")
    for n in ("train", "validation", "test"):
        print(f"  text windows ({n}): {processed / f'text_{n}.parquet'}")
    for p in graphs:
        print(f"  graph: {p}")
    print("\nThe 2025 test split was not used for model selection or scoring.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fantasy Football Role Prediction")
    parser.add_argument(
        "--check-data",
        action="store_true",
        help="Download 2024 weekly player stats, print a summary, and save a top-targets chart.",
    )
    parser.add_argument(
        "--prepare-data",
        action="store_true",
        help="Load 2022-2025 data, clean, build features and role labels, split chronologically, "
        "save Parquet files to data/processed/, and generate graphs.",
    )
    parser.add_argument(
        "--train-baselines",
        action="store_true",
        help="Build play-by-play text features and train/evaluate the dummy, statistics-only, "
        "text-only and combined logistic regression baselines (train 2022-2023, validate 2024).",
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="With --prepare-data / --train-baselines: ignore cached Parquet files and download again.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    status = 0
    ran = False
    if args.check_data:
        ran = True
        status = check_data() or status
    if args.prepare_data:
        ran = True
        status = prepare_data(refresh=args.refresh_data) or status
    if args.train_baselines:
        ran = True
        status = train_baselines(refresh=args.refresh_data) or status
    if not ran:
        parser.print_help()
    return status


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Paired bootstrap for model comparisons (baseline vs. ablation) with paired resampling.

- Uses identical outer CV splits for baseline and each ablation so OOF predictions are aligned.
- For each model and each dropped predictor computes paired bootstrap distribution of metric differences
  (delta = baseline_metric - drop_metric) using n_bootstraps resamples.
- Reports 95% percentile CIs for the paired delta, empirical two-sided p-value, and effect size (mean/std of deltas).

Exports:
- results/tables/table5_paired_bootstrap.csv
- results/tables/table5_paired_bootstrap.xlsx
- results/tables/table5_paired_bootstrap.tex

Best-practice notes implemented:
- Paired resampling (same indices across baseline and drop) to preserve correlation structure.
- Uses aggregated out-of-fold predictions from nested CV trained with identical outer splits.

This script re-trains models under the same nested CV scheme used elsewhere in the repo to obtain
aligned OOF predictions, so it must be run in the same environment and may be time-consuming.
"""
from pathlib import Path
import sys
import warnings
import numpy as np
import pandas as pd
import math

# make src importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from config import Config
from data_loader import load_data
from preprocessing import build_preprocessing_pipeline
from feature_engineering import add_feature_pipeline
from models import _get_models

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, precision_score, recall_score, f1_score


def safe_roc_auc(y_true, y_score):
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return np.nan


def safe_balanced_accuracy(y_true, y_pred):
    try:
        return float(balanced_accuracy_score(y_true, y_pred))
    except Exception:
        return np.nan


def safe_precision(y_true, y_pred):
    try:
        return float(precision_score(y_true, y_pred, zero_division=0))
    except Exception:
        return np.nan


def safe_recall(y_true, y_pred):
    try:
        return float(recall_score(y_true, y_pred, zero_division=0))
    except Exception:
        return np.nan


def safe_f1(y_true, y_pred):
    try:
        return float(f1_score(y_true, y_pred, zero_division=0))
    except Exception:
        return np.nan


def compute_metrics_from_preds(y_true, y_proba, y_pred):
    return {
        "roc_auc": safe_roc_auc(y_true, y_proba),
        "balanced_accuracy": safe_balanced_accuracy(y_true, y_pred),
        "precision": safe_precision(y_true, y_pred),
        "recall": safe_recall(y_true, y_pred),
        "f1": safe_f1(y_true, y_pred),
    }


def paired_bootstrap_delta(y_true, base_proba, base_pred, drop_proba, drop_pred, metric_name, n_bootstraps=1000, random_state=None):
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    deltas = []

    # select metric function
    if metric_name == "roc_auc":
        metric_func = lambda yt, prob, pred: safe_roc_auc(yt, prob)
    elif metric_name == "balanced_accuracy":
        metric_func = lambda yt, prob, pred: safe_balanced_accuracy(yt, pred)
    elif metric_name == "precision":
        metric_func = lambda yt, prob, pred: safe_precision(yt, pred)
    elif metric_name == "recall":
        metric_func = lambda yt, prob, pred: safe_recall(yt, pred)
    elif metric_name == "f1":
        metric_func = lambda yt, prob, pred: safe_f1(yt, pred)
    else:
        raise ValueError(f"Unknown metric {metric_name}")

    for i in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        yt = y_true[idx]
        try:
            base_val = metric_func(yt, base_proba[idx], base_pred[idx])
            drop_val = metric_func(yt, drop_proba[idx], drop_pred[idx])
            if math.isnan(base_val) or math.isnan(drop_val):
                continue
            deltas.append(base_val - drop_val)
        except Exception:
            continue

    deltas = np.array(deltas)
    if deltas.size == 0:
        return {
            "delta_mean": np.nan,
            "delta_ci_lower": np.nan,
            "delta_ci_upper": np.nan,
            "p_value": np.nan,
            "effect_size": np.nan,
            "n_samples": n,
            "n_valid_bootstraps": 0
        }

    delta_mean = deltas.mean()
    lower = np.percentile(deltas, 2.5)
    upper = np.percentile(deltas, 97.5)
    # two-sided empirical p-value: proportion of bootstrap deltas <= 0 or >=0
    # p = 2 * min(prop <= 0, prop >= 0)
    prop_le_zero = np.mean(deltas <= 0)
    prop_ge_zero = np.mean(deltas >= 0)
    p_val = 2.0 * min(prop_le_zero, prop_ge_zero)
    # guard p_val in [0,1]
    p_val = min(max(p_val, 0.0), 1.0)
    # effect size: mean / std of deltas (Cohen's dz-like)
    delta_std = deltas.std(ddof=1)
    if delta_std == 0 or math.isnan(delta_std):
        effect = np.nan
    else:
        effect = delta_mean / delta_std

    return {
        "delta_mean": float(delta_mean),
        "delta_ci_lower": float(lower),
        "delta_ci_upper": float(upper),
        "p_value": float(p_val),
        "effect_size": float(effect),
        "n_samples": int(n),
        "n_valid_bootstraps": int(deltas.size)
    }


def collect_oof_predictions(X_feat, y, cfg, models):
    """Collect OOF predictions for each model using identical outer CV splits.
    Returns dict: preds[model] = {"y_true": np.array, "y_proba": np.array, "y_pred": np.array}
    Also returns the outer_cv splits order (list of test indices concatenated) to validate alignment.
    """
    outer_cv = StratifiedKFold(n_splits=cfg.n_splits_outer, shuffle=True, random_state=cfg.random_state)
    # precompute splits in order
    splits = list(outer_cv.split(X_feat, y))

    aggregated = {name: {"y_true": [], "y_proba": [], "y_pred": []} for name in models.keys()}

    for train_idx, test_idx in splits:
        X_train, X_test = X_feat.iloc[train_idx], X_feat.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        for name, (estimator, param_grid) in models.items():
            inner_cv = StratifiedKFold(n_splits=cfg.n_splits_inner, shuffle=True, random_state=cfg.random_state)
            search = RandomizedSearchCV(estimator, param_distributions=param_grid, n_iter=cfg.n_iter_search,
                                        cv=inner_cv, scoring="roc_auc", n_jobs=1, random_state=cfg.random_state)
            search.fit(X_train, y_train)
            best = search.best_estimator_

            if hasattr(best, "predict_proba"):
                y_proba = best.predict_proba(X_test)[:, 1]
            else:
                if hasattr(best, "decision_function"):
                    scores = best.decision_function(X_test)
                    mn, mx = scores.min(), scores.max()
                    if mx > mn:
                        y_proba = (scores - mn) / (mx - mn)
                    else:
                        y_proba = np.zeros_like(scores)
                else:
                    y_proba = best.predict(X_test)
            y_pred = best.predict(X_test)

            aggregated[name]["y_true"].extend(list(y_test))
            aggregated[name]["y_proba"].extend(list(y_proba))
            aggregated[name]["y_pred"].extend(list(y_pred))

    # convert to numpy
    for name in aggregated.keys():
        aggregated[name]["y_true"] = np.array(aggregated[name]["y_true"])
        aggregated[name]["y_proba"] = np.array(aggregated[name]["y_proba"])
        aggregated[name]["y_pred"] = np.array(aggregated[name]["y_pred"])

    # also return concatenated test index order for reference
    concat_test_idx = np.concatenate([test_idx for _, test_idx in splits])
    return aggregated, concat_test_idx


def main(n_bootstraps=1000):
    cfg = Config()
    results_tables = Path("results/tables")
    results_tables.mkdir(parents=True, exist_ok=True)

    predictors = ["aggression", "police", "psychosis", "suicidality", "akute_intoxikation"]

    # load data
    X, y, df = load_data(cfg)

    # subset predictors if available
    if isinstance(X, pd.DataFrame) and set(predictors).issubset(X.columns):
        X_raw = X[predictors].copy()
    elif df is not None and set(predictors).issubset(df.columns):
        X_raw = df[predictors].copy()
        if "art_massnahme" in df.columns:
            y = df["art_massnahme"].copy()
    else:
        warnings.warn("Requested predictors not found; using all features")
        X_raw = pd.DataFrame(X)

    # preprocessing and feature engineering
    preproc = build_preprocessing_pipeline(cfg)
    X_processed = preproc.fit_transform(X_raw)
    X_feat = add_feature_pipeline(X_processed, cfg)

    models = _get_models(cfg)

    # collect baseline OOF predictions
    print("Collecting baseline OOF predictions...")
    baseline_preds, index_order = collect_oof_predictions(pd.DataFrame(X_feat), y.reset_index(drop=True), cfg, models)

    records = []

    for pred in predictors:
        print(f"Processing ablation: drop {pred}")
        if pred not in X_raw.columns:
            warnings.warn(f"Predictor {pred} not in data; skipping.")
            continue
        X_drop = X_raw.drop(columns=[pred]).copy()
        X_proc_drop = preproc.fit_transform(X_drop)
        X_feat_drop = add_feature_pipeline(X_proc_drop, cfg)

        # collect drop preds using the same outer splits (collect_oof computes splits internally but with same seed/order)
        print("Collecting drop OOF predictions...")
        drop_preds, index_order_drop = collect_oof_predictions(pd.DataFrame(X_feat_drop), y.reset_index(drop=True), cfg, models)

        # verify index orders match length
        if len(index_order) != len(index_order_drop):
            warnings.warn("Mismatch in concatenated test indices lengths between baseline and drop — alignment may be incorrect")

        for model_name in models.keys():
            base = baseline_preds[model_name]
            drop = drop_preds[model_name]
            # ensure same length
            if len(base["y_true"]) != len(drop["y_true"]):
                warnings.warn(f"Length mismatch for model {model_name} baseline vs drop; skipping")
                continue

            y_true = np.array(base["y_true"])
            base_proba = np.array(base["y_proba"])
            base_pred = np.array(base["y_pred"])
            drop_proba = np.array(drop["y_proba"])
            drop_pred = np.array(drop["y_pred"])

            # compute observed deltas
            metrics_base = compute_metrics_from_preds(y_true, base_proba, base_pred)
            metrics_drop = compute_metrics_from_preds(y_true, drop_proba, drop_pred)

            for metric in ["roc_auc", "balanced_accuracy", "precision", "recall", "f1"]:
                observed_delta = None
                try:
                    vb = metrics_base.get(metric)
                    vd = metrics_drop.get(metric)
                    if vb is None or vd is None:
                        observed_delta = np.nan
                    else:
                        observed_delta = float(vb - vd) if (not np.isnan(vb) and not np.isnan(vd)) else np.nan
                except Exception:
                    observed_delta = np.nan

                res = paired_bootstrap_delta(y_true, base_proba, base_pred, drop_proba, drop_pred,
                                             metric_name=metric, n_bootstraps=n_bootstraps, random_state=cfg.random_state)

                rec = {
                    "dropped_predictor": pred,
                    "model": model_name,
                    "metric": metric,
                    "observed_delta": observed_delta,
                    "delta_mean": res["delta_mean"],
                    "delta_ci_lower": res["delta_ci_lower"],
                    "delta_ci_upper": res["delta_ci_upper"],
                    "p_value": res["p_value"],
                    "effect_size": res["effect_size"],
                    "n_samples": res["n_samples"],
                    "n_valid_bootstraps": res["n_valid_bootstraps"]
                }
                records.append(rec)

    df_out = pd.DataFrame(records)
    out_csv = results_tables / "table5_paired_bootstrap.csv"
    out_xlsx = results_tables / "table5_paired_bootstrap.xlsx"
    out_tex = results_tables / "table5_paired_bootstrap.tex"

    df_out.to_csv(out_csv, index=False)
    # excel
    with pd.ExcelWriter(out_xlsx) as writer:
        df_out.to_excel(writer, sheet_name="paired_bootstrap", index=False)
    # latex
    try:
        tex = df_out.to_latex(index=False, float_format="{:.4f}".format)
        out_tex.write_text(tex)
    except Exception as e:
        warnings.warn(f"Failed to write LaTeX: {e}")

    # copy to manuscript
    manuscript_dir = Path("manuscript")
    manuscript_dir.mkdir(parents=True, exist_ok=True)
    (manuscript_dir / out_csv.name).write_bytes(out_csv.read_bytes())
    (manuscript_dir / out_xlsx.name).write_bytes(out_xlsx.read_bytes())
    if out_tex.exists():
        (manuscript_dir / out_tex.name).write_text(out_tex.read_text())

    print(f"Paired bootstrap results saved to {out_csv}, {out_xlsx}, {out_tex}")


if __name__ == '__main__':
    # default 1000 bootstrap samples per metric
    main(n_bootstraps=1000)

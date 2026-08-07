#!/usr/bin/env python3
"""
Ablation study implementation with bootstrap confidence intervals.

Trains the existing models while removing one clinical predictor at a time.
Uses the same preprocessing exactly as implemented in src/preprocessing.py and the
same nested cross-validation model set from src/models.py (we call _get_models to
get the same estimators and search spaces).

Outputs:
- results/ablation_fold_results.csv (per-model aggregated metrics and CIs for baseline and drops)
- results/ablation_summary.csv (mean decreases per predictor and CIs)
- manuscript/ copies of the CSVs
- figures saved under cfg.figures_dir and results/

This script DOES NOT modify existing model implementations; it re-implements
nested CV here only to collect predictions for bootstrap CI estimation.
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange

from config import Config
from data_loader import load_data
from preprocessing import build_preprocessing_pipeline
from feature_engineering import add_feature_pipeline
from models import _get_models

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, precision_score, recall_score, f1_score


def _safe_roc_auc(y_true, y_score):
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return np.nan


def _safe_balanced_accuracy(y_true, y_pred):
    try:
        return float(balanced_accuracy_score(y_true, y_pred))
    except Exception:
        return np.nan


def _safe_precision(y_true, y_pred):
    try:
        return float(precision_score(y_true, y_pred, zero_division=0))
    except Exception:
        return np.nan


def _safe_recall(y_true, y_pred):
    try:
        return float(recall_score(y_true, y_pred, zero_division=0))
    except Exception:
        return np.nan


def _safe_f1(y_true, y_pred):
    try:
        return float(f1_score(y_true, y_pred, zero_division=0))
    except Exception:
        return np.nan


def bootstrap_ci(y_true, y_proba, y_pred, metric_func, n_bootstraps=1000, alpha=0.05, random_state=None):
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    stats = []
    for i in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        yt = np.asarray(y_true)[idx]
        # metric_func may need probabilities or preds; pass both and let wrapper handle
        try:
            stat = metric_func(yt, np.asarray(y_proba)[idx], np.asarray(y_pred)[idx])
        except Exception:
            stat = np.nan
        stats.append(stat)
    stats = np.array(stats, dtype=float)
    # drop NaNs
    stats = stats[~np.isnan(stats)]
    if len(stats) == 0:
        return np.nan, np.nan
    lower = np.percentile(stats, 100 * (alpha / 2))
    upper = np.percentile(stats, 100 * (1 - alpha / 2))
    return float(lower), float(upper)


# wrappers to adapt metric_func signature for bootstrap
def metric_roc_auc(yt, yprob, ypred):
    return _safe_roc_auc(yt, yprob)


def metric_balanced_accuracy(yt, yprob, ypred):
    return _safe_balanced_accuracy(yt, ypred)


def metric_precision(yt, yprob, ypred):
    return _safe_precision(yt, ypred)


def metric_recall(yt, yprob, ypred):
    return _safe_recall(yt, ypred)


def metric_f1(yt, yprob, ypred):
    return _safe_f1(yt, ypred)


def run_nested_cv_collect_predictions(X_feat: pd.DataFrame, y: pd.Series, cfg: Config, models=None):
    """Run nested CV and collect out-of-fold predictions concatenated per model.
    Returns a dict: results[model_name] = {"y_true": array, "y_proba": array, "y_pred": array}
    """
    outer_cv = StratifiedKFold(n_splits=cfg.n_splits_outer, shuffle=True, random_state=cfg.random_state)
    if models is None:
        models = _get_models(cfg)

    # containers
    aggregated = {name: {"y_true": [], "y_proba": [], "y_pred": []} for name in models.keys()}

    for train_idx, test_idx in outer_cv.split(X_feat, y):
        X_train, X_test = X_feat.iloc[train_idx], X_feat.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        for name, (estimator, param_grid) in models.items():
            inner_cv = StratifiedKFold(n_splits=cfg.n_splits_inner, shuffle=True, random_state=cfg.random_state)
            search = RandomizedSearchCV(estimator, param_distributions=param_grid, n_iter=cfg.n_iter_search,
                                        cv=inner_cv, scoring="roc_auc", n_jobs=1, random_state=cfg.random_state)
            # fit on training fold
            search.fit(X_train, y_train)
            best = search.best_estimator_

            # predict on outer test fold
            if hasattr(best, "predict_proba"):
                y_proba = best.predict_proba(X_test)[:, 1]
            else:
                # fallback to decision_function min-max scaling
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

    # convert to numpy arrays
    for name in aggregated.keys():
        aggregated[name]["y_true"] = np.array(aggregated[name]["y_true"])
        aggregated[name]["y_proba"] = np.array(aggregated[name]["y_proba"])
        aggregated[name]["y_pred"] = np.array(aggregated[name]["y_pred"])
    return aggregated


def run_ablation(cfg: Config, predictors=None, outcome="art_massnahme", n_bootstraps=1000):
    # default predictors
    if predictors is None:
        predictors = ["aggression", "police", "psychosis", "suicidality", "akute_intoxikation"]

    results_dir = Path("results")
    manuscript_dir = Path("manuscript")
    results_dir.mkdir(parents=True, exist_ok=True)
    manuscript_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg.figures_dir).mkdir(parents=True, exist_ok=True)

    # load data
    X, y, df = load_data(cfg)

    # Try to source predictors from X or df
    if isinstance(X, pd.DataFrame) and set(predictors).issubset(X.columns):
        X_raw = X[predictors].copy()
    elif df is not None and set(predictors).issubset(df.columns):
        X_raw = df[predictors].copy()
        # if y not provided or different, take from df if available
        if outcome in df.columns:
            y = df[outcome].copy()
    else:
        # If the dataset does not contain the named predictors, warn and use whatever X has
        warnings.warn("Requested predictors not found in data. Falling back to all available features.")
        if isinstance(X, pd.DataFrame):
            X_raw = X.copy()
        else:
            X_raw = pd.DataFrame(X)

    # ensure y is a pandas Series
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    # preprocessing
    preproc = build_preprocessing_pipeline(cfg)

    # baseline: all predictors
    print("Running baseline (all predictors) and collecting predictions...")
    X_processed = preproc.fit_transform(X_raw)
    X_feat = add_feature_pipeline(X_processed, cfg)

    models = _get_models(cfg)

    baseline_preds = run_nested_cv_collect_predictions(pd.DataFrame(X_feat), y.reset_index(drop=True), cfg, models=models)

    # compute baseline metrics and bootstrap CIs per model
    baseline_stats = {}
    for name, data in baseline_preds.items():
        yt = data["y_true"]
        yprob = data["y_proba"]
        ypred = data["y_pred"]
        roc = _safe_roc_auc(yt, yprob)
        bal = _safe_balanced_accuracy(yt, ypred)
        prec = _safe_precision(yt, ypred)
        rec = _safe_recall(yt, ypred)
        f1 = _safe_f1(yt, ypred)

        # bootstrap CIs
        roc_ci = bootstrap_ci(yt, yprob, ypred, metric_roc_auc, n_bootstraps=n_bootstraps, random_state=cfg.random_state)
        bal_ci = bootstrap_ci(yt, yprob, ypred, metric_balanced_accuracy, n_bootstraps=n_bootstraps, random_state=cfg.random_state)
        prec_ci = bootstrap_ci(yt, yprob, ypred, metric_precision, n_bootstraps=n_bootstraps, random_state=cfg.random_state)
        rec_ci = bootstrap_ci(yt, yprob, ypred, metric_recall, n_bootstraps=n_bootstraps, random_state=cfg.random_state)
        f1_ci = bootstrap_ci(yt, yprob, ypred, metric_f1, n_bootstraps=n_bootstraps, random_state=cfg.random_state)

        baseline_stats[name] = {
            "roc_auc": roc,
            "roc_auc_ci_lower": roc_ci[0],
            "roc_auc_ci_upper": roc_ci[1],
            "balanced_accuracy": bal,
            "balanced_accuracy_ci_lower": bal_ci[0],
            "balanced_accuracy_ci_upper": bal_ci[1],
            "precision": prec,
            "precision_ci_lower": prec_ci[0],
            "precision_ci_upper": prec_ci[1],
            "recall": rec,
            "recall_ci_lower": rec_ci[0],
            "recall_ci_upper": rec_ci[1],
            "f1": f1,
            "f1_ci_lower": f1_ci[0],
            "f1_ci_upper": f1_ci[1]
        }

    # run ablations
    ablation_records = []

    for pred in predictors:
        print(f"Running ablation: drop {pred} and collecting predictions...")
        if pred in X_raw.columns:
            X_drop = X_raw.drop(columns=[pred]).copy()
        else:
            warnings.warn(f"Predictor {pred} not present in data; recording NA results.")
            for m in models.keys():
                rec = {"dropped_predictor": pred, "model": m}
                # baseline values (if present)
                b = baseline_stats.get(m, {})
                for k, v in b.items():
                    rec[f"baseline_{k}"] = v
                # drop metrics NA
                rec.update({
                    "drop_roc_auc": np.nan, "drop_roc_auc_ci_lower": np.nan, "drop_roc_auc_ci_upper": np.nan,
                    "drop_balanced_accuracy": np.nan, "drop_balanced_accuracy_ci_lower": np.nan, "drop_balanced_accuracy_ci_upper": np.nan,
                    "drop_precision": np.nan, "drop_precision_ci_lower": np.nan, "drop_precision_ci_upper": np.nan,
                    "drop_recall": np.nan, "drop_recall_ci_lower": np.nan, "drop_recall_ci_upper": np.nan,
                    "drop_f1": np.nan, "drop_f1_ci_lower": np.nan, "drop_f1_ci_upper": np.nan,
                })
                ablation_records.append(rec)
            continue

        # preprocessing and feature engineering
        X_proc_drop = preproc.fit_transform(X_drop)
        X_feat_drop = add_feature_pipeline(X_proc_drop, cfg)

        drop_preds = run_nested_cv_collect_predictions(pd.DataFrame(X_feat_drop), y.reset_index(drop=True), cfg, models=models)

        for name, data in drop_preds.items():
            yt = data["y_true"]
            yprob = data["y_proba"]
            ypred = data["y_pred"]
            roc = _safe_roc_auc(yt, yprob)
            bal = _safe_balanced_accuracy(yt, ypred)
            prec = _safe_precision(yt, ypred)
            rec = _safe_recall(yt, ypred)
            f1 = _safe_f1(yt, ypred)

            # bootstrap CIs
            roc_ci = bootstrap_ci(yt, yprob, ypred, metric_roc_auc, n_bootstraps=n_bootstraps, random_state=cfg.random_state)
            bal_ci = bootstrap_ci(yt, yprob, ypred, metric_balanced_accuracy, n_bootstraps=n_bootstraps, random_state=cfg.random_state)
            prec_ci = bootstrap_ci(yt, yprob, ypred, metric_precision, n_bootstraps=n_bootstraps, random_state=cfg.random_state)
            rec_ci = bootstrap_ci(yt, yprob, ypred, metric_recall, n_bootstraps=n_bootstraps, random_state=cfg.random_state)
            f1_ci = bootstrap_ci(yt, yprob, ypred, metric_f1, n_bootstraps=n_bootstraps, random_state=cfg.random_state)

            rec = {"dropped_predictor": pred, "model": name}
            b = baseline_stats.get(name, {})
            for k, v in b.items():
                rec[f"baseline_{k}"] = v
            rec.update({
                "drop_roc_auc": roc,
                "drop_roc_auc_ci_lower": roc_ci[0],
                "drop_roc_auc_ci_upper": roc_ci[1],
                "drop_balanced_accuracy": bal,
                "drop_balanced_accuracy_ci_lower": bal_ci[0],
                "drop_balanced_accuracy_ci_upper": bal_ci[1],
                "drop_precision": prec,
                "drop_precision_ci_lower": prec_ci[0],
                "drop_precision_ci_upper": prec_ci[1],
                "drop_recall": rec,
                "drop_recall_ci_lower": rec_ci[0],
                "drop_recall_ci_upper": rec_ci[1],
                "drop_f1": f1,
                "drop_f1_ci_lower": f1_ci[0],
                "drop_f1_ci_upper": f1_ci[1],
            })
            ablation_records.append(rec)

    ablation_df = pd.DataFrame(ablation_records)
    # compute deltas and delta CIs if possible (delta = baseline - drop)
    def safe_subtract(a, b):
        try:
            if pd.isna(a) or pd.isna(b):
                return np.nan
            return float(a - b)
        except Exception:
            return np.nan

    # derive delta columns
    ablation_df["roc_auc_delta"] = ablation_df.apply(lambda r: safe_subtract(r.get("baseline_roc_auc"), r.get("drop_roc_auc")), axis=1)
    ablation_df["balanced_accuracy_delta"] = ablation_df.apply(lambda r: safe_subtract(r.get("baseline_balanced_accuracy"), r.get("drop_balanced_accuracy")), axis=1)

    # Save full ablation results
    ablation_df.to_csv(results_dir / "ablation_full_results.csv", index=False)
    ablation_df.to_csv(manuscript_dir / "ablation_full_results.csv", index=False)

    # summary: average delta across models per predictor
    summary = ablation_df.groupby("dropped_predictor").agg({
        "roc_auc_delta": ["mean", "std"],
        "balanced_accuracy_delta": ["mean", "std"]
    }).reset_index()
    summary.columns = ["dropped_predictor", "roc_auc_delta_mean", "roc_auc_delta_std", "balanced_accuracy_delta_mean", "balanced_accuracy_delta_std"]
    summary.to_csv(results_dir / "ablation_summary.csv", index=False)
    summary.to_csv(manuscript_dir / "ablation_summary.csv", index=False)

    # plot average delta per predictor (mean roc delta)
    plt.figure(figsize=(6,4))
    sns.barplot(data=summary, x="dropped_predictor", y="roc_auc_delta_mean")
    plt.ylabel("Mean decrease in ROC AUC (baseline - drop)")
    plt.xlabel("Dropped predictor")
    plt.title("Ablation: mean ROC AUC decrease per dropped predictor")
    plt.tight_layout()
    plt.savefig(Path(cfg.figures_dir) / "ablation_roc_delta.png")
    plt.savefig(results_dir / "ablation_roc_delta.png")
    plt.close()

    # heatmap of roc_auc_delta per model/predictor
    heat = ablation_df.pivot(index="model", columns="dropped_predictor", values="roc_auc_delta")
    plt.figure(figsize=(8,4))
    sns.heatmap(heat, annot=True, fmt=".4f", cmap="coolwarm", center=0)
    plt.title("ROC AUC decrease (baseline - drop) per model and predictor")
    plt.tight_layout()
    plt.savefig(Path(cfg.figures_dir) / "ablation_heatmap.png")
    plt.savefig(results_dir / "ablation_heatmap.png")
    plt.close()

    print("Ablation study with bootstrap CIs completed. Results saved to results/ and manuscript/")
    return ablation_df, summary


if __name__ == '__main__':
    cfg = Config()
    # default to 1000 bootstrap samples; can be changed by calling run_ablation directly
    run_ablation(cfg, n_bootstraps=1000)

#!/usr/bin/env python3
"""
Ablation study implementation.

Trains the existing models while removing one clinical predictor at a time.
Uses the same preprocessing exactly as implemented in src/preprocessing.py and the
same nested cross-validation and model set from src/models.py (via run_nested_cv_for_models).

Outputs:
- results/ablation_fold_results.csv (per-fold results for each ablation)
- results/ablation_summary.csv (mean decreases per model/predictor)
- manuscript/ copies of the CSVs
- figures saved under cfg.figures_dir and results/

This script DOES NOT modify existing model implementations; it calls run_nested_cv_for_models
which already implements nested stratified CV used elsewhere in the repo.
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import Config
from data_loader import load_data
from preprocessing import build_preprocessing_pipeline
from feature_engineering import add_feature_pipeline
from models import run_nested_cv_for_models


def run_ablation(cfg: Config, predictors=None, outcome="art_massnahme"):
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

    # baseline: all predictors
    print("Running baseline (all predictors)...")
    preproc = build_preprocessing_pipeline(cfg)
    X_processed = preproc.fit_transform(X_raw)
    X_feat = add_feature_pipeline(X_processed, cfg)

    baseline_df = run_nested_cv_for_models(pd.DataFrame(X_feat), y.reset_index(drop=True), cfg, df)
    # baseline_df has columns model, fold, roc_auc, balanced_accuracy, best_params

    # compute baseline means per model
    baseline_means = baseline_df.groupby("model").agg({"roc_auc": "mean", "balanced_accuracy": "mean"}).rename(columns={"roc_auc": "roc_auc_baseline", "balanced_accuracy": "balanced_accuracy_baseline"})

    ablation_records = []

    for pred in predictors:
        print(f"Running ablation: drop {pred}")
        if pred in X_raw.columns:
            X_drop = X_raw.drop(columns=[pred]).copy()
        else:
            # if pred not present, skip but record NA
            warnings.warn(f"Predictor {pred} not present in data; recording NA results.")
            for m in baseline_means.index:
                ablation_records.append({"dropped_predictor": pred, "model": m, "roc_auc": np.nan, "balanced_accuracy": np.nan})
            continue

        # preprocessing and feature engineering
        X_proc_drop = preproc.fit_transform(X_drop)
        X_feat_drop = add_feature_pipeline(X_proc_drop, cfg)

        # run nested CV
        drop_df = run_nested_cv_for_models(pd.DataFrame(X_feat_drop), y.reset_index(drop=True), cfg, df)

        # compute mean metrics per model
        drop_means = drop_df.groupby("model").agg({"roc_auc": "mean", "balanced_accuracy": "mean"}).rename(columns={"roc_auc": "roc_auc_drop", "balanced_accuracy": "balanced_accuracy_drop"})

        # merge with baseline and compute deltas = baseline - drop
        merged = baseline_means.join(drop_means, how="left")
        merged = merged.reset_index()
        for _, row in merged.iterrows():
            m = row["model"]
            roc_base = row["roc_auc_baseline"]
            roc_drop = row.get("roc_auc_drop", np.nan)
            bal_base = row["balanced_accuracy_baseline"]
            bal_drop = row.get("balanced_accuracy_drop", np.nan)
            roc_delta = np.nan
            bal_delta = np.nan
            if not pd.isna(roc_base) and not pd.isna(roc_drop):
                roc_delta = roc_base - roc_drop
            if not pd.isna(bal_base) and not pd.isna(bal_drop):
                bal_delta = bal_base - bal_drop
            ablation_records.append({
                "dropped_predictor": pred,
                "model": m,
                "roc_auc_baseline": roc_base,
                "roc_auc_drop": roc_drop,
                "roc_auc_delta": roc_delta,
                "balanced_accuracy_baseline": bal_base,
                "balanced_accuracy_drop": bal_drop,
                "balanced_accuracy_delta": bal_delta
            })

    ablation_df = pd.DataFrame(ablation_records)
    ablation_df.to_csv(results_dir / "ablation_fold_results.csv", index=False)
    ablation_df.to_csv(manuscript_dir / "ablation_fold_results.csv", index=False)

    # summary: average delta across models per predictor
    summary = ablation_df.groupby("dropped_predictor").agg({"roc_auc_delta": ["mean", "std"], "balanced_accuracy_delta": ["mean", "std"]}).reset_index()
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

    print("Ablation study completed. Results saved to results/ and manuscript/")
    return ablation_df, summary


if __name__ == '__main__':
    cfg = Config()
    run_ablation(cfg)

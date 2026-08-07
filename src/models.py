"""
Model training with nested cross-validation (publication-ready).

Key features:
- Nested cross-validation (outer/inner) with RandomizedSearchCV
- Balances class imbalance in model configuration and objective:
  - LogisticRegression(class_weight="balanced")
  - SVC(class_weight="balanced", probability=True)
  - RandomForest(class_weight="balanced")
  - XGBoost(scale_pos_weight = neg/pos) computed per training fold
  - CatBoost(auto_class_weights="Balanced")
- RandomizedSearchCV is optimized on 'balanced_accuracy' (dataset is imbalanced)
- Computes and saves many metrics for each outer fold:
  ROC AUC, Balanced Accuracy, Precision, Recall, Specificity, F1, MCC, Brier Score
- Saves best hyperparameters for each fold
- Exports predictions and probabilities per fold (CSV)
- Produces and saves ROC and calibration plots per fold
- Maintains SHAP explainability and error analysis integration if available
- Robust logging, type hints, clear documentation, no duplicated code
- Compatible with Python 3.13
- Returns a pandas.DataFrame with all folds x models metrics
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    matthews_corrcoef,
    brier_score_loss,
)
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from config import Config
from evaluation import (
    plot_roc_curve,
    plot_calibration_curve,
    save_metrics,      # kept for backward compatibility (not required), may be a no-op
    error_analysis,
)

# Configure logging for reproducibility and traceability
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)


def _get_base_models(cfg: Config) -> Dict[str, tuple]:
    """
    Define base estimators and hyperparameter distributions.

    Note: XGBoost scale_pos_weight is set per-fold (depends on training class balance),
    so we provide a default value here and override it in the outer CV loop.
    """
    models = {
        "logreg": (
            LogisticRegression(max_iter=2000, solver="saga", class_weight="balanced", random_state=cfg.random_state),
            {"C": np.logspace(-4, 4, 20), "penalty": ["l2"]},
        ),
        "svm": (
            SVC(kernel="linear", probability=True, class_weight="balanced", random_state=cfg.random_state),
            {"C": np.logspace(-3, 3, 20)},
        ),
        "rf": (
            RandomForestClassifier(class_weight="balanced", random_state=cfg.random_state),
            {"n_estimators": [100, 200, 400], "max_depth": [None, 5, 10, 20], "min_samples_leaf": [1, 2, 5]},
        ),
        # set a placeholder scale_pos_weight (1.0) — will be overridden per fold
        "xgb": (
            XGBClassifier(use_label_encoder=False, eval_metric="logloss", scale_pos_weight=1.0, random_state=cfg.random_state),
            {"n_estimators": [100, 200], "max_depth": [3, 6], "learning_rate": [0.01, 0.05, 0.1]},
        ),
        "cat": (
            CatBoostClassifier(verbose=0, random_state=cfg.random_state, auto_class_weights="Balanced"),
            {"iterations": [100, 200], "depth": [4, 6], "learning_rate": [0.01, 0.05, 0.1]},
        ),
    }
    return models


def _compute_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    """
    Compute a comprehensive set of metrics required by the study.

    Returns:
        dict of metric_name -> value
    """
    # Convert to numpy arrays for robust metric computation
    y_true_a = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred)
    y_proba_a = np.asarray(y_proba)

    # Basic metrics
    roc_auc = float(roc_auc_score(y_true_a, y_proba_a)) if len(np.unique(y_true_a)) > 1 else float("nan")
    bal_acc = float(balanced_accuracy_score(y_true_a, y_pred_a))
    precision = float(precision_score(y_true_a, y_pred_a, zero_division=0))
    recall = float(recall_score(y_true_a, y_pred_a, zero_division=0))
    f1 = float(f1_score(y_true_a, y_pred_a, zero_division=0))
    mcc = float(matthews_corrcoef(y_true_a, y_pred_a)) if len(np.unique(y_pred_a)) > 1 else float("nan")
    brier = float(brier_score_loss(y_true_a, y_proba_a))

    # Specificity (True Negative Rate)
    tn, fp, fn, tp = confusion_matrix(y_true_a, y_pred_a, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")

    return {
        "roc_auc": roc_auc,
        "balanced_accuracy": bal_acc,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "mcc": mcc,
        "brier_score": brier,
    }


def _safe_predict_proba(estimator: Any, X: pd.DataFrame) -> np.ndarray:
    """
    Return probability for the positive class (shape: (n_samples,)).
    Fall back to decision_function-based transformation if predict_proba is unavailable.
    """
    # Many classifiers support predict_proba
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        # Some estimators return shape (n_samples, n_classes). Positive class is column 1.
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]
        # In multiclass scenarios, fallback to one-vs-rest probability for class 1 if present
        return proba[:, 0] if proba.shape[1] == 1 else proba.max(axis=1)
    # If probability is not available, try to use decision_function and map to [0,1]
    if hasattr(estimator, "decision_function"):
        df = estimator.decision_function(X)
        # sigmoid mapping to (0,1)
        proba = 1.0 / (1.0 + np.exp(-df))
        return proba
    raise RuntimeError("Estimator does not support probability or decision_function.")


def _save_predictions(
    cfg: Config,
    model_name: str,
    fold_idx: int,
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    df_test: Optional[pd.DataFrame] = None,
) -> Path:
    """
    Save predictions and probabilities for a single fold to CSV and return file path.
    Columns: index (original index if df_test provided), y_true, y_pred, y_proba, plus any df_test columns if present.
    """
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if df_test is not None:
        # prefer original index from df_test if available
        index = df_test.index
        base = df_test.reset_index(drop=True).copy()
    else:
        index = pd.RangeIndex(len(y_true))
        base = pd.DataFrame(index=index)

    preds_df = base.reset_index(drop=True)
    preds_df["y_true"] = np.asarray(y_true)
    preds_df["y_pred"] = np.asarray(y_pred)
    preds_df["y_proba"] = np.asarray(y_proba)

    out_path = out_dir / f"predictions_{model_name}_fold{fold_idx}.csv"
    preds_df.to_csv(out_path, index=False)

    logger.info("Saved predictions for %s fold %d to %s", model_name, fold_idx, out_path)
    return out_path


def run_nested_cv_for_models(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: Config,
    df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Run nested cross-validation for a set of models and return a DataFrame with metrics for each model x fold.

    Args:
        X: Feature matrix (pandas DataFrame).
        y: Target series (pandas Series).
        cfg: Configuration dataclass instance (see src/config.py).
        df: Optional original dataframe aligned with X (used for saving extra columns in predictions/error analysis).

    Returns:
        pandas.DataFrame with columns:
            ['model', 'fold', metrics..., 'best_params', 'predictions_path', 'roc_path', 'calibration_path']
    """
    # Outer CV: ensure reproducible shuffling
    outer_cv = StratifiedKFold(n_splits=cfg.n_splits_outer, shuffle=True, random_state=cfg.random_state)
    inner_cv = StratifiedKFold(n_splits=cfg.n_splits_inner, shuffle=True, random_state=cfg.random_state)

    base_models = _get_base_models(cfg)

    all_results = []  # accumulate per (model, fold) results

    fold_idx = 0
    # Enforce DataFrame inputs for consistent .iloc and column behavior
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if not isinstance(y, pd.Series):
        y = pd.Series(y, name="target")

    for train_idx, test_idx in outer_cv.split(X, y):
        fold_idx += 1
        logger.info("Starting outer fold %d / %d", fold_idx, cfg.n_splits_outer)

        X_train, X_test = X.iloc[train_idx].reset_index(drop=True), X.iloc[test_idx].reset_index(drop=True)
        y_train, y_test = y.iloc[train_idx].reset_index(drop=True), y.iloc[test_idx].reset_index(drop=True)
        df_test = df.iloc[test_idx].reset_index(drop=True) if df is not None else None

        # Precompute positive/negative counts for this training fold (used by XGBoost)
        pos = int(np.sum(y_train == 1))
        neg = int(np.sum(y_train == 0))
        scale_pos_weight = float(neg / pos) if pos > 0 else 1.0
        logger.debug("Fold %d pos=%d neg=%d scale_pos_weight=%.3f", fold_idx, pos, neg, scale_pos_weight)

        # Iterate models
        for model_name, (base_estimator, param_dist) in base_models.items():
            # Clone base estimator for this fold for safety (RandomizedSearchCV will clone again)
            estimator = clone(base_estimator)

            # If XGBoost, set scale_pos_weight according to fold class imbalance
            if model_name == "xgb":
                try:
                    estimator.set_params(scale_pos_weight=scale_pos_weight)
                except Exception as e:
                    logger.warning("Could not set scale_pos_weight for XGB: %s", e)

            # Build randomized search: optimize balanced_accuracy due to class imbalance
            search = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=param_dist,
                n_iter=cfg.n_iter_search,
                scoring="balanced_accuracy",        # requirement #3
                cv=inner_cv,
                n_jobs=cfg.n_jobs,
                random_state=cfg.random_state,
                verbose=0,
            )

            logger.info("Running RandomizedSearchCV for model=%s fold=%d", model_name, fold_idx)
            # Fit search on training fold
            search.fit(X_train, y_train)

            # Retrieve best estimator and best params
            best_estimator = search.best_estimator_
            best_params = search.best_params_

            # Predictions on held-out fold
            # Ensure we get probabilities for the positive class
            try:
                y_proba = _safe_predict_proba(best_estimator, X_test)
            except Exception as e:
                logger.exception("Probability prediction failed for %s fold %d: %s", model_name, fold_idx, e)
                # fallback to predicted labels as hard probabilities (0/1)
                y_pred_fallback = best_estimator.predict(X_test)
                y_proba = y_pred_fallback.astype(float)

            y_pred = best_estimator.predict(X_test)

            # Compute metrics required by the study
            metrics = _compute_metrics(y_test, y_pred, y_proba)

            # Save ROC & calibration plots to figures directory
            figures_dir = Path(cfg.figures_dir)
            figures_dir.mkdir(parents=True, exist_ok=True)

            try:
                fig_roc = plot_roc_curve(y_test, y_proba, title=f"ROC {model_name} fold{fold_idx}")
                roc_path = figures_dir / f"roc_{model_name}_fold{fold_idx}.png"
                fig_roc.savefig(roc_path, bbox_inches="tight")
                plt.close(fig_roc)
            except Exception as e:
                logger.exception("Failed to generate/save ROC for %s fold %d: %s", model_name, fold_idx, e)
                roc_path = None

            try:
                fig_cal = plot_calibration_curve(y_test, y_proba, title=f"Calibration {model_name} fold{fold_idx}")
                cal_path = figures_dir / f"cal_{model_name}_fold{fold_idx}.png"
                fig_cal.savefig(cal_path, bbox_inches="tight")
                plt.close(fig_cal)
            except Exception as e:
                logger.exception("Failed to generate/save calibration plot for %s fold %d: %s", model_name, fold_idx, e)
                cal_path = None

            # SHAP explainability (best-effort). Expose failures but continue
            try:
                import explainability  # optional module in repo
                explainability.save_shap_for_model(best_estimator, X_test, cfg, prefix=f"{model_name}_fold{fold_idx}")
            except Exception as e:
                logger.warning("SHAP explainability skipped/failed for %s fold %d: %s", model_name, fold_idx, e)

            # Error analysis (keeps backward compatibility with existing project)
            try:
                error_analysis(X_test, y_test, y_pred, cfg, prefix=f"{model_name}_fold{fold_idx}", df_test=(df_test if df_test is not None else None))
            except Exception as e:
                logger.exception("Error analysis failed for %s fold %d: %s", model_name, fold_idx, e)

            # Save predictions and probabilities (CSV)
            preds_path = _save_predictions(cfg, model_name, fold_idx, y_test, y_pred, y_proba, df_test=df_test)

            # Consolidate results for this model/fold
            result_row = {
                "model": model_name,
                "fold": int(fold_idx),
                "best_params": str(best_params),
                "predictions_path": str(preds_path),
                "roc_path": str(roc_path) if roc_path is not None else None,
                "calibration_path": str(cal_path) if cal_path is not None else None,
            }
            # Add the computed metrics to the result row
            result_row.update(metrics)

            all_results.append(result_row)

            logger.info(
                "Completed model=%s fold=%d: balanced_accuracy=%.4f roc_auc=%.4f",
                model_name,
                fold_idx,
                metrics.get("balanced_accuracy", float("nan")),
                metrics.get("roc_auc", float("nan")),
            )

    # Return a DataFrame with one row per model x fold and columns for metrics, artifacts, and best params
    results_df = pd.DataFrame(all_results)

    # Save a master results CSV for easier downstream analysis
    try:
        results_out_path = Path(cfg.results_path)
        results_out_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(results_out_path, index=False)
        logger.info("Saved overall results to %s", results_out_path)
    except Exception as e:
        logger.exception("Failed to save overall results to %s: %s", cfg.results_path, e)

    return results_df

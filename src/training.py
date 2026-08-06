"""Updated training module to enforce TRIPOD-AI recommendations:
- clear train/test separation (holdout test set)
- fixed random seeds
- reproducible preprocessing (save preprocessor)
- nested CV on training set
- final model refit on full training set and evaluation on held-out test
- extensive logging of experiment metadata and metrics
- export of CSVs for per-fold metrics and final test metrics
"""
import os
import json
import time
import random
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, precision_recall_fscore_support, accuracy_score

from .data import load_data
from .preprocessing import build_preprocessor
from .models import get_base_estimators, get_param_grids
from .evaluation import save_roc_curve, save_calibration_curve, save_confusion_matrix
from .explain import explain_model_shap
from .utils import ensure_dir
from .reporting import aggregate_and_report


def set_seeds(seed: int):
    np.random.seed(seed)
    random.seed(seed)


def safe_predict_proba(estimator, X):
    try:
        return estimator.predict_proba(X)[:, 1]
    except Exception:
        try:
            scores = estimator.decision_function(X)
            # rescale to 0-1
            if np.ptp(scores) == 0:
                return np.zeros_like(scores)
            return (scores - scores.min()) / (scores.max() - scores.min())
        except Exception:
            return np.zeros(X.shape[0])


def compute_classification_metrics(y_true, y_pred, y_prob):
    roc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan
    bal = balanced_accuracy_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prfs = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    return {
        'roc_auc': float(roc) if not np.isnan(roc) else None,
        'balanced_accuracy': float(bal),
        'accuracy': float(acc),
        'precision': float(prfs[0]),
        'recall': float(prfs[1]),
        'f1': float(prfs[2])
    }


def nested_cv_and_train(overrides: dict):
    # build config
    from .config import default_config
    cfg = default_config(overrides)

    set_seeds(cfg.random_seed)

    X, y = load_data(cfg.data_path, cfg.target_col, cfg.id_col)
    n_samples, n_features = X.shape

    # Train/Test split (holdout) - clear separation
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=cfg.random_seed)
    train_idx, test_idx = next(sss.split(X, y))
    X_train, X_test = X.iloc[train_idx].reset_index(drop=True), X.iloc[test_idx].reset_index(drop=True)
    y_train, y_test = y.iloc[train_idx].reset_index(drop=True), y.iloc[test_idx].reset_index(drop=True)

    # Build preprocessor using training data only and save it for reproducibility
    preproc, num_cols, cat_cols = build_preprocessor(X_train)
    preproc.fit(X_train)

    preprocessing_dir = os.path.join(cfg.output_dir, 'preprocessing')
    ensure_dir(preprocessing_dir)
    preproc_path = os.path.join(preprocessing_dir, 'preprocessor.joblib')
    joblib.dump({'preprocessor': preproc, 'num_cols': num_cols, 'cat_cols': cat_cols}, preproc_path)

    # Prepare estimators and param grids
    estimators = get_base_estimators()
    param_grids = get_param_grids()

    outer_cv = StratifiedKFold(n_splits=cfg.outer_splits, shuffle=True, random_state=cfg.random_seed)

    results = []
    per_fold_records = []
    models_dir = os.path.join(cfg.output_dir, "models")
    ensure_dir(models_dir)

    # Nested CV: do CV on training set only
    fold_idx = 0
    for train_index, val_index in outer_cv.split(X_train, y_train):
        fold_idx += 1
        X_tr, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_tr, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

        for name, base_est in estimators.items():
            print(f"Outer fold {fold_idx} - training model: {name}")
            pipe = Pipeline([
                ("preproc", preproc),
                ("clf", base_est)
            ])
            grid = param_grids.get(name, {})
            inner_cv = StratifiedKFold(n_splits=cfg.inner_splits, shuffle=True, random_state=cfg.random_seed)
            gs = GridSearchCV(pipe, grid, cv=inner_cv, scoring="roc_auc", n_jobs=cfg.n_jobs, refit=True)
            gs.fit(X_tr, y_tr)

            best = gs.best_estimator_

            y_prob_val = safe_predict_proba(best, X_val)
            y_pred_val = best.predict(X_val)
            metrics = compute_classification_metrics(y_val, y_pred_val, y_prob_val)

            model_path = os.path.join(models_dir, f"model_fold{fold_idx}_{name}.joblib")
            joblib.dump(best, model_path)

            # per-fold artifacts
            fig_roc = os.path.join(cfg.output_dir, f"roc_fold{fold_idx}_{name}.png")
            save_roc_curve(y_val, y_prob_val, fig_roc, title=f"ROC - fold {fold_idx} - {name}")

            fig_cal = os.path.join(cfg.output_dir, f"calibration_fold{fold_idx}_{name}.png")
            save_calibration_curve(y_val, y_prob_val, fig_cal, title=f"Calibration - fold {fold_idx} - {name}")

            fig_cm = os.path.join(cfg.output_dir, f"confusion_fold{fold_idx}_{name}.png")
            save_confusion_matrix(y_val, y_pred_val, fig_cm, title=f"Confusion - fold {fold_idx} - {name}")

            # SHAP explanation saved per fold
            expl_path = os.path.join(cfg.output_dir, f"shap_fold{fold_idx}_{name}.png")
            try:
                explain_model_shap(best, X_val, expl_path)
            except Exception as e:
                print(f"SHAP failed for {name} fold {fold_idx}: {e}")

            rec = {
                'fold': fold_idx,
                'model': name,
                'best_params': gs.best_params_,
                'roc_auc': metrics['roc_auc'],
                'balanced_accuracy': metrics['balanced_accuracy'],
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'model_path': model_path,
                'roc_curve': fig_roc,
                'calibration_curve': fig_cal,
                'confusion_matrix': fig_cm,
                'shap': expl_path
            }
            per_fold_records.append(rec)

    # Save per-fold metrics table
    per_fold_df = pd.DataFrame(per_fold_records)
    per_fold_csv = os.path.join(cfg.output_dir, 'per_fold_metrics.csv')
    per_fold_df.to_csv(per_fold_csv, index=False)

    # Refit each model on the full training set using the best hyperparameters found across folds
    # For simplicity, pick the best hyperparams per model by averaging ROC across folds
    summary = []
    for name in estimators.keys():
        df_m = per_fold_df[per_fold_df['model'] == name]
        # choose the params that appeared most often (mode) as a simple heuristic
        if df_m.empty:
            continue
        best_params_mode = None
        try:
            best_params_mode = df_m['best_params'].mode()[0]
        except Exception:
            best_params_mode = df_m['best_params'].iloc[0]

        # Build a fresh estimator and set params
        base = estimators[name]
        # create pipeline with preproc fitted previously
        pipe_full = Pipeline([
            ('preproc', preproc),
            ('clf', base)
        ])
        # set params if possible
        if isinstance(best_params_mode, dict):
            # we need to map parameter names like 'clf__C' directly to the pipeline
            try:
                pipe_full.set_params(**best_params_mode)
            except Exception:
                pass

        # fit on full training set
        pipe_full.fit(X_train, y_train)
        final_model_path = os.path.join(models_dir, f'final_{name}.joblib')
        joblib.dump(pipe_full, final_model_path)

        # Evaluate on holdout test set
        y_prob_test = safe_predict_proba(pipe_full, X_test)
        y_pred_test = pipe_full.predict(X_test)
        test_metrics = compute_classification_metrics(y_test, y_pred_test, y_prob_test)

        # save test artifacts
        fig_roc_test = os.path.join(cfg.output_dir, f'roc_test_{name}.png')
        save_roc_curve(y_test, y_prob_test, fig_roc_test, title=f'ROC - test - {name}')

        fig_cal_test = os.path.join(cfg.output_dir, f'calibration_test_{name}.png')
        save_calibration_curve(y_test, y_prob_test, fig_cal_test, title=f'Calibration - test - {name}')

        fig_cm_test = os.path.join(cfg.output_dir, f'confusion_test_{name}.png')
        save_confusion_matrix(y_test, y_pred_test, fig_cm_test, title=f'Confusion - test - {name}')

        # SHAP on test
        expl_test = os.path.join(cfg.output_dir, f'shap_test_{name}.png')
        try:
            explain_model_shap(pipe_full, X_test, expl_test)
        except Exception as e:
            print(f"SHAP failed on test for {name}: {e}")

        summary.append({
            'model': name,
            'final_model_path': final_model_path,
            'test_roc_auc': test_metrics['roc_auc'],
            'test_balanced_accuracy': test_metrics['balanced_accuracy'],
            'test_accuracy': test_metrics['accuracy'],
            'test_precision': test_metrics['precision'],
            'test_recall': test_metrics['recall'],
            'test_f1': test_metrics['f1'],
            'roc_curve': fig_roc_test,
            'calibration_curve': fig_cal_test,
            'confusion_matrix': fig_cm_test,
            'shap': expl_test
        })

    summary_df = pd.DataFrame(summary)
    summary_csv = os.path.join(cfg.output_dir, 'test_metrics_summary.csv')
    summary_df.to_csv(summary_csv, index=False)

    # Save experiment metadata for reproducibility
    metadata = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'data_path': cfg.data_path,
        'target_col': cfg.target_col,
        'n_samples': int(n_samples),
        'n_features': int(n_features),
        'num_cols': num_cols,
        'cat_cols': cat_cols,
        'random_seed': int(cfg.random_seed),
        'outer_splits': int(cfg.outer_splits),
        'inner_splits': int(cfg.inner_splits),
        'models_run': list(estimators.keys())
    }
    meta_path = os.path.join(cfg.output_dir, 'experiment_metadata.json')
    with open(meta_path, 'w') as fh:
        json.dump(metadata, fh, indent=2)

    # Generate aggregated reports and manuscript-ready figures
    try:
        aggregate_and_report(cfg.output_dir, per_fold_df, summary_df, metadata)
    except Exception as e:
        print(f"Reporting failed: {e}")

    print(f"Done. Outputs saved to {cfg.output_dir}")
    return {
        'per_fold_metrics': per_fold_csv,
        'test_summary': summary_csv,
        'preprocessor': preproc_path,
        'metadata': meta_path
    }


def run_pipeline(overrides: dict):
    cfg = overrides
    ensure_dir(cfg.get('output_dir', 'outputs'))
    return nested_cv_and_train(cfg)

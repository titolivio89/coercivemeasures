"""Training logic: nested cross-validation, training, metric aggregation, and saving artifacts."""
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, roc_curve, confusion_matrix
from sklearn.calibration import calibration_curve

from .data import load_data
from .preprocessing import build_preprocessor
from .models import get_base_estimators, get_param_grids
from .evaluation import save_roc_curve, save_calibration_curve, save_confusion_matrix
from .explain import explain_model_shap
from .utils import ensure_dir


def nested_cv_and_train(cfg):
    X, y = load_data(cfg.data_path, cfg.target_col, cfg.id_col)
    preproc, num_cols, cat_cols = build_preprocessor(X)

    estimators = get_base_estimators()
    param_grids = get_param_grids()

    results = []
    models_dir = os.path.join(cfg.output_dir, "models")
    ensure_dir(models_dir)

    outer_cv = StratifiedKFold(n_splits=cfg.outer_splits, shuffle=True, random_state=cfg.random_seed)
    fold_idx = 0
    for train_idx, test_idx in outer_cv.split(X, y):
        fold_idx += 1
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        for name, base_est in estimators.items():
            print(f"Outer fold {fold_idx} - training model: {name}")
            pipe = Pipeline([
                ("preproc", preproc),
                ("clf", base_est)
            ])
            grid = param_grids.get(name, {})
            inner_cv = StratifiedKFold(n_splits=cfg.inner_splits, shuffle=True, random_state=cfg.random_seed)
            gs = GridSearchCV(pipe, grid, cv=inner_cv, scoring="roc_auc", n_jobs=cfg.n_jobs)
            gs.fit(X_train, y_train)

            best = gs.best_estimator_
            # predict probabilities if available
            try:
                y_prob = best.predict_proba(X_test)[:, 1]
            except Exception:
                # fallback: use decision_function and convert with logistic
                try:
                    scores = best.decision_function(X_test)
                    # rescale to 0-1
                    y_prob = (scores - scores.min()) / (scores.max() - scores.min())
                except Exception:
                    y_prob = np.zeros(len(X_test))

            y_pred = best.predict(X_test)
            roc = roc_auc_score(y_test, y_prob)
            bal = balanced_accuracy_score(y_test, y_pred)

            # save model
            model_path = os.path.join(models_dir, f"model_fold{fold_idx}_{name}.joblib")
            joblib.dump(best, model_path)

            # plots
            fig_roc = os.path.join(cfg.output_dir, f"roc_fold{fold_idx}_{name}.png")
            save_roc_curve(y_test, y_prob, fig_roc, title=f"ROC - fold {fold_idx} - {name}")

            fig_cal = os.path.join(cfg.output_dir, f"calibration_fold{fold_idx}_{name}.png")
            save_calibration_curve(y_test, y_prob, fig_cal, title=f"Calibration - fold {fold_idx} - {name}")

            fig_cm = os.path.join(cfg.output_dir, f"confusion_fold{fold_idx}_{name}.png")
            save_confusion_matrix(y_test, y_pred, fig_cm, title=f"Confusion - fold {fold_idx} - {name}")

            # SHAP explainability
            expl_path = os.path.join(cfg.output_dir, f"shap_fold{fold_idx}_{name}.png")
            try:
                explain_model_shap(best, X_test, expl_path)
            except Exception as e:
                print(f"SHAP explanation failed for {name} on fold {fold_idx}: {e}")

            results.append({
                "fold": fold_idx,
                "model": name,
                "roc_auc": float(roc),
                "balanced_acc": float(bal),
                "model_path": model_path
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(cfg.output_dir, "cv_results.csv"), index=False)
    return results_df


def run_pipeline(overrides: dict):
    # build config object
    from .config import default_config
    cfg = default_config(overrides)
    ensure_dir(cfg.output_dir)
    res = nested_cv_and_train(cfg)
    print("Done. Results saved to:", cfg.output_dir)
    return res

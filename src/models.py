"""
Model training, nested cross-validation for multiple estimators and hyperparameter search.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, roc_curve, precision_recall_curve
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from config import Config
from evaluation import plot_roc_curve, plot_calibration_curve, save_metrics, error_analysis


def _get_models(cfg: Config):
    models = {
        "logreg": (LogisticRegression(max_iter=1000, solver="saga", random_state=cfg.random_state),
                   {"C": np.logspace(-4, 4, 10), "penalty": ["l2"]}),
        "svm": (SVC(kernel="linear", probability=True, random_state=cfg.random_state),
                {"C": np.logspace(-3, 3, 10)}),
        "rf": (RandomForestClassifier(random_state=cfg.random_state),
               {"n_estimators": [100, 200], "max_depth": [None, 5, 10], "class_weight": [None, "balanced"]}),
        "xgb": (XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=cfg.random_state),
                {"n_estimators": [100, 200], "max_depth": [3, 6], "learning_rate": [0.01, 0.1]}),
        "cat": (CatBoostClassifier(verbose=0, random_state=cfg.random_state),
                {"iterations": [100, 200], "depth": [4, 6], "learning_rate": [0.01, 0.1]})
    }
    return models


def run_nested_cv_for_models(X, y, cfg: Config, df=None):
    outer_cv = StratifiedKFold(n_splits=cfg.n_splits_outer, shuffle=True, random_state=cfg.random_state)
    models = _get_models(cfg)

    results = []
    fold_idx = 0
    for train_idx, test_idx in outer_cv.split(X, y):
        fold_idx += 1
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        for name, (estimator, param_grid) in models.items():
            inner_cv = StratifiedKFold(n_splits=cfg.n_splits_inner, shuffle=True, random_state=cfg.random_state)
            search = RandomizedSearchCV(estimator, param_distributions=param_grid, n_iter=cfg.n_iter_search,
                                        cv=inner_cv, scoring="roc_auc", n_jobs=1, random_state=cfg.random_state)
            search.fit(X_train, y_train)
            best = search.best_estimator_
            # predict probabilities
            y_proba = best.predict_proba(X_test)[:, 1]
            y_pred = best.predict(X_test)

            roc = roc_auc_score(y_test, y_proba)
            bal = balanced_accuracy_score(y_test, y_pred)

            # save curves
            fig_roc = plot_roc_curve(y_test, y_proba, title=f"ROC {name} fold{fold_idx}")
            fig_cal = plot_calibration_curve(y_test, y_proba, title=f"Calibration {name} fold{fold_idx}")
            fig_roc.savefig(cfg.figures_dir / f"roc_{name}_fold{fold_idx}.png")
            fig_cal.savefig(cfg.figures_dir / f"cal_{name}_fold{fold_idx}.png")
            plt.close(fig_roc); plt.close(fig_cal)

            # SHAP explainability and error analysis
            try:
                import explainability
                explainability.save_shap_for_model(best, X_test, cfg, prefix=f"{name}_fold{fold_idx}")
            except Exception as e:
                print("SHAP failed for", name, e)

            # error analysis
            error_analysis(X_test, y_test, y_pred, cfg, prefix=f"{name}_fold{fold_idx}", df_test=(df.iloc[test_idx] if df is not None else None))

            results.append({"model": name, "fold": fold_idx, "roc_auc": float(roc), "balanced_accuracy": float(bal), "best_params": str(search.best_params_)})
    return pd.DataFrame(results)

"""Configuration helpers for the pipeline."""
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Config:
    data_path: str
    target_col: str
    id_col: str = None
    output_dir: str = "outputs"
    random_seed: int = 42
    outer_splits: int = 5
    inner_splits: int = 3
    n_jobs: int = 1
    models: Dict[str, Dict[str, Any]] = field(default_factory=dict)


def default_config(overrides: Dict[str, Any]):
    cfg = Config(
        data_path=overrides.get("data_path"),
        target_col=overrides.get("target_col"),
        id_col=overrides.get("id_col"),
        output_dir=overrides.get("output_dir", "outputs"),
        random_seed=overrides.get("random_seed", 42),
        outer_splits=overrides.get("outer_splits", 5),
        inner_splits=overrides.get("inner_splits", 3),
        n_jobs=overrides.get("n_jobs", 1),
    )

    # default model parameter grids (kept small for speed)
    cfg.models = {
        "logreg": {
            "estimator": {
                "class": "sklearn.linear_model.LogisticRegression",
                "params": {"solver": "liblinear", "max_iter": 1000}
            },
            "param_grid": {
                "clf__C": [0.01, 0.1, 1.0]
            }
        },
        "svm_linear": {
            "estimator": {
                "class": "sklearn.svm.LinearSVC",
                "params": {"max_iter": 10000}
            },
            "param_grid": {
                "clf__C": [0.01, 0.1, 1.0]
            }
        },
        "random_forest": {
            "estimator": {
                "class": "sklearn.ensemble.RandomForestClassifier",
                "params": {"n_jobs": 1, "random_state": cfg.random_seed}
            },
            "param_grid": {
                "clf__n_estimators": [100],
                "clf__max_depth": [None, 5]
            }
        },
        "xgboost": {
            "estimator": {
                "class": "xgboost.XGBClassifier",
                "params": {"use_label_encoder": False, "eval_metric": "logloss", "verbosity": 0, "random_state": cfg.random_seed}
            },
            "param_grid": {
                "clf__n_estimators": [50, 100],
                "clf__max_depth": [3, 6]
            }
        },
        "catboost": {
            "estimator": {
                "class": "catboost.CatBoostClassifier",
                "params": {"verbose": 0, "random_state": cfg.random_seed}
            },
            "param_grid": {
                "clf__iterations": [100],
                "clf__depth": [4,6]
            }
        }
    }
    return cfg

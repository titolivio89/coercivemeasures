"""Model definitions and param grids. We return sklearn-compatible estimators.
The estimators are wrapped in a pipeline with a preprocessor placeholder named 'preproc' (the training code will combine).
"""
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import catboost as cb
except Exception:
    cb = None


def get_base_estimators():
    estimators = {
        "logreg": LogisticRegression(solver="liblinear", max_iter=1000),
        "svm_linear": LinearSVC(max_iter=10000),
        "random_forest": RandomForestClassifier(n_jobs=1),
    }
    if xgb is not None:
        estimators["xgboost"] = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", verbosity=0)
    if cb is not None:
        estimators["catboost"] = cb.CatBoostClassifier(verbose=0)
    return estimators


def get_param_grids():
    grids = {
        "logreg": {"clf__C": [0.01, 0.1, 1.0]},
        "svm_linear": {"clf__C": [0.01, 0.1, 1.0]},
        "random_forest": {"clf__n_estimators": [100], "clf__max_depth": [None, 5]},
    }
    if xgb is not None:
        grids["xgboost"] = {"clf__n_estimators": [50, 100], "clf__max_depth": [3, 6]}
    if cb is not None:
        grids["catboost"] = {"clf__iterations": [100], "clf__depth": [4, 6]}
    return grids

"""Explainability utilities: SHAP for tree-based models and coefficient plots for linear models.
Generates publication-quality figures (high DPI, vector formats) suitable for manuscript inclusion.
"""
import os
from typing import List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid')
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'font.family': 'serif'
})

import shap


def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def get_feature_names(preprocessor, num_cols: List[str], cat_cols: List[str]) -> List[str]:
    """Reconstruct feature names after ColumnTransformer preprocessing.
    Assumes the ColumnTransformer created in src.preprocessing with transformers named 'num' and 'cat'
    where 'cat' pipeline ends with OneHotEncoder.
    """
    feature_names = []
    # numerical features keep their names
    feature_names.extend(num_cols)

    # categorical features: expand using categories_ from OneHotEncoder
    try:
        cat_transformer = preprocessor.named_transformers_.get('cat')
        # If pipeline: get the OneHotEncoder step
        if hasattr(cat_transformer, 'named_steps'):
            # expects step named 'ohe'
            ohe = cat_transformer.named_steps.get('ohe')
        else:
            ohe = cat_transformer
        if ohe is not None and hasattr(ohe, 'categories_'):
            for col, cats in zip(cat_cols, ohe.categories_):
                feature_names.extend([f"{col}__{str(c)}" for c in cats])
        else:
            # fallback: use original categorical column names
            feature_names.extend(cat_cols)
    except Exception:
        # fallback: return numeric + categorical simple names
        feature_names.extend(cat_cols)

    return feature_names


def explain_shap_tree(pipeline, preprocessor, num_cols, cat_cols, X: pd.DataFrame, out_prefix: str, max_display: int = 20):
    """Compute SHAP explanations for tree-based models.
    pipeline: sklearn Pipeline where the final step is the estimator (tree-based).
    preprocessor: fitted preprocessor used in the pipeline
    X: raw DataFrame (not transformed)
    out_prefix: path prefix (without extension) for saved figures
    """
    clf = None
    if hasattr(pipeline, 'named_steps') and 'clf' in pipeline.named_steps:
        clf = pipeline.named_steps['clf']
    else:
        clf = pipeline

    # transform X with preprocessor to get the features expected by the tree model
    try:
        X_trans = preprocessor.transform(X)
    except Exception:
        # if preprocessor cannot transform, try pipeline without clf
        try:
            X_trans = pipeline[:-1].transform(X)
        except Exception as e:
            raise RuntimeError(f"Preprocessor transform failed: {e}")

    feature_names = get_feature_names(preprocessor, num_cols, cat_cols)
    try:
        X_trans_df = pd.DataFrame(X_trans, columns=feature_names)
    except Exception:
        # if shapes mismatch or sparse matrix, convert differently
        X_trans_arr = X_trans.toarray() if hasattr(X_trans, 'toarray') else X_trans
        X_trans_df = pd.DataFrame(X_trans_arr, columns=feature_names[:X_trans_arr.shape[1]])

    _ensure_dir(out_prefix + '.png')

    # Use TreeExplainer when possible
    try:
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_trans_df)
        # shap_values can be list for multiclass; handle binary
        if isinstance(shap_values, list) and len(shap_values) > 1:
            sv = shap_values[1]
        else:
            sv = shap_values

        # summary (beeswarm)
        plt.figure(figsize=(8, 6))
        shap.summary_plot(sv, X_trans_df, plot_type='dot', show=False, max_display=max_display)
        plt.tight_layout()
        plt.savefig(out_prefix + '_shap_beeswarm.png', dpi=300)
        plt.savefig(out_prefix + '_shap_beeswarm.svg')
        plt.close()

        # bar plot (importance)
        plt.figure(figsize=(8, 6))
        shap.summary_plot(sv, X_trans_df, plot_type='bar', show=False, max_display=max_display)
        plt.tight_layout()
        plt.savefig(out_prefix + '_shap_bar.png', dpi=300)
        plt.savefig(out_prefix + '_shap_bar.svg')
        plt.close()

        # dependence plot for top features
        importances = np.abs(sv).mean(axis=0)
        top_idx = np.argsort(importances)[-min(max_display, len(importances)):][::-1]
        top_features = [feature_names[i] for i in top_idx]
        for feat in top_features[:5]:
            try:
                plt.figure(figsize=(6, 4))
                shap.dependence_plot(feat, sv, X_trans_df, show=False)
                plt.tight_layout()
                safe_feat = feat.replace('/', '_').replace(' ', '_')
                plt.savefig(f"{out_prefix}_shap_dependence_{safe_feat}.png", dpi=300)
                plt.savefig(f"{out_prefix}_shap_dependence_{safe_feat}.svg")
                plt.close()
            except Exception:
                continue

    except Exception as e:
        raise RuntimeError(f"SHAP TreeExplainer failed: {e}")


def explain_coefficients(pipeline, preprocessor, num_cols, cat_cols, X: pd.DataFrame, out_path: str, top_n: int = 20):
    """Create coefficient visualization for linear models (LogisticRegression, LinearSVC).
    pipeline: sklearn Pipeline ending with linear estimator having coef_ attribute
    preprocessor: fitted preprocessor used in the pipeline
    X: raw DataFrame (used only to infer feature names)
    out_path: path prefix (without extension)
    """
    clf = None
    if hasattr(pipeline, 'named_steps') and 'clf' in pipeline.named_steps:
        clf = pipeline.named_steps['clf']
    else:
        clf = pipeline

    if not hasattr(clf, 'coef_'):
        raise RuntimeError('Provided estimator does not expose coef_')

    feature_names = get_feature_names(preprocessor, num_cols, cat_cols)
    coef = clf.coef_
    # multiclass vs binary
    if coef.ndim == 2 and coef.shape[0] > 1:
        # for multiclass, we plot coefficients for each class vs rest for top features per class
        for i in range(coef.shape[0]):
            coefs = coef[i, :]
            _plot_coef_series(coefs, feature_names, f"{out_path}_coef_class{i}", top_n)
    else:
        coefs = coef.ravel()
        _plot_coef_series(coefs, feature_names, out_path, top_n)


def _plot_coef_series(coefs, feature_names, out_prefix, top_n=20):
    # create dataframe
    df = pd.DataFrame({'feature': feature_names[:len(coefs)], 'coef': coefs})
    df['abscoef'] = df['coef'].abs()
    df = df.sort_values('abscoef', ascending=False).head(top_n)
    df = df.sort_values('coef')  # for horizontal bar ordering

    plt.figure(figsize=(8, max(4, 0.25 * len(df))))
    colors = ['#d73027' if v < 0 else '#1a9850' for v in df['coef']]
    plt.barh(df['feature'], df['coef'], color=colors)
    plt.xlabel('Coefficient')
    plt.title('Top coefficient contributions')
    plt.tight_layout()
    _ensure_dir(out_prefix + '.png')
    plt.savefig(out_prefix + '.png', dpi=300)
    plt.savefig(out_prefix + '.svg')
    plt.close()


def explain_model(pipeline, preprocessor, num_cols, cat_cols, X: pd.DataFrame, out_prefix: str):
    """Convenience wrapper that selects the appropriate explainability method.
    Returns a dict with produced file paths.
    """
    produced = {}
    clf = pipeline.named_steps['clf'] if hasattr(pipeline, 'named_steps') and 'clf' in pipeline.named_steps else pipeline

    # detect tree-based by feature_importances_
    if hasattr(clf, 'feature_importances_') or clf.__class__.__name__.lower().startswith(('xgb', 'lgb', 'cat')):
        try:
            explain_shap_tree(pipeline, preprocessor, num_cols, cat_cols, X, out_prefix)
            produced['shap_beeswarm'] = out_prefix + '_shap_beeswarm.png'
            produced['shap_bar'] = out_prefix + '_shap_bar.png'
        except Exception as e:
            produced['shap_error'] = str(e)
    # detect linear models
    elif hasattr(clf, 'coef_'):
        try:
            explain_coefficients(pipeline, preprocessor, num_cols, cat_cols, X, out_prefix)
            produced['coefficients'] = out_prefix + '.png'
        except Exception as e:
            produced['coeff_error'] = str(e)
    else:
        produced['explain_error'] = 'Model type not recognized for explainability'

    return produced

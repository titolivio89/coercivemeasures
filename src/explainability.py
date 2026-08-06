"""
SHAP explainability helpers. Saves summary plots and feature importance.
"""
import shap
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd


def save_shap_for_model(model, X_test, cfg, prefix="model"):
    out_dir = Path(cfg.figures_dir)
    # convert X_test to dataframe if needed
    if not hasattr(X_test, "columns"):
        X = pd.DataFrame(X_test)
    else:
        X = X_test

    # choose appropriate explainer
    try:
        if hasattr(shap, 'TreeExplainer') and (model.__class__.__name__.lower().find('xgb')>=0 or model.__class__.__name__.lower().find('catboost')>=0 or model.__class__.__name__.lower().find('randomforest')>=0):
            explainer = shap.TreeExplainer(model)
        else:
            # LinearExplainer covers linear models
            explainer = shap.LinearExplainer(model, X, feature_perturbation="correlation_dependent")
    except Exception:
        explainer = shap.Explainer(model, X)

    shap_values = explainer(X)

    # summary plot
    plt.figure()
    try:
        shap.summary_plot(shap_values, X, show=False)
        plt.savefig(out_dir / f"{prefix}_shap_summary.png", bbox_inches='tight')
        plt.close()
    except Exception as e:
        print("Failed to create SHAP summary plot:", e)

    # bar plot
    plt.figure()
    try:
        shap.plots.bar(shap_values, show=False, max_display=20)
        plt.savefig(out_dir / f"{prefix}_shap_bar.png", bbox_inches='tight')
        plt.close()
    except Exception as e:
        print("Failed to create SHAP bar plot:", e)

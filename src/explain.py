"""SHAP explainability utilities."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import numpy as np
import pandas as pd


def explain_model_shap(model, X: pd.DataFrame, save_path: str, max_display: int = 20):
    # convert X to pandas DataFrame if needed
    if not isinstance(X, pd.DataFrame):
        X_df = pd.DataFrame(X)
    else:
        X_df = X

    # Create an explainer that tries to pick the best approach
    try:
        explainer = shap.Explainer(model, X_df)
        shap_values = explainer(X_df)
        plt.figure(figsize=(8, 6))
        shap.plots.bar(shap_values, max_display=max_display, show=False)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    except Exception:
        # fallback: use TreeExplainer if model has get_booster or trees
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_df)
            plt.figure(figsize=(8, 6))
            shap.summary_plot(shap_values, X_df, show=False)
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
        except Exception as e:
            raise RuntimeError(f"SHAP failed: {e}")

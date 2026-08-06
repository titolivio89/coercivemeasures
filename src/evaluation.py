"""
Evaluation utilities: plotting ROC, calibration, saving metrics and error analysis figures.
"""
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve
import pandas as pd
from pathlib import Path


def plot_roc_curve(y_true, y_score, title="ROC"):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0,1],[0,1], linestyle="--", color="grey")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()
    return fig


def plot_calibration_curve(y_true, y_prob, n_bins=10, title="Calibration"):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    fig, ax = plt.subplots()
    ax.plot(prob_pred, prob_true, marker='o')
    ax.plot([0,1],[0,1], linestyle='--', color='grey')
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.set_title(title)
    return fig


def save_metrics(df_metrics, cfg):
    cfg.results_path.parent.mkdir(parents=True, exist_ok=True)
    df_metrics.to_csv(cfg.results_path, index=False)


def save_overall_results(results_df, cfg):
    save_metrics(results_df, cfg)
    print(f"Saved results to {cfg.results_path}")


def error_analysis(X_test, y_test, y_pred, cfg, prefix="err", df_test=None):
    # confusion table and example indexing
    import pandas as pd
    out_dir = Path(cfg.figures_dir)
    df = pd.DataFrame(X_test.copy())
    df["y_true"] = y_test.values
    df["y_pred"] = y_pred
    df["correct"] = df["y_true"] == df["y_pred"]
    # save confusion counts
    counts = df["correct"].value_counts()
    counts.to_csv(out_dir / f"{prefix}_correct_counts.csv")
    # save top misclassified examples
    mis = df[df["correct"]==False]
    mis.head(20).to_csv(out_dir / f"{prefix}_misclassified_examples.csv", index=False)

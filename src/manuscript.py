"""Generate manuscript-ready tables and figures (Table 1, ROC curves, calibration plots, SHAP summaries, feature importance).
Saves outputs to <output_dir>/manuscript/figures and <output_dir>/manuscript/tables.
"""
import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, auc
import joblib

from .explain import explain_model, explain_coefficients, get_feature_names

sns.set(style='whitegrid')
plt.rcParams.update({'figure.dpi': 300, 'savefig.dpi': 300, 'font.family': 'serif'})


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def generate_table1(output_dir: str, per_fold_df: pd.DataFrame, summary_df: pd.DataFrame):
    tables_dir = os.path.join(output_dir, 'manuscript', 'tables')
    _ensure_dir(tables_dir)

    # aggregate CV metrics
    agg = per_fold_df.groupby('model').agg({'roc_auc': ['mean', 'std'], 'balanced_accuracy': ['mean', 'std']})
    agg.columns = ['roc_mean', 'roc_std', 'bal_mean', 'bal_std']
    agg = agg.reset_index()

    # merge with test summary
    merged = agg.merge(summary_df, on='model', how='left')

    # Keep relevant columns and format for manuscript
    out_cols = ['model', 'roc_mean', 'roc_std', 'test_roc_auc', 'bal_mean', 'bal_std', 'test_balanced_accuracy',
                'test_accuracy', 'test_precision', 'test_recall', 'test_f1']
    table1 = merged[out_cols]
    # format numeric columns to 3 decimals and produce mean (std) strings
    def fmt_mean_std(mean, std):
        if pd.isna(mean):
            return ''
        return f"{mean:.3f} ({std:.3f})"

    table1['CV ROC AUC (mean ± sd)'] = table1.apply(lambda r: fmt_mean_std(r['roc_mean'], r['roc_std']), axis=1)
    table1['Temporal test ROC AUC'] = table1['test_roc_auc'].round(3)
    table1['CV Balanced Acc (mean ± sd)'] = table1.apply(lambda r: fmt_mean_std(r['bal_mean'], r['bal_std']), axis=1)
    table1['Temporal test Balanced Acc'] = table1['test_balanced_accuracy'].round(3)

    table1_final = table1[['model', 'CV ROC AUC (mean ± sd)', 'Temporal test ROC AUC',
                           'CV Balanced Acc (mean ± sd)', 'Temporal test Balanced Acc',
                           'test_accuracy', 'test_precision', 'test_recall', 'test_f1']]

    csv_path = os.path.join(tables_dir, 'table1_model_comparison.csv')
    md_path = os.path.join(tables_dir, 'table1_model_comparison.md')
    table1_final.to_csv(csv_path, index=False)
    with open(md_path, 'w') as fh:
        fh.write(table1_final.to_markdown(index=False))
    return csv_path, md_path


def plot_combined_roc(output_dir: str, models_dir: str, preprocessor, num_cols, cat_cols, X_test, y_test, sample_n: int = 2000):
    figs_dir = os.path.join(output_dir, 'manuscript', 'figures')
    _ensure_dir(figs_dir)

    # sample test set for plotting
    if sample_n and len(X_test) > sample_n:
        idx = np.random.choice(len(X_test), sample_n, replace=False)
        Xs = X_test.iloc[idx].reset_index(drop=True)
        ys = y_test.iloc[idx].reset_index(drop=True)
    else:
        Xs = X_test.reset_index(drop=True)
        ys = y_test.reset_index(drop=True)

    plt.figure(figsize=(8, 6))
    for fn in os.listdir(models_dir):
        if not fn.startswith('final_') or not fn.endswith('.joblib'):
            continue
        model_name = fn.replace('final_', '').replace('.joblib', '')
        pipe = joblib.load(os.path.join(models_dir, fn))
        try:
            probs = pipe.predict_proba(Xs)[:, 1]
        except Exception:
            try:
                scores = pipe.decision_function(Xs)
                probs = (scores - scores.min()) / (scores.max() - scores.min()) if np.ptp(scores) != 0 else np.zeros_like(scores)
            except Exception:
                continue
        fpr, tpr, _ = roc_curve(ys, probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC={roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves on temporal test set')
    plt.legend(loc='lower right', fontsize='small')
    plt.tight_layout()
    png = os.path.join(figs_dir, 'roc_curves_temporal_test.png')
    svg = os.path.join(figs_dir, 'roc_curves_temporal_test.svg')
    plt.savefig(png, dpi=300)
    plt.savefig(svg)
    plt.close()
    return png, svg


def plot_combined_calibration(output_dir: str, models_dir: str, preprocessor, num_cols, cat_cols, X_test, y_test, sample_n: int = 2000, n_bins: int = 10):
    figs_dir = os.path.join(output_dir, 'manuscript', 'figures')
    _ensure_dir(figs_dir)

    # sample
    if sample_n and len(X_test) > sample_n:
        idx = np.random.choice(len(X_test), sample_n, replace=False)
        Xs = X_test.iloc[idx].reset_index(drop=True)
        ys = y_test.iloc[idx].reset_index(drop=True)
    else:
        Xs = X_test.reset_index(drop=True)
        ys = y_test.reset_index(drop=True)

    plt.figure(figsize=(8, 6))
    for fn in os.listdir(models_dir):
        if not fn.startswith('final_') or not fn.endswith('.joblib'):
            continue
        model_name = fn.replace('final_', '').replace('.joblib', '')
        pipe = joblib.load(os.path.join(models_dir, fn))
        try:
            probs = pipe.predict_proba(Xs)[:, 1]
        except Exception:
            try:
                scores = pipe.decision_function(Xs)
                probs = (scores - scores.min()) / (scores.max() - scores.min()) if np.ptp(scores) != 0 else np.zeros_like(scores)
            except Exception:
                continue
        prob_true, prob_pred = calibration_curve(ys, probs, n_bins=n_bins)
        plt.plot(prob_pred, prob_true, marker='o', lw=1.5, label=model_name)

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('Predicted probability')
    plt.ylabel('Observed proportion')
    plt.title('Calibration plots (temporal test)')
    plt.legend(loc='lower right', fontsize='small')
    plt.tight_layout()
    png = os.path.join(figs_dir, 'calibration_temporal_test.png')
    svg = os.path.join(figs_dir, 'calibration_temporal_test.svg')
    plt.savefig(png, dpi=300)
    plt.savefig(svg)
    plt.close()
    return png, svg


def generate_shap_and_feature_importance(output_dir: str, models_dir: str, preprocessor, num_cols, cat_cols, X_test, sample_n: int = 1000):
    figs_dir = os.path.join(output_dir, 'manuscript', 'figures')
    _ensure_dir(figs_dir)

    # sample
    if sample_n and len(X_test) > sample_n:
        idx = np.random.choice(len(X_test), sample_n, replace=False)
        Xs = X_test.iloc[idx].reset_index(drop=True)
    else:
        Xs = X_test.reset_index(drop=True)

    produced_files = []
    for fn in os.listdir(models_dir):
        if not fn.startswith('final_') or not fn.endswith('.joblib'):
            continue
        model_name = fn.replace('final_', '').replace('.joblib', '')
        pipe = joblib.load(os.path.join(models_dir, fn))
        out_prefix = os.path.join(figs_dir, f"{model_name}")
        try:
            # use explain_model wrapper which will detect tree vs linear
            res = explain_model(pipe, preprocessor, num_cols, cat_cols, Xs, out_prefix)
            # move produced files (they are saved next to out_prefix)
            for k, v in res.items():
                if isinstance(v, str) and os.path.exists(v):
                    produced_files.append(v)
        except Exception as e:
            # fallback: try coefficient plot for linear models
            try:
                explain_coefficients(pipe, preprocessor, num_cols, cat_cols, Xs, out_prefix)
                produced_files.append(out_prefix + '.png')
            except Exception:
                produced_files.append(f"explain_failure_{model_name}")
    return produced_files


def generate_manuscript_outputs(output_dir: str, per_fold_df: pd.DataFrame, summary_df: pd.DataFrame,
                                models_dir: str, preprocessor, num_cols, cat_cols, X_test, y_test):
    _ensure_dir(os.path.join(output_dir, 'manuscript', 'figures'))
    _ensure_dir(os.path.join(output_dir, 'manuscript', 'tables'))

    table_csv, table_md = generate_table1(output_dir, per_fold_df, summary_df)
    roc_png, roc_svg = plot_combined_roc(output_dir, models_dir, preprocessor, num_cols, cat_cols, X_test, y_test)
    cal_png, cal_svg = plot_combined_calibration(output_dir, models_dir, preprocessor, num_cols, cat_cols, X_test, y_test)
    shap_files = generate_shap_and_feature_importance(output_dir, models_dir, preprocessor, num_cols, cat_cols, X_test)

    return {
        'table_csv': table_csv,
        'table_md': table_md,
        'roc_png': roc_png,
        'roc_svg': roc_svg,
        'cal_png': cal_png,
        'cal_svg': cal_svg,
        'shap_files': shap_files
    }

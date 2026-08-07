#!/usr/bin/env python3
"""
Generate publication-ready tables from ablation/benchmark outputs.

Produces:
- Table 1: Model comparison (metrics with bootstrap CIs)
- Feature importance table: mean ROC AUC decrease when dropping each predictor
- Bootstrap confidence intervals table (per model, baseline and drop metrics)
- Calibration statistics per model (Brier score, calibration-in-the-large, ECE, slope/intercept)

Saves each table as CSV, Excel, and LaTeX in results/ and manuscript/.

This script will re-run nested CV to collect OOF predictions if needed for calibration
statistics using the same models and CV settings as the ablation script.
"""
from pathlib import Path
import sys
from pprint import pprint
import numpy as np
import pandas as pd
import warnings

# make src importable when run from repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from config import Config

# import helper functions from ablation (collect predictions + bootstrap)
try:
    from ablation import run_nested_cv_collect_predictions, bootstrap_ci, metric_roc_auc, metric_balanced_accuracy, metric_precision, metric_recall, metric_f1
except Exception:
    # If import fails, we'll reimplement minimal bootstrap here
    run_nested_cv_collect_predictions = None
    bootstrap_ci = None

from sklearn.metrics import brier_score_loss
from sklearn.linear_model import LinearRegression

RESULTS_DIR = Path("results")
MANUSCRIPT_DIR = Path("manuscript")
TABLES_DIR = RESULTS_DIR / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MANUSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def format_ci(mean, lower, upper, fmt="{:.3f}"):
    try:
        return f"{fmt.format(mean)} ({fmt.format(lower)}--{fmt.format(upper)})"
    except Exception:
        return "NA"


def load_ablation_results(path=RESULTS_DIR / "ablation_full_results.csv"):
    if not path.exists():
        raise FileNotFoundError(f"Expected ablation results at {path} — run src/ablation.py first")
    return pd.read_csv(path)


def make_model_comparison_table(ablation_df: pd.DataFrame):
    # extract baseline metrics per model (baseline_* columns present in each row)
    models = ablation_df["model"].unique()
    rows = []
    for m in models:
        # take first occurrence for baseline values
        sub = ablation_df[ablation_df["model"] == m].iloc[0]
        mean_roc = sub.get("baseline_roc_auc", np.nan)
        roc_lo = sub.get("baseline_roc_auc_ci_lower", np.nan)
        roc_hi = sub.get("baseline_roc_auc_ci_upper", np.nan)
        mean_bal = sub.get("baseline_balanced_accuracy", np.nan)
        bal_lo = sub.get("baseline_balanced_accuracy_ci_lower", np.nan)
        bal_hi = sub.get("baseline_balanced_accuracy_ci_upper", np.nan)
        mean_prec = sub.get("baseline_precision", np.nan)
        prec_lo = sub.get("baseline_precision_ci_lower", np.nan)
        prec_hi = sub.get("baseline_precision_ci_upper", np.nan)
        mean_rec = sub.get("baseline_recall", np.nan)
        rec_lo = sub.get("baseline_recall_ci_lower", np.nan)
        rec_hi = sub.get("baseline_recall_ci_upper", np.nan)
        mean_f1 = sub.get("baseline_f1", np.nan)
        f1_lo = sub.get("baseline_f1_ci_lower", np.nan)
        f1_hi = sub.get("baseline_f1_ci_upper", np.nan)

        rows.append({
            "model": m,
            "roc_auc": mean_roc,
            "roc_auc_ci": format_ci(mean_roc, roc_lo, roc_hi),
            "balanced_accuracy": mean_bal,
            "balanced_accuracy_ci": format_ci(mean_bal, bal_lo, bal_hi),
            "precision": mean_prec,
            "precision_ci": format_ci(mean_prec, prec_lo, prec_hi),
            "recall": mean_rec,
            "recall_ci": format_ci(mean_rec, rec_lo, rec_hi),
            "f1": mean_f1,
            "f1_ci": format_ci(mean_f1, f1_lo, f1_hi),
        })
    df = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    return df


def make_feature_importance_table(ablation_summary_path=RESULTS_DIR / "ablation_summary.csv"):
    if not ablation_summary_path.exists():
        raise FileNotFoundError(f"Expected ablation summary at {ablation_summary_path}")
    df = pd.read_csv(ablation_summary_path)
    # rename for clarity
    df = df.rename(columns={"roc_auc_delta_mean": "mean_roc_auc_decrease", "roc_auc_delta_std": "std_roc_auc_decrease"})
    df = df.sort_values("mean_roc_auc_decrease", ascending=False)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def make_bootstrap_ci_table(ablation_df: pd.DataFrame):
    # Produce a wide table with baseline and drop metrics (with CIs) per model and predictor
    rows = []
    for _, r in ablation_df.iterrows():
        model = r["model"]
        pred = r["dropped_predictor"]
        row = {
            "model": model,
            "dropped_predictor": pred,
            "baseline_roc_auc": r.get("baseline_roc_auc"),
            "baseline_roc_auc_ci": format_ci(r.get("baseline_roc_auc"), r.get("baseline_roc_auc_ci_lower"), r.get("baseline_roc_auc_ci_upper")),
            "drop_roc_auc": r.get("drop_roc_auc"),
            "drop_roc_auc_ci": format_ci(r.get("drop_roc_auc"), r.get("drop_roc_auc_ci_lower"), r.get("drop_roc_auc_ci_upper")),
            "roc_auc_delta": r.get("roc_auc_delta"),

            "baseline_balanced_accuracy": r.get("baseline_balanced_accuracy"),
            "baseline_balanced_accuracy_ci": format_ci(r.get("baseline_balanced_accuracy"), r.get("baseline_balanced_accuracy_ci_lower"), r.get("baseline_balanced_accuracy_ci_upper")),
            "drop_balanced_accuracy": r.get("drop_balanced_accuracy"),
            "drop_balanced_accuracy_ci": format_ci(r.get("drop_balanced_accuracy"), r.get("drop_balanced_accuracy_ci_lower"), r.get("drop_balanced_accuracy_ci_upper")),
            "balanced_accuracy_delta": r.get("balanced_accuracy_delta"),
        }
        rows.append(row)
    wide = pd.DataFrame(rows)
    # pivot so that predictors form columns for deltas? We'll return both long and pivot
    pivot_roc = wide.pivot(index="model", columns="dropped_predictor", values="roc_auc_delta")
    return wide, pivot_roc


def compute_calibration_stats(cfg: Config, n_bootstraps=1000):
    # collect OOF predictions using run_nested_cv_collect_predictions from ablation
    if run_nested_cv_collect_predictions is None:
        raise RuntimeError("run_nested_cv_collect_predictions not available (import failed). Run src/ablation.py to generate OOF predictions first.")

    # load data and predictors as used in ablation
    predictors = ["aggression", "police", "psychosis", "suicidality", "akute_intoxikation"]
    X, y, df = load_data(cfg)
    if isinstance(X, pd.DataFrame) and set(predictors).issubset(X.columns):
        X_raw = X[predictors].copy()
    elif df is not None and set(predictors).issubset(df.columns):
        X_raw = df[predictors].copy()
    else:
        X_raw = pd.DataFrame(X)

    preproc = build_preprocessing_pipeline(cfg)
    X_processed = preproc.fit_transform(X_raw)
    X_feat = add_feature_pipeline(X_processed, cfg)

    preds = run_nested_cv_collect_predictions(pd.DataFrame(X_feat), pd.Series(y).reset_index(drop=True), cfg)

    cal_rows = []
    for model, data in preds.items():
        y_true = data["y_true"]
        y_prob = data["y_proba"]
        y_pred = data["y_pred"]
        # Brier score
        try:
            brier = float(brier_score_loss(y_true, y_prob))
        except Exception:
            brier = np.nan
        # calibration-in-the-large: mean predicted - mean observed
        try:
            calib_in_the_large = float(np.mean(y_prob) - np.mean(y_true))
        except Exception:
            calib_in_the_large = np.nan
        # ECE
        try:
            n_bins = 10
            bins = np.linspace(0.0, 1.0, n_bins + 1)
            inds = np.digitize(y_prob, bins) - 1
            ece = 0.0
            for b in range(n_bins):
                mask = inds == b
                if mask.sum() == 0:
                    continue
                prop = mask.sum() / len(y_true)
                avg_pred = y_prob[mask].mean()
                avg_true = y_true[mask].mean()
                ece += prop * abs(avg_pred - avg_true)
            ece = float(ece)
        except Exception:
            ece = np.nan
        # calibration slope/intercept via linear regression on logit(p)
        try:
            p = np.clip(y_prob, 1e-6, 1 - 1e-6)
            logit = np.log(p / (1 - p)).reshape(-1, 1)
            lr = LinearRegression().fit(logit, y_true)
            slope = float(lr.coef_[0])
            intercept = float(lr.intercept_)
        except Exception:
            slope = np.nan
            intercept = np.nan

        # bootstrap CIs for Brier and ECE
        brier_lo, brier_hi = (np.nan, np.nan)
        ece_lo, ece_hi = (np.nan, np.nan)
        if bootstrap_ci is not None:
            # define wrappers
            def metric_brier(yt, yprob, ypred):
                try:
                    return float(brier_score_loss(yt, yprob))
                except Exception:
                    return np.nan

            def metric_ece(yt, yprob, ypred):
                try:
                    n_bins = 10
                    bins = np.linspace(0.0, 1.0, n_bins + 1)
                    inds = np.digitize(yprob, bins) - 1
                    e = 0.0
                    for b in range(n_bins):
                        mask = inds == b
                        if mask.sum() == 0:
                            continue
                        prop = mask.sum() / len(yt)
                        avg_pred = yprob[mask].mean()
                        avg_true = yt[mask].mean()
                        e += prop * abs(avg_pred - avg_true)
                    return float(e)
                except Exception:
                    return np.nan

            brier_lo, brier_hi = bootstrap_ci(y_true, y_prob, y_pred, metric_brier, n_bootstraps=n_bootstraps, random_state=cfg.random_state)
            ece_lo, ece_hi = bootstrap_ci(y_true, y_prob, y_pred, metric_ece, n_bootstraps=n_bootstraps, random_state=cfg.random_state)

        cal_rows.append({
            "model": model,
            "brier_score": brier,
            "brier_ci": format_ci(brier, brier_lo, brier_hi),
            "calibration_in_the_large": calib_in_the_large,
            "ece": ece,
            "ece_ci": format_ci(ece, ece_lo, ece_hi),
            "calibration_slope": slope,
            "calibration_intercept": intercept,
        })

    cal_df = pd.DataFrame(cal_rows).sort_values("brier_score")
    return cal_df


def save_table_variants(df: pd.DataFrame, name: str):
    # CSV
    csv_path = TABLES_DIR / f"{name}.csv"
    df.to_csv(csv_path, index=False)
    (MANUSCRIPT_DIR / f"{name}.csv").write_text(csv_path.read_text())

    # Excel
    xlsx_path = TABLES_DIR / f"{name}.xlsx"
    with pd.ExcelWriter(xlsx_path) as writer:
        df.to_excel(writer, sheet_name=name[:31], index=False)
    # copy to manuscript
    (MANUSCRIPT_DIR / f"{name}.xlsx").write_bytes(xlsx_path.read_bytes())

    # LaTeX
    tex_path = TABLES_DIR / f"{name}.tex"
    try:
        tex = df.to_latex(index=False, float_format="{:.3f}".format)
        tex_path.write_text(tex)
        (MANUSCRIPT_DIR / f"{name}.tex").write_text(tex)
    except Exception as e:
        warnings.warn(f"Failed to write LaTeX for {name}: {e}")

    print(f"Saved table {name} to {csv_path}, {xlsx_path}, and {tex_path}")


def main():
    cfg = Config()
    # load ablation results
    ablation_df = load_ablation_results()

    # Table 1: model comparison
    model_comp = make_model_comparison_table(ablation_df)
    save_table_variants(model_comp, "table1_model_comparison")

    # Feature importance table
    feat_imp = make_feature_importance_table()
    save_table_variants(feat_imp, "table2_feature_importance")

    # Bootstrap CI table (detailed)
    wide, pivot_roc = make_bootstrap_ci_table(ablation_df)
    save_table_variants(wide, "table3_bootstrap_cis_long")
    # also save pivoted ROC deltas as table
    pivot_roc.reset_index().to_csv(TABLES_DIR / "table3_roc_delta_pivot.csv", index=False)
    (MANUSCRIPT_DIR / "table3_roc_delta_pivot.csv").write_text((TABLES_DIR / "table3_roc_delta_pivot.csv").read_text())
    try:
        with pd.ExcelWriter(TABLES_DIR / "table3_roc_delta_pivot.xlsx") as writer:
            pivot_roc.to_excel(writer, sheet_name="roc_delta_pivot")
        (MANUSCRIPT_DIR / "table3_roc_delta_pivot.xlsx").write_bytes((TABLES_DIR / "table3_roc_delta_pivot.xlsx").read_bytes())
    except Exception:
        pass

    # Calibration statistics (this will rerun nested CV to collect OOF predictions if necessary)
    try:
        cal_df = compute_calibration_stats(cfg, n_bootstraps=1000)
        save_table_variants(cal_df, "table4_calibration_statistics")
    except Exception as e:
        warnings.warn(f"Calibration statistics generation failed: {e}")

    print("All tables generated and saved to results/tables/ and manuscript/")


if __name__ == '__main__':
    main()

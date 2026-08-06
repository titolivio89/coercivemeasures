"""Clinical error analysis utilities: generate FP/FN CSVs, patient-level review tables,
and visualizations comparing misclassified vs correctly classified patients.
"""
import os
import json
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import joblib
import shap

sns.set(style='whitegrid')
plt.rcParams.update({'figure.dpi': 300, 'savefig.dpi': 300})


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _safe_name(s: str) -> str:
    return str(s).replace('/', '_').replace(' ', '_')


def saved_final_models(models_dir: str) -> List[str]:
    files = [f for f in os.listdir(models_dir) if f.startswith('final_') and f.endswith('.joblib')]
    return files


def compute_standardized_mean_diff(a: pd.Series, b: pd.Series) -> float:
    # pooled standard deviation
    na = a.dropna()
    nb = b.dropna()
    if len(na) + len(nb) < 2:
        return 0.0
    sa = na.std(ddof=1)
    sb = nb.std(ddof=1)
    pooled = np.sqrt(((len(na)-1)*sa*sa + (len(nb)-1)*sb*sb) / max(len(na)+len(nb)-2, 1))
    if pooled == 0:
        return 0.0
    return (na.mean() - nb.mean()) / pooled


def aggregate_errors(output_dir: str, X_test: pd.DataFrame, y_test: pd.Series, ids_test: Optional[pd.Series],
                     preprocessor, num_cols: List[str], cat_cols: List[str], models_dir: str = None,
                     top_features: int = 10, shap_sample: int = 1000):
    """Analyze misclassifications for final models saved in models_dir and export CSVs and figures.
    Exports:
      - outputs/error_analysis/{model}_false_positives.csv
      - outputs/error_analysis/{model}_false_negatives.csv
      - outputs/error_analysis/{model}_patient_review.csv
      - outputs/error_analysis/{model}_feature_smd_fp_vs_tn.png (and fn_vs_tp)
      - outputs/error_analysis/confusion_summary.csv
    """
    if models_dir is None:
        models_dir = os.path.join(output_dir, 'models')
    ea_dir = os.path.join(output_dir, 'error_analysis')
    _ensure_dir(ea_dir)

    # collect confusion summary
    confusion_records = []
    patient_reviews = []

    model_files = saved_final_models(models_dir)

    for mf in model_files:
        model_name = mf.replace('final_', '').replace('.joblib', '')
        model_path = os.path.join(models_dir, mf)
        try:
            pipe = joblib.load(model_path)
        except Exception as e:
            print(f"Failed to load {model_path}: {e}")
            continue

        # predict
        try:
            y_prob = pipe.predict_proba(X_test)[:, 1]
        except Exception:
            try:
                scores = pipe.decision_function(X_test)
                y_prob = (scores - scores.min()) / (scores.max() - scores.min()) if np.ptp(scores) != 0 else np.zeros_like(scores)
            except Exception:
                y_prob = np.zeros(len(X_test))
        y_pred = pipe.predict(X_test)

        # build patient dataframe
        df_pat = X_test.copy().reset_index(drop=True)
        df_pat['true_label'] = y_test.reset_index(drop=True)
        df_pat['pred_label'] = y_pred
        df_pat['pred_prob'] = y_prob
        if ids_test is not None:
            df_pat['patient_id'] = ids_test.reset_index(drop=True)
        else:
            df_pat['patient_id'] = df_pat.index.astype(str)

        # misclassifications
        fp = df_pat[(df_pat['true_label'] == 0) & (df_pat['pred_label'] == 1)].copy()
        fn = df_pat[(df_pat['true_label'] == 1) & (df_pat['pred_label'] == 0)].copy()
        tp = df_pat[(df_pat['true_label'] == 1) & (df_pat['pred_label'] == 1)].copy()
        tn = df_pat[(df_pat['true_label'] == 0) & (df_pat['pred_label'] == 0)].copy()

        # save CSVs
        fp_csv = os.path.join(ea_dir, f"{model_name}_false_positives.csv")
        fn_csv = os.path.join(ea_dir, f"{model_name}_false_negatives.csv")
        review_csv = os.path.join(ea_dir, f"{model_name}_patient_review.csv")
        fp.to_csv(fp_csv, index=False)
        fn.to_csv(fn_csv, index=False)

        # Add patient review entries
        for _, row in df_pat.iterrows():
            patient_reviews.append({
                'model': model_name,
                'patient_id': row['patient_id'],
                'true_label': int(row['true_label']),
                'pred_label': int(row['pred_label']),
                'pred_prob': float(row['pred_prob'])
            })

        # confusion summary
        confusion_records.append({
            'model': model_name,
            'n_tp': int(len(tp)),
            'n_tn': int(len(tn)),
            'n_fp': int(len(fp)),
            'n_fn': int(len(fn)),
            'n_total': int(len(df_pat))
        })

        # Feature-level comparison using standardized mean differences for numeric features
        try:
            smd_fp_vs_tn = {}
            for col in num_cols:
                a = fp[col] if col in fp.columns else pd.Series(dtype=float)
                b = tn[col] if col in tn.columns else pd.Series(dtype=float)
                smd_fp_vs_tn[col] = compute_standardized_mean_diff(a, b)
            smd_series = pd.Series(smd_fp_vs_tn).abs().sort_values(ascending=False).head(top_features)

            plt.figure(figsize=(8, 4))
            smd_series.sort_values().plot(kind='barh', color='C0')
            plt.title(f'Top {top_features} standardized mean differences (FP vs TN) - {model_name}')
            plt.xlabel('|Standardized Mean Difference|')
            plt.tight_layout()
            plt.savefig(os.path.join(ea_dir, f'{_safe_name(model_name)}_smd_fp_vs_tn.png'))
            plt.savefig(os.path.join(ea_dir, f'{_safe_name(model_name)}_smd_fp_vs_tn.svg'))
            plt.close()
        except Exception as e:
            print(f"SMD FP vs TN failed for {model_name}: {e}")

        try:
            smd_fn_vs_tp = {}
            for col in num_cols:
                a = fn[col] if col in fn.columns else pd.Series(dtype=float)
                b = tp[col] if col in tp.columns else pd.Series(dtype=float)
                smd_fn_vs_tp[col] = compute_standardized_mean_diff(a, b)
            smd_series2 = pd.Series(smd_fn_vs_tp).abs().sort_values(ascending=False).head(top_features)

            plt.figure(figsize=(8, 4))
            smd_series2.sort_values().plot(kind='barh', color='C1')
            plt.title(f'Top {top_features} standardized mean differences (FN vs TP) - {model_name}')
            plt.xlabel('|Standardized Mean Difference|')
            plt.tight_layout()
            plt.savefig(os.path.join(ea_dir, f'{_safe_name(model_name)}_smd_fn_vs_tp.png'))
            plt.savefig(os.path.join(ea_dir, f'{_safe_name(model_name)}_smd_fn_vs_tp.svg'))
            plt.close()
        except Exception as e:
            print(f"SMD FN vs TP failed for {model_name}: {e}")

        # Top categorical differences: difference in proportions for category levels (for each cat col)
        try:
            cat_diffs = []
            for col in cat_cols:
                if col not in df_pat.columns:
                    continue
                fp_counts = fp[col].value_counts(normalize=True)
                tn_counts = tn[col].value_counts(normalize=True)
                for level in set(fp[col].dropna().unique()).union(set(tn[col].dropna().unique())):
                    p_fp = fp_counts.get(level, 0)
                    p_tn = tn_counts.get(level, 0)
                    cat_diffs.append({'feature': col, 'level': level, 'abs_diff': abs(p_fp - p_tn), 'p_fp': p_fp, 'p_tn': p_tn})
            if cat_diffs:
                cat_df = pd.DataFrame(cat_diffs).sort_values('abs_diff', ascending=False).head(top_features)
                cat_df.to_csv(os.path.join(ea_dir, f'{_safe_name(model_name)}_cat_level_diffs_fp_vs_tn.csv'), index=False)
        except Exception as e:
            print(f"Categorical diffs failed for {model_name}: {e}")

        # Patient-level top feature contributions using SHAP if available, otherwise simple coefficient*value for linear models
        try:
            # prepare transformed features and names
            try:
                # try to get preprocessor from pipeline
                pp = preprocessor
                feat_names = []
                if pp is not None:
                    # reconstruct names for transformed X
                    from .explain import get_feature_names
                    feat_names = get_feature_names(pp, num_cols, cat_cols)
                    X_trans = pp.transform(X_test)
                    X_trans_arr = X_trans.toarray() if hasattr(X_trans, 'toarray') else X_trans
                    X_trans_df = pd.DataFrame(X_trans_arr, columns=feat_names[:X_trans_arr.shape[1]])
                else:
                    X_trans_df = X_test.copy()

                # SHAP for tree models
                clf = pipe.named_steps['clf'] if hasattr(pipe, 'named_steps') and 'clf' in pipe.named_steps else pipe
                if hasattr(clf, 'feature_importances_'):
                    explainer = shap.TreeExplainer(clf)
                    shap_values = explainer.shap_values(X_trans_df)
                    # get class 1 shap
                    if isinstance(shap_values, list) and len(shap_values) > 1:
                        sv = shap_values[1]
                    else:
                        sv = shap_values
                    # for each misclassified patient, compute top 3 features
                    for df_row in pd.concat([fp, fn]).itertuples():
                        idx = int(df_row.Index) if hasattr(df_row, 'Index') else None
                        # map original index to row in X_test reset index: we reset earlier
                        # We'll find by patient_id
                        pid = df_row['patient_id'] if 'patient_id' in df_row._fields else df_row.patient_id
                        # find row in X_trans_df by matching patient_id column in original df
                        # we added patient_id as column in df_pat; so find index
                        try:
                            match_idx = df_pat[df_pat['patient_id'] == pid].index[0]
                        except Exception:
                            match_idx = None
                        if match_idx is None:
                            continue
                        contribs = pd.Series(sv[match_idx], index=X_trans_df.columns).abs().sort_values(ascending=False).head(3)
                        top_feats = ','.join([f"{f}:{float(X_trans_df.iloc[match_idx][f]):.3g}" for f in contribs.index])
                        # add to patient_reviews
                        patient_reviews.append({
                            'model': model_name,
                            'patient_id': pid,
                            'true_label': int(df_row.true_label),
                            'pred_label': int(df_row.pred_label),
                            'pred_prob': float(df_row.pred_prob),
                            'top_features': top_feats
                        })
                elif hasattr(clf, 'coef_'):
                    # linear model: contributions = coef * feature_value
                    coefs = clf.coef_.ravel()
                    for df_row in pd.concat([fp, fn]).itertuples():
                        pid = df_row.patient_id
                        try:
                            match_idx = df_pat[df_pat['patient_id'] == pid].index[0]
                        except Exception:
                            match_idx = None
                        if match_idx is None:
                            continue
                        # use X_trans_df if available, else original X_test values
                        vals = X_trans_df.iloc[match_idx] if 'X_trans_df' in locals() else X_test.iloc[match_idx]
                        # align length
                        fvals = vals.values[:len(coefs)]
                        contribs = pd.Series(np.abs(coefs * fvals), index=(feat_names[:len(coefs)] if feat_names else X_test.columns[:len(coefs)])).sort_values(ascending=False).head(3)
                        top_feats = ','.join([f"{f}:{float(vals[f]):.3g}" for f in contribs.index])
                        patient_reviews.append({
                            'model': model_name,
                            'patient_id': pid,
                            'true_label': int(df_row.true_label),
                            'pred_label': int(df_row.pred_label),
                            'pred_prob': float(df_row.pred_prob),
                            'top_features': top_feats
                        })
        except Exception as e:
            print(f"Per-patient feature contributions failed for {model_name}: {e}")

    # save confusion summary
    confusion_df = pd.DataFrame(confusion_records)
    confusion_df.to_csv(os.path.join(ea_dir, 'confusion_summary.csv'), index=False)

    # save aggregated patient review table
    if patient_reviews:
        patient_df = pd.DataFrame(patient_reviews).drop_duplicates()
        patient_df.to_csv(os.path.join(ea_dir, 'patient_level_review.csv'), index=False)

    return True

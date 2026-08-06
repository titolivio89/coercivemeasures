"""Reporting and figure generation for TRIPOD-AI compliance.
Provides aggregated metrics tables, model comparison tables, manuscript-ready figures,
exports CSVs for all metrics, and temporal vs CV comparison figures.
"""
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid')


def aggregate_and_report(output_dir: str, per_fold_df: pd.DataFrame, summary_df: pd.DataFrame, metadata: dict):
    ensure_dir = lambda p: os.makedirs(p, exist_ok=True)
    plots_dir = os.path.join(output_dir, 'figures')
    ensure_dir(plots_dir)

    # Save a model comparison table: mean and std for key metrics across folds
    agg = per_fold_df.groupby('model').agg({
        'roc_auc': ['mean', 'std'],
        'balanced_accuracy': ['mean', 'std'],
        'accuracy': ['mean', 'std'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'f1': ['mean', 'std']
    })
    # flatten columns
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()
    cmp_table_path = os.path.join(output_dir, 'model_comparison_table.csv')
    agg.to_csv(cmp_table_path, index=False)

    # Export per-fold metrics already saved; ensure also per-model boxplots for metrics
    metrics = ['roc_auc', 'balanced_accuracy', 'accuracy', 'f1']
    for m in metrics:
        plt.figure(figsize=(8, 5))
        sns.boxplot(x='model', y=m, data=per_fold_df, palette='Set2')
        plt.title(f'Model comparison - {m}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'model_comparison_{m}.png'))
        plt.close()

    # Test-set performance comparison (bar chart)
    if not summary_df.empty:
        df = summary_df.copy()
        df_plot = df[['model', 'test_roc_auc', 'test_balanced_accuracy']].set_index('model')
        df_plot = df_plot.sort_values(by='test_roc_auc', ascending=False)
        df_plot.plot(kind='bar', figsize=(8,5))
        plt.ylabel('Score')
        plt.title('Test-set performance comparison (ROC AUC and Balanced Accuracy)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'test_performance_comparison.png'))
        plt.close()

    # Create manuscript-ready tables (CSV and simple markdown)
    md_path = os.path.join(output_dir, 'manuscript_tables.md')
    with open(md_path, 'w') as fh:
        fh.write('# Model Comparison Table\n\n')
        fh.write(agg.to_markdown(index=False))
        fh.write('\n\n')
        fh.write('## Experiment metadata\n')
        for k, v in metadata.items():
            fh.write(f'- **{k}**: {v}\\n')

    return True


def temporal_vs_cv_plot(output_dir: str, temporal_df: pd.DataFrame):
    """Create publication-ready comparison figures between temporal validation and standard CV.
    temporal_df must contain columns: model, cv_mean_roc, cv_std_roc, test_roc, cv_mean_bal, cv_std_bal, test_bal
    """
    plots_dir = os.path.join(output_dir, 'figures')
    os.makedirs(plots_dir, exist_ok=True)

    df = temporal_df.copy()
    df = df.set_index('model')

    # ROC comparison: error bars for CV, point for test
    plt.figure(figsize=(8, 5))
    x = range(len(df))
    plt.errorbar(x, df['cv_mean_roc'], yerr=df['cv_std_roc'], fmt='o', label='CV (mean ± std)', color='C0')
    plt.scatter(x, df['test_roc'], marker='s', label='Temporal test', color='C1')
    plt.xticks(x, df.index, rotation=45)
    plt.ylabel('ROC AUC')
    plt.title('Temporal validation vs standard CV (ROC AUC)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'temporal_vs_cv_roc.png'))
    plt.savefig(os.path.join(plots_dir, 'temporal_vs_cv_roc.svg'))
    plt.close()

    # Balanced accuracy comparison
    plt.figure(figsize=(8,5))
    plt.errorbar(x, df['cv_mean_bal'], yerr=df['cv_std_bal'], fmt='o', label='CV (mean ± std)', color='C0')
    plt.scatter(x, df['test_bal'], marker='s', label='Temporal test', color='C1')
    plt.xticks(x, df.index, rotation=45)
    plt.ylabel('Balanced Accuracy')
    plt.title('Temporal validation vs standard CV (Balanced Accuracy)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'temporal_vs_cv_balanced_accuracy.png'))
    plt.savefig(os.path.join(plots_dir, 'temporal_vs_cv_balanced_accuracy.svg'))
    plt.close()

    # Save temporal_df to CSV
    temporal_df.to_csv(os.path.join(output_dir, 'temporal_vs_cv_comparison.csv'), index=False)
    return True

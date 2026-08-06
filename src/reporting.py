"""Reporting and figure generation for TRIPOD-AI compliance.
Provides aggregated metrics tables, model comparison tables, manuscript-ready figures,
and exports CSVs for all metrics.
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

    # Combined ROC curves on test set (if summary_df has roc_curve paths)
    # Try to plot the saved ROC images as a mosaic or overlay if possible by reading ROC data; but we have only images
    # Instead, create a bar chart of test ROC and balanced accuracy
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

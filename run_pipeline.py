"""
Main entry point for the Coercive Measures prediction pipeline.

Workflow
--------
1. Load clinical dataset
2. Preprocess features
3. Feature engineering
4. Nested cross-validation
5. Save results
"""

from pathlib import Path
import sys

# ---------------------------------------------------------------------
# Make current directory importable
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------

from config import Config
from data_loader import load_data
from preprocessing import build_preprocessing_pipeline
from feature_engineering import add_feature_pipeline
from models import run_nested_cv_for_models
from evaluation import save_overall_results


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    print("=" * 70)
    print("Coercive Measures Prediction Pipeline")
    print("=" * 70)

    cfg = Config()

    print("\nLoading data...")
    X, y, df = load_data(cfg)

    print(f"Patients: {len(df)}")
    print(f"Predictors: {X.shape[1]}")

    print("\nOutcome distribution:")
    print(y.value_counts())

    # -------------------------------------------------------------
    # preprocessing
    # -------------------------------------------------------------

    print("\nPreprocessing...")

    preprocessor = build_preprocessing_pipeline(cfg)

    X_processed = preprocessor.fit_transform(X)

    # -------------------------------------------------------------
    # feature engineering
    # -------------------------------------------------------------

    print("Feature engineering...")

    X_features = add_feature_pipeline(
        X_processed,
        cfg
    )

    print(f"Final feature matrix: {X_features.shape}")

    # -------------------------------------------------------------
    # nested CV
    # -------------------------------------------------------------

    print("\nRunning nested cross-validation...")

    results = run_nested_cv_for_models(
        X_features,
        y,
        cfg,
        df
    )

    # -------------------------------------------------------------
    # save results
    # -------------------------------------------------------------

    print("\nSaving results...")

    save_overall_results(
        results,
        cfg
    )

    print("\n")
    print("=" * 70)
    print("Pipeline finished successfully.")
    print("=" * 70)

    print("\nMean performance:\n")

    print(
        results.groupby("model")[
            [
                "roc_auc",
                "balanced_accuracy",
                "precision",
                "recall",
                "specificity",
                "f1",
                "mcc",
                "brier"
            ]
        ].mean().round(3)
    )


# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()

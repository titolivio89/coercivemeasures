# Reproducible ML pipeline entrypoint

import sys
from pathlib import Path

# make src importable
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from config import Config
from data_loader import load_data
from preprocessing import build_preprocessing_pipeline
from feature_engineering import add_feature_pipeline
from models import run_nested_cv_for_models
from evaluation import save_overall_results


def main():
    cfg = Config()

    X, y, df = load_data(cfg)

    preproc = build_preprocessing_pipeline(cfg)
    X_processed = preproc.fit_transform(X)

    X_feat = add_feature_pipeline(X_processed, cfg)

    results = run_nested_cv_for_models(X_feat, y, cfg, df)

    save_overall_results(results, cfg)


if __name__ == '__main__':
    main()

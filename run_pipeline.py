#!/usr/bin/env python3
"""Top-level entrypoint to run the TRIPOD-compliant ML pipeline.
Usage: python run_pipeline.py --data data.csv --target target_column --output outputs/
"""
import argparse
import os
from src.training import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Run TRIPOD-AI compliant ML pipeline")
    parser.add_argument("--data", required=True, help="Path to CSV dataset")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--id", default=None, help="ID column to ignore (optional)")
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--outer_splits", type=int, default=5, help="Outer CV folds")
    parser.add_argument("--inner_splits", type=int, default=3, help="Inner CV folds")
    parser.add_argument("--n_jobs", type=int, default=1, help="Parallel jobs for model selection")
    parser.add_argument("--test_size", type=float, default=0.2, help="Holdout test set fraction")
    parser.add_argument("--date_col", type=str, default=None, help="Date column for temporal validation (optional)")
    parser.add_argument("--train_years", type=str, default=None, help="Comma-separated years for training (e.g. 2022,2023)")
    parser.add_argument("--test_years", type=str, default=None, help="Comma-separated years for temporal testing (e.g. 2024)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    train_years = [int(y) for y in args.train_years.split(',')] if args.train_years else None
    test_years = [int(y) for y in args.test_years.split(',')] if args.test_years else None

    cfg = {
        "data_path": args.data,
        "target_col": args.target,
        "id_col": args.id,
        "output_dir": args.output,
        "random_seed": args.seed,
        "outer_splits": args.outer_splits,
        "inner_splits": args.inner_splits,
        "n_jobs": args.n_jobs,
        "test_size": args.test_size,
        "date_col": args.date_col,
        "train_years": train_years,
        "test_years": test_years
    }
    run_pipeline(cfg)


if __name__ == "__main__":
    main()

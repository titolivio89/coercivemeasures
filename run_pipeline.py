#!/usr/bin/env python3
"""Top-level entrypoint to run the ML pipeline.
Usage: python run_pipeline.py --data data.csv --target target_column --output outputs/
"""
import argparse
import os
from src.pipeline import run


def main():
    parser = argparse.ArgumentParser(description="Run ML pipeline")
    parser.add_argument("--data", required=True, help="Path to CSV dataset")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--id", default=None, help="ID column to ignore (optional)")
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--outer_splits", type=int, default=5, help="Outer CV folds")
    parser.add_argument("--inner_splits", type=int, default=3, help="Inner CV folds")
    parser.add_argument("--n_jobs", type=int, default=1, help="Parallel jobs for model selection")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    cfg = {
        "data_path": args.data,
        "target_col": args.target,
        "id_col": args.id,
        "output_dir": args.output,
        "random_seed": args.seed,
        "outer_splits": args.outer_splits,
        "inner_splits": args.inner_splits,
        "n_jobs": args.n_jobs,
    }
    run(cfg)


if __name__ == "__main__":
    main()

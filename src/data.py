"""Data loading and basic inspection utilities."""
import pandas as pd


def load_data(path: str, target_col: str, id_col: str = None):
    df = pd.read_csv(path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data columns: {df.columns.tolist()}")
    if id_col is not None and id_col in df.columns:
        df = df.drop(columns=[id_col])
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

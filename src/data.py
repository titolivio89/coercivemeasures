"""Data loading and basic inspection utilities.
Returns X (features), y (target), and optionally ids (if id_col provided and present).
"""
import pandas as pd
from typing import Tuple, Optional


def load_data(path: str, target_col: str, id_col: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
    df = pd.read_csv(path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data columns: {df.columns.tolist()}")
    ids = None
    if id_col is not None and id_col in df.columns:
        ids = df[id_col].copy()
        df = df.drop(columns=[id_col])
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y, ids

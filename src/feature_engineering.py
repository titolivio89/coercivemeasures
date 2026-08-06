"""
Feature engineering utilities. For now, simple polynomial interactions and optional selection.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from config import Config


def add_feature_pipeline(X_processed, cfg: Config):
    # X_processed may be numpy array or DataFrame
    if not isinstance(X_processed, pd.DataFrame):
        X = pd.DataFrame(X_processed, columns=[f"f{i}" for i in range(X_processed.shape[1])])
    else:
        X = X_processed.copy()

    # add low-degree polynomial interactions (degree=2) but avoid explosion
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    poly_feats = poly.fit_transform(X.values)
    poly_cols = [f"p{i}" for i in range(poly_feats.shape[1])]
    X_poly = pd.DataFrame(poly_feats, columns=poly_cols)

    # keep original + poly
    X_out = pd.concat([X.reset_index(drop=True), X_poly.reset_index(drop=True)], axis=1)
    return X_out

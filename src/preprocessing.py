"""
Preprocessing pipeline: imputation, scaling, optional encoding.
Returns a sklearn Pipeline
"""
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from config import Config


def build_preprocessing_pipeline(cfg: Config):
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    return pipe

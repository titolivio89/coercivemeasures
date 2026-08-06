"""
Data loading utilities. Loads data/data.csv if present, otherwise generates a synthetic dataset.
Returns X (DataFrame), y (Series), and original df (DataFrame)
"""
from pathlib import Path
import pandas as pd
from sklearn.datasets import make_classification
from config import Config


def load_data(cfg: Config):
    data_path = Path("data/data.csv")
    if data_path.exists():
        df = pd.read_csv(data_path)
        if "target" not in df.columns:
            raise ValueError("data.csv must contain a 'target' column")
        y = df["target"].copy()
        X = df.drop(columns=["target"])
        return X, y, df
    else:
        # generate synthetic dataset
        X_np, y_np = make_classification(n_samples=2000, n_features=20, n_informative=8,
                                         n_redundant=2, n_clusters_per_class=2, weights=[0.7,0.3],
                                         flip_y=0.01, random_state=cfg.random_state)
        cols = [f"f{i}" for i in range(X_np.shape[1])]
        X = pd.DataFrame(X_np, columns=cols)
        y = pd.Series(y_np, name="target")
        df = pd.concat([X, y], axis=1)
        return X, y, df

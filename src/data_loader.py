"""
Data loading utilities. Loads data/data.csv if present, otherwise generates a synthetic dataset.
Returns X (DataFrame), y (Series), and original df (DataFrame)
"""
from pathlib import Path
import pandas as pd
from config import Config


def load_data(cfg: Config):

    data_path = Path(
        r"\\med.tu-dresden.de\zfs\folder\CALVIGUIL\Desktop\reruncm\df_merged_with_llama.csv"
    )

    df = pd.read_csv(data_path)

    predictors = [
        "aggression",
        "police",
        "psychosis",
        "suicidality",
        "akute_intoxikation"
    ]

    # Binäres Outcome:
    # 0 = keine Zwangsmaßnahme
    # 1 = mindestens eine Zwangsmaßnahme
    df["target"] = df["Art_Massnahme"].notna().astype(int)

    # Nur vollständige Fälle verwenden
    df = df.dropna(subset=predictors)

    X = df[predictors].copy()
    y = df["target"].copy()

    return X, y, df

"""Leakage-controlled data utilities shared by baselines and AIGQFusion."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from config import FEATURES, LABEL_COLUMN, GROUP_COLUMN, N_INNER_FOLDS


def load_clinvar_mve(path: str):
    df = pd.read_csv(path)
    missing = [c for c in FEATURES + [LABEL_COLUMN, GROUP_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"ClinVar-MVE is missing required columns: {missing}")
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    y = df[LABEL_COLUMN].astype(int).to_numpy()
    groups = df[GROUP_COLUMN].astype(str).to_numpy()
    if set(np.unique(y)) - {0, 1}:
        raise ValueError("label must be binary {0,1}.")
    return df, X, y, groups


class FoldPreprocessor:
    """Median imputation + standardization fitted on training data only."""
    def __init__(self):
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()

    def fit(self, X):
        z = self.imputer.fit_transform(X)
        self.scaler.fit(z)
        return self

    def transform(self, X):
        return self.scaler.transform(self.imputer.transform(X)).astype(np.float32)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def inner_splits(X, y, groups, seed):
    cv = StratifiedGroupKFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=seed)
    return list(cv.split(X, y, groups))


def outer_splits(X, y, groups, seed, n_splits=5):
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(X, y, groups))

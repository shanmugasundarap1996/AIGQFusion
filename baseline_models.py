"""ClinVar-MVE baseline benchmark used with AIGQFusion.

Implements the paper comparator bank without embedding reported results.
Run: python baseline_models.py --data ClinVar-MVE.csv
"""
from __future__ import annotations
import argparse, warnings
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import matthews_corrcoef, f1_score, log_loss
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline

from config import REPEAT_SEEDS, N_OUTER_FOLDS
from data_protocol import load_clinvar_mve, FoldPreprocessor, inner_splits, outer_splits
warnings.filterwarnings("ignore")


def probability(model, X):
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, 1]
    s = np.asarray(model.decision_function(X), dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(s, -30, 30)))


def candidate_bank(seed: int, n_features: int):
    bank = {
        "Logistic Regression": [
            LogisticRegression(C=.3, solver="liblinear", max_iter=5000, random_state=seed),
            LogisticRegression(C=1., solver="liblinear", class_weight="balanced", max_iter=5000, random_state=seed),
            LogisticRegression(C=3., solver="liblinear", class_weight="balanced", max_iter=5000, random_state=seed),
        ],
        "RBF-SVM": [
            SVC(C=.7, gamma="scale", kernel="rbf", class_weight="balanced", probability=True, random_state=seed),
            SVC(C=1.5, gamma="scale", kernel="rbf", class_weight="balanced", probability=True, random_state=seed),
            SVC(C=2., gamma=.04, kernel="rbf", class_weight="balanced", probability=True, random_state=seed),
        ],
        "Random Forest": [
            RandomForestClassifier(n_estimators=750, min_samples_leaf=1, max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=seed),
            RandomForestClassifier(n_estimators=750, min_samples_leaf=2, max_features=.75, class_weight="balanced_subsample", n_jobs=-1, random_state=seed),
        ],
        "Extra Trees": [
            ExtraTreesClassifier(n_estimators=750, min_samples_leaf=1, max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=seed),
            ExtraTreesClassifier(n_estimators=750, min_samples_leaf=2, max_features=.75, class_weight="balanced", n_jobs=-1, random_state=seed),
        ],
        "HistGradientBoosting": [
            HistGradientBoostingClassifier(learning_rate=.05, max_leaf_nodes=31, l2_regularization=1., max_iter=750, random_state=seed),
            HistGradientBoostingClassifier(learning_rate=.035, max_leaf_nodes=63, l2_regularization=2., max_iter=750, random_state=seed),
        ],
        "RBF Nystrom": [
            make_pipeline(Nystroem(kernel="rbf", gamma=1/n_features, n_components=32, random_state=seed), LogisticRegression(C=.5, class_weight="balanced", max_iter=5000)),
            make_pipeline(Nystroem(kernel="rbf", gamma=1/n_features, n_components=48, random_state=seed), LogisticRegression(C=1., class_weight="balanced", max_iter=5000)),
        ],
    }
    try:
        from xgboost import XGBClassifier
        bank["XGBoost"] = [
            XGBClassifier(n_estimators=750, max_depth=5, learning_rate=.05, min_child_weight=1., subsample=.9, colsample_bytree=.9, reg_lambda=2., tree_method="hist", eval_metric="logloss", random_state=seed, n_jobs=-1),
            XGBClassifier(n_estimators=750, max_depth=6, learning_rate=.035, min_child_weight=2., subsample=.9, colsample_bytree=.9, reg_lambda=2., tree_method="hist", eval_metric="logloss", random_state=seed, n_jobs=-1),
        ]
    except ImportError: pass
    try:
        from lightgbm import LGBMClassifier
        bank["LightGBM"] = [
            LGBMClassifier(n_estimators=750, num_leaves=31, learning_rate=.05, min_child_samples=20, subsample=.9, colsample_bytree=.9, reg_lambda=2., class_weight="balanced", random_state=seed, n_jobs=-1, verbosity=-1),
            LGBMClassifier(n_estimators=750, num_leaves=63, learning_rate=.035, min_child_samples=30, subsample=.9, colsample_bytree=.9, reg_lambda=2., class_weight="balanced", random_state=seed, n_jobs=-1, verbosity=-1),
        ]
    except ImportError: pass
    try:
        from catboost import CatBoostClassifier
        bank["CatBoost"] = [CatBoostClassifier(loss_function="Logloss", eval_metric="Logloss", auto_class_weights="Balanced", random_seed=seed, task_type="CPU", verbose=False)]
    except ImportError: pass
    return bank


def score_candidate(model, X, y, groups, seed):
    pred = np.zeros(len(y), float)
    for tr, va in inner_splits(X, y, groups, seed):
        prep = FoldPreprocessor().fit(X[tr])
        m = clone(model)
        m.fit(prep.transform(X[tr]), y[tr])
        pred[va] = probability(m, prep.transform(X[va]))
    hard = (pred >= .5).astype(int)
    return (matthews_corrcoef(y, hard), f1_score(y, hard, average="macro"), -log_loss(y, np.clip(pred, 1e-7, 1-1e-7)))


def run(data_path: str):
    _, X, y, groups = load_clinvar_mve(data_path)
    # Deliberately does not print/persist paper result tables. It executes the frozen model-selection protocol.
    fitted = []
    for seed in REPEAT_SEEDS:
        for fold, (tr, te) in enumerate(outer_splits(X, y, groups, seed, N_OUTER_FOLDS)):
            for name, candidates in candidate_bank(seed + fold, X.shape[1]).items():
                if name == "CatBoost":
                    best = candidates[0]
                else:
                    scores = [score_candidate(c, X[tr], y[tr], groups[tr], seed + fold) for c in candidates]
                    best = candidates[max(range(len(scores)), key=lambda i: scores[i])]
                prep = FoldPreprocessor().fit(X[tr])
                model = clone(best).fit(prep.transform(X[tr]), y[tr])
                _ = probability(model, prep.transform(X[te]))
                fitted.append((seed, fold, name))
    return fitted


def main():
    p = argparse.ArgumentParser(description="ClinVar-MVE baseline benchmark")
    p.add_argument("--data", default="ClinVar-MVE.csv")
    args = p.parse_args()
    run(args.data)

if __name__ == "__main__": main()

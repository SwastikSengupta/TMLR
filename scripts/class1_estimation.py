"""
EXPERIMENT 1 -- Independent replication of Class I (estimation leakage).

The original study reports that fitting scalers or encoders on the full dataset
before splitting is negligible, with all nine conditions producing
|Delta AUC| <= 0.005 across 2,047 datasets. We test the same claim on fifteen
real UCI datasets that were not part of that corpus, with three classifiers and
two preprocessing operations.

correct : operation fitted inside the training fold
leaky   : operation fitted once on the full matrix, then cross-validate
Only the position of the preprocessing boundary differs.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
warnings.filterwarnings("ignore")

from datasets import load_all
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEEDS = [0, 1, 2, 3, 4]
FOLDS = 5

CLFS = {
    "logreg": lambda: LogisticRegression(max_iter=5000),
    "lda": lambda: LinearDiscriminantAnalysis(),
    "rf": lambda: RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=2),
}


def auc(y, p):
    return roc_auc_score(y, p)


def run_scaling(X, y, mk, seed):
    cv = StratifiedKFold(FOLDS, shuffle=True, random_state=seed)
    p_ok = cross_val_predict(Pipeline([("s", StandardScaler()), ("c", mk())]),
                             X, y, cv=cv, method="predict_proba")[:, 1]
    Xl = StandardScaler().fit_transform(X)
    p_lk = cross_val_predict(mk(), Xl, y, cv=cv, method="predict_proba")[:, 1]
    return auc(y, p_ok), auc(y, p_lk)


def run_imputation(X, y, mk, seed, rate=0.15):
    rng = np.random.default_rng(seed)
    Xm = X.copy()
    Xm[rng.random(X.shape) < rate] = np.nan
    cv = StratifiedKFold(FOLDS, shuffle=True, random_state=seed)
    p_ok = cross_val_predict(
        Pipeline([("i", SimpleImputer()), ("s", StandardScaler()), ("c", mk())]),
        Xm, y, cv=cv, method="predict_proba")[:, 1]
    Xl = SimpleImputer().fit_transform(Xm)
    p_lk = cross_val_predict(Pipeline([("s", StandardScaler()), ("c", mk())]),
                             Xl, y, cv=cv, method="predict_proba")[:, 1]
    return auc(y, p_ok), auc(y, p_lk)


OPS = {"scaling": run_scaling, "imputation": run_imputation}

if __name__ == "__main__":
    print("Loading real datasets\n")
    data = load_all(max_n=6000)
    rows = []
    print(f"\n{'dataset':16s} {'clf':8s} {'op':11s} {'correct':>8} {'leaky':>7} {'dAUC':>8}")
    for name, X, y in data:
        for cname, mk in CLFS.items():
            for oname, fn in OPS.items():
                ok, lk = [], []
                for s in SEEDS:
                    a, b = fn(X, y, mk, s)
                    ok.append(a); lk.append(b)
                ok, lk = np.array(ok), np.array(lk)
                d = lk - ok
                rows.append({"dataset": name, "n": int(len(y)), "p": int(X.shape[1]),
                             "clf": cname, "op": oname,
                             "correct": float(ok.mean()), "leaky": float(lk.mean()),
                             "dAUC": float(d.mean()), "dAUC_sd": float(d.std(ddof=1))})
                print(f"{name:16s} {cname:8s} {oname:11s} {ok.mean():8.4f} "
                      f"{lk.mean():7.4f} {d.mean():+8.4f}")

    d_all = np.array([r["dAUC"] for r in rows])
    within = float(np.mean(np.abs(d_all) <= 0.005))
    print("\n=== Class I replication verdict ===")
    print(f"  conditions tested                : {len(rows)}")
    print(f"  mean |dAUC|                      : {np.abs(d_all).mean():.5f}")
    print(f"  max  |dAUC|                      : {np.abs(d_all).max():.5f}")
    print(f"  fraction within original's 0.005 : {within:.3f}")
    # paired effect size across conditions
    dz = float(d_all.mean() / d_all.std(ddof=1)) if d_all.std(ddof=1) > 0 else 0.0
    print(f"  paired effect size d_z           : {dz:+.3f}")

    json.dump({"rows": rows,
               "summary": {"n_conditions": len(rows),
                           "mean_abs_dAUC": float(np.abs(d_all).mean()),
                           "max_abs_dAUC": float(np.abs(d_all).max()),
                           "frac_within_0.005": within, "d_z": dz}},
              open("results/class1_estimation.json", "w"), indent=1)
    print("\nwrote results/class1_estimation.json")

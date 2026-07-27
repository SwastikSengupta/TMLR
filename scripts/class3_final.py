"""
Class III (memorization leakage): is it capacity, or is it locality?

Roth (arXiv:2604.04199) reports Class III "scales with model capacity", evidenced
by a span from d_z = 0.37 (Naive Bayes) to d_z = 1.11 (Decision Tree) at 10
percent duplication. That comparison changes capacity and algorithm family at the
same time, so it cannot separate them.

We propose an alternative account. What makes an exact duplicate dangerous is not
how flexible the learner is, but whether a single training point can dominate its
own prediction. Call a learner LOCAL if it can: k-NN gives a distance-zero copy
maximal weight, and a tree can isolate a point in its own leaf. Call it GLOBAL if
every training point enters through a pooled fit: naive Bayes, LDA, and logistic
regression estimate shared parameters that one duplicate barely moves.

Locality and capacity are not the same thing, and they come apart at k-NN, which
is non-parametric with no fitted parameters yet maximally local.

Design. Identical data in both arms; only the split changes.
  leaky   : random stratified CV, copies straddle the split
  correct : grouped CV, all copies of a row stay together
Per-dataset effects are retained so that every comparison can be paired.
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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (StratifiedGroupKFold, StratifiedKFold,
                                     cross_val_predict)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEEDS = [0, 1, 2]
FOLDS = 5
DUP_RATE = 0.10
MAX_N = 1500

# locality is the hypothesised driver; capacity_rank is the competing account
MODELS = {
    "naive_bayes":    (lambda: GaussianNB(),                                   "global", 1),
    "lda":            (lambda: LinearDiscriminantAnalysis(),                   "global", 2),
    "logreg":         (lambda: LogisticRegression(max_iter=5000),              "global", 3),
    "knn_k25":        (lambda: KNeighborsClassifier(25),                       "local",  2),
    "knn_k5":         (lambda: KNeighborsClassifier(5),                        "local",  4),
    "knn_k1":         (lambda: KNeighborsClassifier(1),                        "local",  6),
    "decision_tree":  (lambda: DecisionTreeClassifier(random_state=0),         "local",  6),
    "random_forest":  (lambda: RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=2), "local", 7),
    "grad_boost":     (lambda: GradientBoostingClassifier(random_state=0),     "local",  6),
    # within-family capacity ladder, all local, capacity varies fully
    "rf_depth_1":     (lambda: RandomForestClassifier(n_estimators=60, max_depth=1, random_state=0, n_jobs=2),  "local", 1),
    "rf_depth_3":     (lambda: RandomForestClassifier(n_estimators=60, max_depth=3, random_state=0, n_jobs=2),  "local", 3),
    "rf_depth_10":    (lambda: RandomForestClassifier(n_estimators=60, max_depth=10, random_state=0, n_jobs=2), "local", 6),
}

ACROSS_FAMILY = ["naive_bayes", "lda", "logreg", "knn_k5", "decision_tree",
                 "random_forest", "grad_boost"]
WITHIN_RF = ["rf_depth_1", "rf_depth_3", "rf_depth_10", "random_forest"]
GLOBAL = [m for m, v in MODELS.items() if v[1] == "global"]
LOCAL_MAIN = ["knn_k5", "decision_tree", "random_forest", "grad_boost"]


def duplicate(X, y, rate, rng):
    n = len(y)
    g = np.arange(n)
    k = int(round(rate * n))
    if k == 0:
        return X, y, g
    pick = rng.choice(n, k, replace=False)
    return (np.vstack([X, X[pick]]), np.concatenate([y, y[pick]]),
            np.concatenate([g, g[pick]]))


def effect(X, y, mk, seed):
    rng = np.random.default_rng(seed)
    Xd, yd, gd = duplicate(X, y, DUP_RATE, rng)
    pipe = lambda: Pipeline([("s", StandardScaler()), ("c", mk())])
    lk = cross_val_predict(pipe(), Xd, yd,
                           cv=StratifiedKFold(FOLDS, shuffle=True, random_state=seed),
                           method="predict_proba")[:, 1]
    ok = cross_val_predict(pipe(), Xd, yd, groups=gd,
                           cv=StratifiedGroupKFold(FOLDS, shuffle=True, random_state=seed),
                           method="predict_proba")[:, 1]
    return roc_auc_score(yd, lk) - roc_auc_score(yd, ok)


if __name__ == "__main__":
    data = load_all(max_n=MAX_N, verbose=False)
    names = [d[0] for d in data]
    print(f"{len(data)} real datasets | duplication {DUP_RATE} | {len(SEEDS)} seeds\n")

    per = {m: {} for m in MODELS}
    print(f"{'model':16s} {'mean dAUC':>10} {'d_z':>7}  (per-dataset effects retained)")
    for mname, (mk, locality, cap) in MODELS.items():
        for dname, X, y in data:
            vals = []
            for s in SEEDS:
                try:
                    vals.append(effect(X, y, mk, s))
                except Exception:
                    pass
            if vals:
                per[mname][dname] = float(np.mean(vals))
        v = np.array(list(per[mname].values()))
        dz = float(v.mean() / v.std(ddof=1)) if len(v) > 1 and v.std(ddof=1) > 0 else float("nan")
        print(f"{mname:16s} {v.mean():+10.4f} {dz:+7.3f}")
        json.dump({"per_dataset": per,
                   "meta": {m: {"locality": MODELS[m][1], "capacity_rank": MODELS[m][2]}
                            for m in MODELS},
                   "datasets": names, "dup_rate": DUP_RATE, "seeds": list(SEEDS)},
                  open("results/class3_per_dataset.json", "w"), indent=1)

    print("\nwrote results/class3_per_dataset.json")

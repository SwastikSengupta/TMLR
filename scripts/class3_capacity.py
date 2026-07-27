"""
EXPERIMENT 2 -- Class III (memorization leakage): capacity or inductive bias?

The original study reports that memorization leakage is "amplified by model
capacity", evidencing this with a span across algorithm families: d_z = 0.37 for
Naive Bayes up to 1.11 for a Decision Tree at 10 percent duplication.

That comparison varies capacity and inductive bias together. A decision tree is
not merely a higher-capacity naive Bayes; it partitions the input space with
axis-aligned cuts, which is precisely the operation needed to isolate an exact
duplicate. So the reported span is consistent with two different explanations,
and they carry different advice.

We separate them:
  ACROSS families : seven algorithms, default capacity
  WITHIN a family : random forest, depth 1 to unrestricted

If capacity is the driver, the within-family ladder should reproduce a large
span. If inductive bias is the driver, the within-family span should be small
while the across-family span stays large.

Counterfactual design. The dataset is identical in both arms; only the split
changes. Duplicated rows form a group.
  leaky   : random stratified CV, so copies straddle the split
  correct : grouped CV, so all copies of a row stay on one side
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
from sklearn.ensemble import (GradientBoostingClassifier,
                              RandomForestClassifier)
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

ACROSS = {
    "naive_bayes": lambda: GaussianNB(),
    "logreg": lambda: LogisticRegression(max_iter=5000),
    "lda": lambda: LinearDiscriminantAnalysis(),
    "knn5": lambda: KNeighborsClassifier(5),
    "decision_tree": lambda: DecisionTreeClassifier(random_state=0),
    "random_forest": lambda: RandomForestClassifier(n_estimators=100,
                                                    random_state=0, n_jobs=2),
    "grad_boost": lambda: GradientBoostingClassifier(random_state=0),
}

WITHIN = {
    f"rf_depth_{d}": (lambda dd=d: RandomForestClassifier(
        n_estimators=100, max_depth=dd, random_state=0, n_jobs=2))
    for d in [1, 2, 3, 5, 10, None]
}


def duplicate(X, y, rate, rng):
    """Duplicate a fraction of rows. Group id ties each row to its copies."""
    n = len(y)
    g = np.arange(n)
    k = int(round(rate * n))
    if k == 0:
        return X, y, g
    pick = rng.choice(n, k, replace=False)
    return (np.vstack([X, X[pick]]), np.concatenate([y, y[pick]]),
            np.concatenate([g, g[pick]]))


def one(X, y, mk, seed):
    rng = np.random.default_rng(seed)
    Xd, yd, gd = duplicate(X, y, DUP_RATE, rng)
    pipe = lambda: Pipeline([("s", StandardScaler()), ("c", mk())])
    p_lk = cross_val_predict(pipe(), Xd, yd,
                             cv=StratifiedKFold(FOLDS, shuffle=True,
                                                random_state=seed),
                             method="predict_proba")[:, 1]
    p_ok = cross_val_predict(pipe(), Xd, yd, groups=gd,
                             cv=StratifiedGroupKFold(FOLDS, shuffle=True,
                                                     random_state=seed),
                             method="predict_proba")[:, 1]
    return roc_auc_score(yd, p_ok), roc_auc_score(yd, p_lk)


def sweep(data, models, label, out):
    print(f"\n--- {label} ---")
    print(f"{'model':16s} {'correct':>8} {'leaky':>7} {'dAUC':>8} {'d_z':>7}")
    for mname, mk in models.items():
        per_ds = []
        for name, X, y in data:
            ok, lk = [], []
            for s in SEEDS:
                try:
                    a, b = one(X, y, mk, s)
                except Exception:
                    continue
                ok.append(a); lk.append(b)
            if ok:
                per_ds.append(np.mean(lk) - np.mean(ok))
        per_ds = np.array(per_ds)
        dz = float(per_ds.mean() / per_ds.std(ddof=1)) if len(per_ds) > 1 and per_ds.std(ddof=1) > 0 else float("nan")
        rec = {"arm": label, "model": mname, "n_datasets": int(len(per_ds)),
               "dAUC": float(per_ds.mean()), "dAUC_sd": float(per_ds.std(ddof=1)),
               "d_z": dz}
        out.append(rec)
        print(f"{mname:16s} {'':>8} {'':>7} {per_ds.mean():+8.4f} {dz:+7.3f}")
        json.dump(out, open("results/class3_capacity.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    data = load_all(max_n=2000, verbose=False)
    print(f"{len(data)} real datasets, duplication rate {DUP_RATE}")
    out = []
    sweep(data, ACROSS, "across_family", out)
    sweep(data, WITHIN, "within_family_rf", out)

    a = [r for r in out if r["arm"] == "across_family"]
    w = [r for r in out if r["arm"] == "within_family_rf"]
    span_a = max(r["dAUC"] for r in a) - min(r["dAUC"] for r in a)
    span_w = max(r["dAUC"] for r in w) - min(r["dAUC"] for r in w)
    print("\n=== capacity or inductive bias? ===")
    print(f"  across-family dAUC span : {span_a:.4f}")
    print(f"  within-family dAUC span : {span_w:.4f}")
    print(f"  ratio                   : {span_a/span_w:.2f}x"
          if span_w > 0 else "  within-family span is zero")
    json.dump({"rows": out, "span_across": span_a, "span_within": span_w},
              open("results/class3_capacity.json", "w"), indent=1)
    print("\nwrote results/class3_capacity.json")

"""
The practical output of this study, as a drop-in function.

    from capacity_index import capacity_index, interpret
    c = capacity_index(model, X, y)

Fits the model to randomised labels on your own feature matrix and returns
training AUC. High values mean the learner can memorise individual records, and
duplicate or near-duplicate rows in your data will inflate a randomly split
cross-validation estimate. Requires no ground truth and no held-out set.

Calibration observed across 11 learners and 15 UCI datasets at 10% duplication:
    index ~0.60  ->  leakage below 0.005 AUC
    index ~0.78  ->  leakage around 0.007 AUC
    index ~0.99  ->  leakage above 0.017 AUC
These are descriptive of this study, not a certified threshold.
"""
import numpy as np
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def capacity_index(model, X, y, n_repeats=3, random_state=0):
    """Training AUC when the labels are permuted. Higher means more memorisation."""
    scores = []
    for r in range(n_repeats):
        rng = np.random.default_rng(random_state + r)
        y_perm = rng.permutation(np.asarray(y))
        m = make_pipeline(StandardScaler(), clone(model)).fit(X, y_perm)
        if hasattr(m, "predict_proba"):
            p = m.predict_proba(X)[:, 1]
        else:
            p = m.decision_function(X)
        scores.append(roc_auc_score(y_perm, p))
    return float(np.mean(scores))


def interpret(c):
    if c < 0.65:
        return ("low memorisation capacity; duplicate records should inflate "
                "cross-validation by under ~0.005 AUC")
    if c < 0.90:
        return ("moderate memorisation capacity; expect inflation on the order "
                "of 0.005 to 0.01 AUC if duplicates are present")
    return ("high memorisation capacity; duplicates will inflate substantially, "
            "on the order of 0.02 AUC. Use grouped cross-validation.")


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    d = load_breast_cancer()
    for name, m in [("logreg", LogisticRegression(max_iter=5000)),
                    ("random_forest", RandomForestClassifier(n_estimators=100,
                                                             random_state=0))]:
        c = capacity_index(m, d.data, d.target)
        print(f"{name:15s} index {c:.3f}\n  {interpret(c)}")

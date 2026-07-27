"""
Objective capacity measure: the random-label fit.

The capacity ranks used in the first analysis were assigned by judgement, which
makes any regression on them partly circular. We replace them with a measured
quantity in the tradition of Zhang et al.: fit each model to RANDOMISED labels on
each real dataset and record training AUC. A model that can drive training AUC to
1.0 on random labels has the capacity to memorise; one that cannot, does not.
The features are the real ones, so this is not a synthetic dataset.
"""
import json, sys, warnings
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"src")); warnings.filterwarnings("ignore")
from datasets import load_all
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from class3_final import MODELS

data = load_all(max_n=1500, verbose=False)
out = {}
print("model            random-label train AUC (memorisation capacity)")
for mname,(mk,loc,cap) in MODELS.items():
    vals=[]
    for dname,X,y in data:
        for s in [0,1]:
            rng=np.random.default_rng(s)
            yr=rng.permutation(y)                      # destroy the signal only
            try:
                p=Pipeline([("s",StandardScaler()),("c",mk())]).fit(X,yr)
                pr=p.predict_proba(X)[:,1]
                vals.append(roc_auc_score(yr,pr))
            except Exception: pass
    if vals:
        out[mname]=float(np.mean(vals))
        print(f"  {mname:16s} {np.mean(vals):.4f}")
    json.dump(out,open("results/capacity_measured.json","w"),indent=1)
print("\nwrote results/capacity_measured.json")

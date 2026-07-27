import json, sys, warnings
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"src")); warnings.filterwarnings("ignore")
from datasets import load_all
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from class3_final import effect, MODELS
d=json.load(open("results/class3_per_dataset.json")); per=d["per_dataset"]
data=load_all(max_n=1500,verbose=False)
REST={k:MODELS[k] for k in ["grad_boost","rf_depth_1","rf_depth_3","rf_depth_10"] if not per.get(k)}
print("remaining:",list(REST))
for mname,(mk,loc,cap) in REST.items():
    per.setdefault(mname,{})
    for dname,X,y in data:
        vals=[]
        for s in [0,1,2]:
            try: vals.append(effect(X,y,mk,s))
            except Exception: pass
        if vals: per[mname][dname]=float(np.mean(vals))
    v=np.array(list(per[mname].values()))
    dz=v.mean()/v.std(ddof=1) if len(v)>1 else float('nan')
    print(f"  {mname:14s} {v.mean():+.4f}  d_z {dz:+.3f}")
    d["per_dataset"]=per; json.dump(d,open("results/class3_per_dataset.json","w"),indent=1)
print("done")

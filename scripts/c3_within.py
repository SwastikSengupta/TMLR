import json, sys, warnings
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent/"src"))
warnings.filterwarnings("ignore")
from datasets import load_all
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
SEEDS=[0,1]; FOLDS=5; DUP=0.10
def duplicate(X,y,rate,rng):
    n=len(y); g=np.arange(n); k=int(round(rate*n))
    pick=rng.choice(n,k,replace=False)
    return np.vstack([X,X[pick]]), np.concatenate([y,y[pick]]), np.concatenate([g,g[pick]])
def one(X,y,mk,seed):
    rng=np.random.default_rng(seed); Xd,yd,gd=duplicate(X,y,DUP,rng)
    pipe=lambda: Pipeline([("s",StandardScaler()),("c",mk())])
    lk=cross_val_predict(pipe(),Xd,yd,cv=StratifiedKFold(FOLDS,shuffle=True,random_state=seed),method="predict_proba")[:,1]
    ok=cross_val_predict(pipe(),Xd,yd,groups=gd,cv=StratifiedGroupKFold(FOLDS,shuffle=True,random_state=seed),method="predict_proba")[:,1]
    return roc_auc_score(yd,ok), roc_auc_score(yd,lk)
data=load_all(max_n=1500,verbose=False)
print(f"{len(data)} datasets")
out=[]
LADDERS={
 "within_family_rf":{f"rf_depth_{d}":(lambda dd=d: RandomForestClassifier(n_estimators=60,max_depth=dd,random_state=0,n_jobs=2)) for d in [1,3,5,10,None]},
 "within_family_knn":{f"knn_k{k}":(lambda kk=k: KNeighborsClassifier(kk)) for k in [1,5,25]},
}
for arm,models in LADDERS.items():
    print(f"\n--- {arm} ---")
    for mname,mk in models.items():
        per=[]
        for name,X,y in data:
            ok,lk=[],[]
            for s in SEEDS:
                try: a,b=one(X,y,mk,s); ok.append(a); lk.append(b)
                except Exception: pass
            if ok: per.append(np.mean(lk)-np.mean(ok))
        per=np.array(per)
        dz=float(per.mean()/per.std(ddof=1)) if len(per)>1 and per.std(ddof=1)>0 else float('nan')
        out.append({"arm":arm,"model":mname,"n_datasets":int(len(per)),
                    "dAUC":float(per.mean()),"dAUC_sd":float(per.std(ddof=1)),"d_z":dz})
        print(f"  {mname:14s} dAUC {per.mean():+.4f}  d_z {dz:+.3f}")
        json.dump(out,open("results/class3_within.json","w"),indent=1)
print("\ndone")

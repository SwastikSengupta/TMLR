"""
Statistical analysis of the Class III results.

Unit of analysis is the dataset, so every comparison is paired: each dataset
contributes one effect per model, and tests are run on the within-dataset
differences. Sample size is the number of datasets, not the number of rows,
which is the conservative choice.

Tests:
  T1  local versus global, paired Wilcoxon signed-rank
  T2  the crucial single comparison: k-NN with k=25, the lowest-capacity local
      learner, against each global learner, Holm-corrected
  T3  within-family capacity span versus across-family span, paired
  T4  regression of effect on locality and on capacity rank, to see which
      account survives when both are entered
"""
import json
import itertools
import numpy as np
from scipy import stats

D = json.load(open("results/class3_per_dataset.json"))
PER, META = D["per_dataset"], D["meta"]
MODELS = [m for m in PER if PER[m]]
DATASETS = sorted(set.intersection(*[set(PER[m]) for m in MODELS]))
print(f"{len(MODELS)} models x {len(DATASETS)} datasets, all paired\n")

def vec(m):
    return np.array([PER[m][d] for d in DATASETS])

GLOBAL = [m for m in MODELS if META[m]["locality"] == "global"]
LOCAL_MAIN = ["knn_k5", "decision_tree", "random_forest", "grad_boost"]
LOCAL_MAIN = [m for m in LOCAL_MAIN if m in MODELS]

out = {}

# ---------------------------------------------------------------- descriptive
print("model            mean dAUC     d_z   locality  cap")
desc = {}
for m in MODELS:
    v = vec(m)
    dz = v.mean() / v.std(ddof=1)
    desc[m] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
               "d_z": float(dz), "locality": META[m]["locality"],
               "capacity_rank": META[m]["capacity_rank"]}
    print(f"{m:16s} {v.mean():+9.4f} {dz:+7.3f}   {META[m]['locality']:7s} {META[m]['capacity_rank']}")
out["descriptive"] = desc

# ------------------------------------------------------------------------- T1
print("\n=== T1: local vs global (paired by dataset, Wilcoxon) ===")
loc_mean = np.mean([vec(m) for m in LOCAL_MAIN], axis=0)
glo_mean = np.mean([vec(m) for m in GLOBAL], axis=0)
diff = loc_mean - glo_mean
w, p = stats.wilcoxon(loc_mean, glo_mean)
# rank-biserial effect size for paired Wilcoxon
n = len(diff)
r_rb = 1 - (2 * w) / (n * (n + 1) / 2)
boot = [np.mean(np.random.default_rng(s).choice(diff, n, replace=True)) for s in range(5000)]
ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
print(f"  local mean  {loc_mean.mean():+.4f}   global mean {glo_mean.mean():+.4f}")
print(f"  difference  {diff.mean():+.4f}  95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]")
print(f"  Wilcoxon W={w:.1f}  p={p:.2e}   rank-biserial r={r_rb:+.3f}")
print(f"  datasets where local > global: {int((diff > 0).sum())}/{n}")
out["T1"] = {"local_mean": float(loc_mean.mean()), "global_mean": float(glo_mean.mean()),
             "diff": float(diff.mean()), "ci95": ci, "W": float(w), "p": float(p),
             "rank_biserial": float(r_rb), "n_datasets_positive": int((diff > 0).sum()),
             "n": int(n)}

# ------------------------------------------------------------------------- T2
print("\n=== T2: knn_k25 (lowest-capacity local) vs each global ===")
print("    if capacity drove the effect this comparison should not be significant")
if "knn_k25" in MODELS:
    k25 = vec("knn_k25")
    raw = {}
    for g in GLOBAL:
        w2, p2 = stats.wilcoxon(k25, vec(g))
        raw[g] = (float(w2), float(p2), float((k25 - vec(g)).mean()))
    ps = np.array([raw[g][1] for g in GLOBAL])
    order = np.argsort(ps)
    holm = np.empty_like(ps)
    for rank, i in enumerate(order):
        holm[i] = min(1.0, ps[i] * (len(ps) - rank))
    holm = np.maximum.accumulate(holm[order])[np.argsort(order)]
    for g, h in zip(GLOBAL, holm):
        W2, p2, dm = raw[g]
        print(f"  knn_k25 vs {g:12s}  diff {dm:+.4f}  p={p2:.4f}  Holm p={h:.4f}"
              f"  {'SIGNIFICANT' if h < 0.05 else ''}")
    out["T2"] = {g: {"W": raw[g][0], "p_raw": raw[g][1], "p_holm": float(h),
                     "mean_diff": raw[g][2]} for g, h in zip(GLOBAL, holm)}

# ------------------------------------------------------------------------- T3
print("\n=== T3: within-family capacity span vs across-family span ===")
rf_ladder = [m for m in ["rf_depth_1", "rf_depth_3", "random_forest"] if m in MODELS]
across = [m for m in ["naive_bayes", "lda", "logreg", "knn_k5", "decision_tree",
                      "random_forest", "grad_boost"] if m in MODELS]
if len(rf_ladder) >= 2:
    M_w = np.vstack([vec(m) for m in rf_ladder])
    M_a = np.vstack([vec(m) for m in across])
    span_w = M_w.max(0) - M_w.min(0)          # per dataset
    span_a = M_a.max(0) - M_a.min(0)
    w3, p3 = stats.wilcoxon(span_a, span_w)
    print(f"  within-family RF span (capacity 1 -> unrestricted): {span_w.mean():.4f}")
    print(f"  across-family span at default capacity           : {span_a.mean():.4f}")
    print(f"  ratio {span_a.mean()/span_w.mean():.2f}x   Wilcoxon p={p3:.2e}"
          f"   larger on {int((span_a>span_w).sum())}/{len(DATASETS)} datasets")
    out["T3"] = {"span_within": float(span_w.mean()), "span_across": float(span_a.mean()),
                 "ratio": float(span_a.mean()/span_w.mean()), "p": float(p3),
                 "n_datasets_across_larger": int((span_a > span_w).sum())}

# ------------------------------------------------------------------------- T4
print("\n=== T4: which account survives when both are entered? ===")
rows_m = [m for m in MODELS]
y = np.array([desc[m]["mean"] for m in rows_m])
loc = np.array([1.0 if desc[m]["locality"] == "local" else 0.0 for m in rows_m])
cap = np.array([desc[m]["capacity_rank"] for m in rows_m], dtype=float)
cap = (cap - cap.mean()) / cap.std()
X = np.column_stack([np.ones_like(y), loc, cap])
beta, *_ = np.linalg.lstsq(X, y, rcond=None)
resid = y - X @ beta
dof = len(y) - X.shape[1]
s2 = (resid @ resid) / dof
cov = s2 * np.linalg.inv(X.T @ X)
se = np.sqrt(np.diag(cov))
tvals = beta / se
pvals = 2 * (1 - stats.t.cdf(np.abs(tvals), dof))
names = ["intercept", "locality", "capacity_rank"]
for nm, b, s, t, p4 in zip(names, beta, se, tvals, pvals):
    print(f"  {nm:14s} beta={b:+.5f}  se={s:.5f}  t={t:+.2f}  p={p4:.4f}")
ss_tot = ((y - y.mean()) ** 2).sum()
print(f"  R^2 = {1 - (resid@resid)/ss_tot:.3f}   (n = {len(y)} models)")
# univariate comparison
r_loc = stats.pearsonr(loc, y); r_cap = stats.pearsonr(cap, y)
print(f"  univariate  locality r={r_loc[0]:+.3f} (p={r_loc[1]:.4f}) | "
      f"capacity r={r_cap[0]:+.3f} (p={r_cap[1]:.4f})")
out["T4"] = {"beta": dict(zip(names, beta.tolist())),
             "p": dict(zip(names, pvals.tolist())),
             "R2": float(1 - (resid@resid)/ss_tot),
             "univariate_locality_r": float(r_loc[0]), "univariate_locality_p": float(r_loc[1]),
             "univariate_capacity_r": float(r_cap[0]), "univariate_capacity_p": float(r_cap[1])}

# separation check
print("\n=== separation ===")
gmax = max(desc[m]["mean"] for m in GLOBAL)
lmin = min(desc[m]["mean"] for m in MODELS if desc[m]["locality"] == "local")
print(f"  largest global effect  {gmax:+.4f}")
print(f"  smallest local effect  {lmin:+.4f}")
print(f"  complete separation: {lmin > gmax}")
out["separation"] = {"max_global": float(gmax), "min_local": float(lmin),
                     "complete": bool(lmin > gmax)}

json.dump(out, open("results/statistics.json", "w"), indent=1)
print("\nwrote results/statistics.json")

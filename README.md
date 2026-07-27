# How Loud Is the Leak?

An independent reproduction of the leakage-class landscape of Roth
(arXiv:2604.04199), on fifteen real UCI datasets outside the original corpus,
plus a measurable index for the leakage class that actually matters.

**No synthetic data is used anywhere in this study.**

---

## What it found

**Class I replicates exactly.** Fitting scalers or imputers on the full dataset
before splitting, across 70 conditions (15 datasets x 3 classifiers x 2
operations): mean |ΔAUC| = **0.00018**, max = **0.0032**, and **100%** of
conditions fall inside the original's stated ±0.005 bound.

**Class III replicates.** Duplication leakage at 10%, paired per dataset, runs
from d_z = 0.31 (naive Bayes) to d_z = 1.23 (1-NN). The original reports 0.37 and
1.11 for its endpoints. Ordering and magnitude both hold.

**A hypothesis that failed, reported as such.** I proposed that Class III tracks
*locality* — whether one training point can dominate its own prediction — rather
than capacity. It does not. In a joint model with capacity measured rather than
assumed, locality carries no weight (β = −0.0017, p = 0.63) and reverses sign.

**What survived is better than what I set out to test.** Replacing algorithm-family
comparison with a *measured* capacity index — each learner's training AUC on
randomised labels — predicts Class III severity across 11 learners at
**Spearman ρ = 0.982** (p = 8.4×10⁻⁸), Pearson r = 0.912, bootstrap 95% CI
[0.834, 0.984], R² = 0.837. The original's qualitative claim becomes a computable
quantity.

## The practical output

```python
from src.capacity_index import capacity_index, interpret

c = capacity_index(model, X, y)   # fits your model to shuffled labels
print(c, interpret(c))
```

Shuffle the labels, fit, record training AUC. No ground truth needed, no held-out
set needed. Observed calibration in this study:

| index | Class III leakage |
|---|---|
| ~0.60 | < 0.005 AUC |
| ~0.78 | ~0.007 AUC |
| ~0.99 | > 0.017 AUC |

Descriptive of this study, not a certified threshold.

## Quick start

```bash
pip install -r requirements.txt
bash run_all.sh
```

Committed JSONs in `results/` let you check every number without running anything.

## Layout

```
src/datasets.py            15 real UCI datasets, downloaded and parsed
src/capacity_index.py      the drop-in diagnostic
scripts/get_data.sh        fetches the datasets (~2.5 MB)
scripts/class1_estimation.py   Class I replication
scripts/class3_final.py        Class III, per-dataset effects retained
scripts/capacity_measure.py    random-label capacity measurement
scripts/statistics.py          paired tests, Holm correction, bootstrap
scripts/figures.py             figures 1-3
docs/paper.pdf             the paper (63 references)
```

| Script | Output | Paper |
|---|---|---|
| `class1_estimation.py` | `results/class1_estimation.json` | §3, Fig 1 |
| `class3_final.py` | `results/class3_per_dataset.json` | §4, Table 1 |
| `capacity_measure.py` | `results/capacity_measured.json` | §6 |
| `statistics.py` | `results/statistics.json` | §5 |

## Statistics

Unit of analysis is the **dataset**, not the row, so every test is paired across
the 15 datasets — the conservative choice. Wilcoxon signed-rank rather than
paired t, because effects are bounded and not obviously normal. Rank-biserial
correlation as effect size. Holm correction for multiplicity. Bootstrap
percentile intervals, 5,000 resamples.

## What this does not establish

- Two of four classes reproduced. Class II (selection) and Class IV (boundary)
  are untouched.
- 11 learners is a small sample for a regression, and the index is fitted across
  model classes, not within one.
- All datasets are tabular UCI classification problems under 18,000 rows.
- Duplication here is exact; near-duplicates are the more common real failure.
- Grouped CV is the comparator throughout, and cross-validation estimates a
  subtler quantity than usually assumed (Bates, Hastie & Tibshirani 2024).

## Citation note

The 63 references in `docs/paper.tex` were assembled from primary sources and
literature searches. 

## License

MIT for the code. UCI datasets belong to their contributors and are not
redistributed.

## Reference verification

All 63 references were queried programmatically against the CrossRef DOI
registry and Semantic Scholar (`scripts/verify_bib.py`, `verify_bib2.py`).
**35 confirmed against publisher metadata; 28 unresolved** — mostly CS
conference papers that CrossRef indexes poorly, plus three books whose titles
the parser could not extract. Status per entry: `results/bib_verification.json`.
Per-reference notes and the recommendation to trim: `docs/NOTES_AND_OUTREACH.md`.


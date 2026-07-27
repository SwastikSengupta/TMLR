# Reference Notes & Outreach

Companion to *How Loud Is the Leak?* Two things: the verification status of every
reference with a one-line note, and the outreach package for the author whose
work I reproduced.

---

## Part 1 — Contact and what to say

### Simon Roth

- **Affiliation:** Independent Researcher / **Epagogy** (per his Google Scholar profile)
- **Google Scholar:** `scholar.google.com/citations?user=GCUoyQgAAAAJ`
- **Site:** `epagogy.ai/papers/`
- **Interests listed:** Machine Learning, Reproducibility, Cross-Validation, Type Systems
- **Email:** not published anywhere I can verify. Get it from the **"view email"
  link on the arXiv abstract page** (arxiv.org/abs/2604.04199, under Submission
  History) or through epagogy.ai. **Do not guess an address.**
- **Papers:** arXiv:2604.04199 (landscape, 35pp, 6 figs, 10 tables) and
  arXiv:2603.10742 (grammar, with Python and R reference implementations)

**Why he is an unusually good person to email.** He is independent, so there is
no gatekeeper. He works on reproducibility, so someone reproducing him is
directly aligned with his stated interest. His paper is four months old and
uncited. And he ships code in two languages, which signals he cares whether
people can actually use his work.

### One thing to read before you write

His paper contains a result I only found while looking him up: *standard CV
confidence intervals achieve only 55% actual coverage at nominal 95%.* This is
the same phenomenon Bates, Hastie & Tibshirani address, and it is almost
certainly where that miscoverage table you were holding came from. **Find that
section in his paper before you email.** If the table came from him, citing it
back to him correctly is a strong signal. If it did not, you have avoided a
serious error.

### The email

Subject: **Independent reproduction of your leakage landscape — Class I confirms, Class III made quantitative**

> Dear Dr Roth,
>
> I reproduced two of the four class-level claims in "Which Leakage Types
> Matter?" on fifteen UCI datasets outside your corpus.
>
> Class I replicates exactly. Across seventy conditions, three classifiers and
> two preprocessing operations, mean |ΔAUC| is 0.00018 and the maximum is 0.0032.
> Every condition falls inside your 0.005 bound.
>
> Class III replicates in ordering and magnitude: d_z runs from 0.31 for naive
> Bayes to 1.23 for 1-NN, against your 0.37 and 1.11.
>
> I then tried to unseat your capacity account. My hypothesis was locality,
> meaning whether a single training point can dominate its own prediction, since
> k-NN is maximally local with no fitted parameters. It failed. With capacity
> measured rather than assumed, locality carries no weight (p = 0.63) and
> reverses sign.
>
> What replaced it may interest you. Defining capacity as each learner's training
> AUC on permuted labels, that index predicts Class III severity across eleven
> learners at Spearman ρ = 0.982 (p = 8.4e-08), Pearson r = 0.912, bootstrap CI
> [0.834, 0.984]. Your qualitative claim becomes a quantity a practitioner can
> compute on their own pipeline in one line.
>
> Code and results: [your GitHub link]
>
> Two questions. Did you consider a measured capacity index rather than family
> comparison? And does your 55% coverage result connect to the Bates, Hastie and
> Tibshirani account of what cross-validation estimates?
>
> Swastik Sengupta
> Las Positas College

**Why this email works.** It leads with confirmation, not criticism. It reports a
failed hypothesis, which almost nobody does and which signals you are not
grinding an axe. It hands him something usable. And it asks two real questions
rather than requesting anything.

**Do not** ask for a recommendation, mention admissions, or attach a PDF in the
first email. Link, don't attach. One follow-up at day ten, never a third.

### Second target, after Roth replies

**Stephen Bates** (Berkeley/MIT lineage; co-author with Hastie and Tibshirani on
"Cross-Validation: What Does It Estimate and How Well Does It Do It?", JASA
119(546), 2024). His group is the right place to take the coverage question. His
students are the junior-but-serious profile worth emailing. Only write once you
can state your own result cleanly, and never send numbers you cannot trace to a
script you own.

---

## Part 2 — Reference verification

**Method:** every `\bibitem` title was queried against the CrossRef DOI registry,
then unresolved entries were re-queried against Semantic Scholar. This compares
against publisher metadata rather than memory.

**Result: 35 of 63 confirmed programmatically. 28 unresolved.**

The unresolved ones are mostly **not** errors. CrossRef indexes CS conference
proceedings (NeurIPS, ICML, ICLR, JMLR) poorly, the Semantic Scholar endpoint
rate-limited partway through, and three entries are books with no quoted title
for the parser to extract. But unresolved is unresolved.

### My recommendation: cut the bibliography to about 40

Sixty-three references on a five-page reproduction reads as padding, and a
reviewer will notice. Cutting to the verified core makes the paper *better* and
removes the risk entirely. Everything in the table below marked **VERIFIED** can
stay as is. Everything marked **CHECK** either gets verified by hand in five
minutes or gets cut.

Full machine-readable status: `results/bib_verification.json`

### Verified, with one-line notes

| Key | One line |
|---|---|
| roth2026 | The paper I reproduce: four leakage classes across 2,047 datasets, textbook emphasis inverted. |
| roth2026b | His companion: a type grammar that makes the worst leakage structurally impossible to express. |
| kapoor2023 | Found leakage in hundreds of papers across 17 fields; the survey that made this a crisis. |
| kaufman2012 | The original formal definition of leakage in data mining. |
| ambroise2002 | Selecting genes before cross-validating produced near-perfect accuracy on noise. The founding cautionary tale. |
| simon2003 | Same failure in cancer microarray classification, from the NCI. |
| varoquaux2018 | Small samples give enormous CV error bars; neuroimaging's version of the problem. |
| varoquaux2017 | Practical guidance on cross-validating brain decoders without fooling yourself. |
| beam2020 | Clinical ML reproducibility failures, in JAMA. |
| roberts2017 | Why data with spatial, temporal, or hierarchical structure needs blocked CV. My grouping justification. |
| bates2024 | Cross-validation does not estimate what you think it does. Complicates my own baseline. |
| zhang2021 | Fitting random labels as a capacity probe. The direct source of my index. |
| varma2006 | Using CV for both tuning and evaluation biases the estimate; use nested CV. |
| arlot2010 | The standard survey of cross-validation procedures. |
| dua2019 | The UCI repository, source of all fifteen datasets. |
| vanschoren2014 | OpenML; how benchmark suites are curated and shared. |
| dacrema2019 | Most neural recommender gains vanished under honest baselines. |
| musgrave2020 | Metric learning progress was largely an evaluation artifact. |
| gundersen2022 | Taxonomy of every way ML results fail to reproduce. |
| dwork2015 | Reusing a holdout adaptively destroys its validity; the reusable holdout fix. |
| bergmeir2012 | Why random CV is wrong for time series. Relevant to Class IV, which I did not test. |
| barz2020 | CIFAR contains near-duplicates across train and test. Duplication leakage in the wild. |
| bousquet2002 | Algorithmic stability and generalization; the theory behind stability arguments. |
| friedman2001 | Gradient boosting, one of my learners. |
| cover1967 | Nearest-neighbour classification; the learner with the highest leakage in my results. |
| stone1974 | The paper that introduced cross-validation. |
| poldrack2020 | Best practices for prediction claims in psychiatry. |
| whalen2022 | Pitfalls of ML in genomics, in Nature Reviews Genetics. |
| teschendorff2019 | Common ML mistakes in omics data. |
| chicco2017 | Ten practical tips for ML in computational biology. |
| lones2021 | A widely circulated checklist of ML pitfalls. |
| hullman2022 | Compares error patterns in psychology and ML; Kapoor and Narayanan are co-authors. |
| paullada2021 | How datasets are built, reused, and quietly break. |
| bouthillier2021 | Benchmark variance is larger than most reported improvements. |
| breiman2001 | Random forests, my main capacity ladder. |

### Needs manual verification before submission

`alomar2025`, `recht2019`, `cawley2010`, `bengio2004`, `dietterich1998`,
`demsar2006`, `holm1979`, `efron1993`, `hastie2009`, `pedregosa2011`,
`buitinck2013`, `ucirepo`, `bischl2021`, `geirhos2020`, `lipton2019`,
`sculley2018`, `henderson2018`, `melis2018`, `pineau2021`, `raff2019`,
`ioannidis2005`, `blum2015`, `guyon2003`, `nogueira2018`, `kohavi1995`,
`liao2021`, `koch2021`, `mcdermott2021`

Most are canonical and almost certainly correct, but "almost certainly" is not a
standard you can defend. Five minutes each on Google Scholar, or cut them.

**One known parser artifact:** `dietterich1998` was flagged for a year mismatch
because my extractor read the page range "1895--1923" and took 1923 as the year.
The citation itself is fine.

---

## Part 3 — Sequence

1. Verify or cut the 28. Do not submit with unresolved references.
2. Push the repo public, get the link.
3. Read Roth's coverage section, resolve where that miscoverage table came from.
4. Send the email above.
5. Wait for his reply before contacting Bates. If Roth engages, you can mention
   the exchange, and that makes the second email far stronger.

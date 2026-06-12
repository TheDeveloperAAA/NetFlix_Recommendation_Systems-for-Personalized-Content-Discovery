# Methodology & Design Decisions

This document explains *why* the project is built the way it is. For the
narrative report see the Technical Report PDF; this is the engineering companion.

---

## 1. Parsing the raw format

The `combined_data_*.txt` files are **not** CSVs. A line ending in `:` is a
movie‑ID header; the rows beneath it belong to that movie until the next header:

```
1:
1488844,3,2005-09-06
822109,5,2005-05-13
2:
...
```

Reading this as a flat CSV would attach every rating to the wrong movie. The
parser (`src/data_processing.parse_combined_file`) reads the file flat, finds the
header rows (where `Rating` parses as `NaN`), and uses `deque.rotate` to slice
each movie's block in a single vectorized pass — much faster than a Python
line‑by‑line loop. Ratings are stored with compact dtypes (`int32`/`int16`/`int8`),
roughly 3× smaller than the defaults, which keeps the full file within a free
16 GB kernel.

## 2. Smart subset

The full user × movie matrix is ~480K × 18K ≈ 8.5 **billion** cells, ~99% empty.
Training on all 100M ratings is slow and memory‑heavy, and the problem statement
explicitly permits subsetting. We keep only users and movies with enough ratings
(defaults: movies ≥ 3000, users ≥ 200), iterating a few times because filtering
one dimension can push the other below threshold.

This is a **deliberate modelling choice**, not just a speed hack: the removed rows
are the extreme cold‑start tail, where there is too little signal to learn from.
The real run keeps **1,730,647 of 24,053,764 ratings** (7.2%), giving a dense
7,198 × 419 working set at 42.62% sparsity.

## 3. Temporal train/test split

For each user, ratings are sorted by date and the **most‑recent 20%** are held out
for testing. This mirrors how Netflix built the original competition's qualifying
set — *predict future preferences from past behaviour* — and avoids the look‑ahead
leakage a random split would cause (training on a user's later ratings to predict
earlier ones inflates every metric). Test rows whose user or movie is unseen in
training are dropped, since a model cannot score an item it never observed.
Result: **1,387,301 train / 343,346 test**.

## 4. The three models

Chosen to span **three distinct paradigms**, so the comparison is informative
rather than three flavours of one idea.

### Weighted‑Mean (non‑personalized baseline)
Damped average rating (IMDb formula):

```
WR = [v/(v+m)] · R  +  [m/(v+m)] · C
```

where `R` = movie mean, `v` = movie rating count, `C` = global mean, `m` = damping
(1000). The damping stops a movie with a few high ratings from outranking an
established classic. Every user gets the same list — the bar personalized models
must beat, especially on ranking.

### Matrix Factorization (SVD via SGD)
```
r_hat(u,i) = μ + b_u + b_i + p_u · q_i
```
Latent vectors `p_u`, `q_i` capture taste affinity; bias terms capture rating
tendencies. Trained with regularized SGD from scratch (40 factors, lr 0.005, reg
0.02, 15 epochs). The Netflix‑Prize workhorse; best accuracy here. Validation RMSE
fell monotonically **0.8893 → 0.8146** over the 15 epochs (≈357 s).

### Item‑based k‑NN (and the explainability engine)
Item–item cosine similarity on **mean‑centered** ratings (centering removes each
movie's popularity offset so similarity reflects co‑taste, not shared popularity).
A user's rating for a movie is the similarity‑weighted average of their ratings on
that movie's 40 nearest neighbours. Slightly less accurate than MF, but
**interpretable**: the neighbours driving a prediction are exactly the "because you
watched X" reasons.

## 5. Evaluation — two metrics, on purpose

- **RMSE** measures rating‑prediction accuracy.
- **MAP@10** measures ranking quality: of the 10 items ranked highest for a user,
  how many are relevant (rating ≥ 3.5) and are they ordered well?

**MAP@10 procedure:** for each sampled test user, take their test‑period items,
mark those rated ≥ 3.5 as relevant, rank all of them by the model's predicted
score, compute Average Precision@10, and average across 2,000 sampled users.

## 6. The central finding

| Model | RMSE | MAP@10 |
|---|---|---|
| Weighted Mean | 0.9661 | 0.7784 |
| Matrix Factorization | 0.8146 | 0.8497 |
| Item‑based k‑NN | 0.8334 | 0.8488 |

The Weighted‑Mean baseline posts a not‑unreasonable RMSE (0.9661) — given the
positive rating skew, predicting "everyone rates a popular film ~4.2" is rarely far
off — yet its **MAP@10 collapses to 0.7784**, far below the personalized models
(~0.849). The reason is structural: a non‑personalized model hands every user the
**identical list**, so it cannot order any individual's preferences well, however
calibrated its point predictions are.

**Takeaway:** optimize for the metric that matches the goal. Content discovery is a
*ranking* problem, so MAP@10 — not RMSE alone — should drive model choice, and
personalization is what lifts it.

## 7. Recommendation generation

A hybrid that uses each model for what it's best at: **Matrix Factorization ranks**
the candidate items (best accuracy), **Item‑k‑NN explains** each pick
(interpretability). Each recommendation carries a human‑readable reason — the same
pattern real streaming products use to build user trust.

## 8. Design principles carried throughout

- **Dependency‑light.** All three models use only NumPy / SciPy / scikit‑learn, so
  the pipeline runs even where heavier libraries (e.g. `surprise`) fail to install
  — a real issue observed on Kaggle (NumPy 2.x incompatibility).
- **Reproducible.** Fixed seeds for models and sampling; deterministic given the
  same data and thresholds.
- **Honest evaluation.** The temporal split and MAP@10 make the numbers trustworthy
  and aligned with the real product goal.

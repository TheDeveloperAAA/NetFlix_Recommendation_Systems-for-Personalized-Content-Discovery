"""
evaluation.py
=============
Deliverable component: EVALUATION SCRIPTS

Two metrics, because they measure different things:

  * RMSE   - rating-prediction accuracy (how close are predicted ratings?)
  * MAP@10 - ranking quality (are the right items ranked at the top?)

A model can win on one and lose on the other; documenting that trade-off is the
central analytical finding of this project.

These are the exact evaluation routines used in the project notebook.

Real-run results (held-out temporal test set):
  Model                  RMSE     MAP@10
  Weighted Mean          0.9661   0.7784
  Matrix Factorization   0.8146   0.8497
  Item-based k-NN        0.8334   0.8488
"""

import numpy as np
from sklearn.metrics import mean_squared_error

RANDOM_SEED = 42


def rmse(y_true, y_pred):
    """Root Mean Squared Error between true and predicted ratings."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def average_precision_at_k(relevance, k=10):
    """Average Precision @ k for a single ranked list.

    Parameters
    ----------
    relevance : list[int]
        0/1 relevance flags in predicted-rank order (index 0 = highest ranked).
    k : int

    Returns
    -------
    float
        AP@k. Returns 0.0 if there are no relevant items in the top-k.
    """
    relevance = relevance[:k]
    if sum(relevance) == 0:
        return 0.0
    score = hits = 0
    for idx, rel in enumerate(relevance):
        if rel:
            hits += 1
            score += hits / (idx + 1)         # precision at this hit's rank
    return score / min(sum(relevance), k)


def evaluate_map(predict_fn, test_df, top_k=10, relevance_thresh=3.5,
                 eval_users=2000, seed=RANDOM_SEED, label=""):
    """Mean Average Precision @ k over a sample of test users.

    Procedure (matches the problem statement's MAP@10 definition):
      1. For each sampled test user, take the items they rated in the test set.
      2. Mark items with true rating >= `relevance_thresh` as relevant.
      3. Score those items with the model and rank them by predicted score.
      4. Compute AP@k on that ranking; average across users -> MAP@k.

    Parameters
    ----------
    predict_fn : callable(u_array, i_array) -> score_array
        A model's predict method (or any function with that signature).
    test_df : DataFrame
        Must contain columns 'u' (user index), 'i' (movie index), 'Rating'.
    top_k : int
        K for MAP@K (default 10).
    relevance_thresh : float
        Rating at/above which an item counts as relevant (default 3.5).
    eval_users : int
        Sample size of test users (full set is slow). Default 2000.

    Returns
    -------
    float
        MAP@k over the sampled users.
    """
    rng = np.random.default_rng(seed)
    users = test_df["u"].unique()
    if len(users) > eval_users:
        users = rng.choice(users, eval_users, replace=False)

    aps = []
    for a, g in test_df[test_df["u"].isin(users)].groupby("u"):
        items, truth = g["i"].values, g["Rating"].values
        if len(items) < 2:
            continue
        preds = predict_fn(np.full(len(items), a), items)        # score this user's items
        order = np.argsort(-preds)                                # rank by predicted score
        ranked_rel = (truth[order] >= relevance_thresh).astype(int).tolist()
        aps.append(average_precision_at_k(ranked_rel, top_k))

    m = float(np.mean(aps)) if aps else 0.0
    if label:
        print(f"  MAP@{top_k} [{label}] = {m:.4f}  ({len(aps):,} users)")
    return m


def evaluate_rmse_sample(predict_fn, test_u, test_i, test_r,
                         sample_size=20000, seed=RANDOM_SEED):
    """RMSE on a random sample of test pairs.

    Used for models whose per-pair prediction is slow (e.g. Item-kNN); for fast
    models, simply call rmse() on the full test set instead.
    """
    rng = np.random.default_rng(seed)
    n = len(test_u)
    idx = rng.choice(n, min(sample_size, n), replace=False)
    preds = predict_fn(test_u[idx], test_i[idx])
    return rmse(test_r[idx], preds)

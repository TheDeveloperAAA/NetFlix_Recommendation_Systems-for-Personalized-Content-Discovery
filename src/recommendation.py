"""
recommendation.py
=================
Deliverable component: RECOMMENDATION GENERATION MODULE

Generates personalized Top-K recommendations with human-readable explanations,
using a simple, effective hybrid:

  * Matrix Factorization RANKS the candidate items (best accuracy)
  * Item-based k-NN EXPLAINS each pick ("because you watched X, Y")

This is the same recommendation logic used in the project notebook, plus the
dashboard-export helper that writes the JSON consumed by dashboard/index.html.

Example real output (User 0):
  Seven Samurai (pred 4.67)  <- because you liked The Godfather, This Is Spinal Tap
  North by Northwest (pred 4.39) <- because you liked The Odd Couple, The Godfather
"""

import json
from pathlib import Path

import numpy as np


def make_title_lookup(movie_titles, idx_to_movie):
    """Build a function internal_movie_index -> human-readable title.

    Parameters
    ----------
    movie_titles : DataFrame
        Indexed by movie Id, with a 'Name' column (from load_titles).
    idx_to_movie : dict
        internal movie index -> original movie Id (from remap_indices).

    Returns
    -------
    callable(i) -> str
    """
    name_by_id = movie_titles["Name"].to_dict()

    def mname(i):
        return name_by_id.get(idx_to_movie[i], f"Movie#{idx_to_movie[i]}")

    return mname


def explained_recommendations(uidx, mf_model, knn_model, train_df, mname,
                              top_k=10, n_reasons=3):
    """Generate Top-K explained recommendations for one user.

    Matrix Factorization ranks; Item-kNN supplies the reasons.

    Parameters
    ----------
    uidx : int
        Internal user index.
    mf_model : MatrixFactorization (fitted)
    knn_model : ItemKNN (fitted)
    train_df : DataFrame
        Training interactions (needs 'u' and 'i' columns) to find already-seen items.
    mname : callable
        Internal movie index -> title (from make_title_lookup).
    top_k : int
    n_reasons : int
        Number of "because you liked" reasons per recommendation.

    Returns
    -------
    list[dict]
        Each: {movie, predicted_rating, because_you_liked: [titles]}.
    """
    known = set(train_df[train_df["u"] == uidx]["i"].values)     # movies already seen
    recs = mf_model.recommend(uidx, known, top_k=top_k)          # MF picks Top-K

    out = []
    for b in recs:
        reasons = [mname(j) for (j, sim, ruj) in knn_model.explain(uidx, b, n_reasons)]
        pred = round(float(mf_model.predict(np.array([uidx]), np.array([b]))[0]), 2)
        out.append({"movie": mname(b),
                    "predicted_rating": pred,
                    "because_you_liked": reasons})
    return out


def print_sample_recommendations(test_df, train_df, mf_model, knn_model, mname,
                                 n_users=3, top_k=5, min_history=20):
    """Print explained recommendations for a few demo users (console output).

    Mirrors the notebook's demonstration cell. Picks users with enough history
    so the explanations are meaningful.
    """
    demo = [u for u in test_df["u"].unique()[:60]
            if len(train_df[train_df["u"] == u]) >= min_history][:n_users]
    for du in demo:
        hist = train_df[train_df["u"] == du].sort_values("Rating", ascending=False).head(5)
        print("=" * 70)
        print(f"User idx {du}  |  top-rated so far: {[mname(i) for i in hist['i'].values]}")
        print("  Recommended for them:")
        for rec in explained_recommendations(du, mf_model, knn_model, train_df, mname, top_k):
            why = ", ".join(rec["because_you_liked"][:2]) or "general popularity"
            print(f"    - {rec['movie']}  (pred {rec['predicted_rating']})")
            print(f"        because you liked: {why}")


def export_dashboard_data(out_path, test_df, train_df, mf_model, knn_model, mname,
                          results_records, n_users_total, n_movies_total, n_ratings_total,
                          n_sample=40, top_k=10, min_history=20, seed=42):
    """Write dashboard_data.json consumed by dashboard/index.html.

    Exports a sample of users (not all) to keep the file small: each user's
    rating history plus their explained recommendations, alongside summary stats
    and the final model-comparison table.

    Parameters
    ----------
    out_path : str or Path
        Where to write the JSON.
    results_records : list[dict]
        results.round(4).to_dict("records") from the evaluation step.
    n_users_total, n_movies_total, n_ratings_total : int
        Dataset summary numbers for the dashboard header.
    """
    rng = np.random.default_rng(seed)
    eligible = [u for u in test_df["u"].unique()
                if len(train_df[train_df["u"] == u]) >= min_history]
    chosen = rng.choice(eligible, min(n_sample, len(eligible)), replace=False)

    payload = {
        "stats": {"n_users": int(n_users_total),
                  "n_movies": int(n_movies_total),
                  "n_ratings": int(n_ratings_total),
                  "models": results_records},
        "users": [],
    }
    for du in chosen:
        hist = train_df[train_df["u"] == du].sort_values("Rating", ascending=False)
        payload["users"].append({
            "id": int(du),
            "history": [{"title": mname(i), "rating": int(r)}
                        for i, r in zip(hist["i"].values[:8], hist["Rating"].values[:8])],
            "recommendations": explained_recommendations(
                int(du), mf_model, knn_model, train_df, mname, top_k),
        })

    out_path = Path(out_path)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out_path}  ({len(chosen)} users)")
    return payload

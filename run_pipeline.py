"""
run_pipeline.py
===============
End-to-end entry point that reproduces the full project from the command line,
tying together all four components:

    data_processing -> models -> evaluation -> recommendation

Usage
-----
    # On Kaggle (data attached) or locally with the path set in CONFIG below:
    python run_pipeline.py

This produces, in results/:
    - results_table.csv           the final RMSE / MAP@10 table
    - figures/*.png               EDA + comparison charts
    - ../dashboard/dashboard_data.json   data for the interactive dashboard

The canonical executed run (with real outputs embedded) is the notebook in
notebooks/. This script is the headless, reproducible equivalent.

Real-run summary (combined_data_1.txt subset):
    24,053,764 ratings -> 1,730,647 subset | 7,198 users x 419 movies
    Weighted Mean        RMSE 0.9661 | MAP@10 0.7784
    Matrix Factorization RMSE 0.8146 | MAP@10 0.8497
    Item-based k-NN      RMSE 0.8334 | MAP@10 0.8488
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import data_processing as dp
from src import models as M
from src import evaluation as E
from src import recommendation as R

# ---------------------------------------------------------------- CONFIG
class CONFIG:
    # Point this at your Netflix Prize data directory. On Kaggle this is the
    # mount path shown in the left file browser (verify it — it can be nested).
    DATA_DIR = Path("../input/netflix-prize-data")
    RATING_FILES = ["combined_data_1.txt"]

    MIN_MOVIE_RATINGS = 3000
    MIN_USER_RATINGS = 200
    TEST_FRACTION = 0.20

    MF_FACTORS = 40
    MF_EPOCHS = 15
    MF_LR = 0.005
    MF_REG = 0.02
    KNN_K = 40

    TOP_K = 10
    RELEVANCE_THRESH = 3.5
    EVAL_USERS = 2000

    RESULTS_DIR = Path("results")
    FIG_DIR = Path("results/figures")
    DASHBOARD_JSON = Path("dashboard/dashboard_data.json")

    SEED = 42

NETFLIX_RED = "#db0000"


def main():
    cfg = CONFIG
    cfg.RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    cfg.FIG_DIR.mkdir(exist_ok=True, parents=True)
    cfg.DASHBOARD_JSON.parent.mkdir(exist_ok=True, parents=True)
    np.random.seed(cfg.SEED)

    # ---- 1. DATA PROCESSING ------------------------------------------------
    print(">> Loading & parsing ratings...")
    t0 = time.time()
    df = dp.load_ratings(cfg.DATA_DIR, cfg.RATING_FILES)
    print(f"   {len(df):,} ratings parsed in {time.time()-t0:.1f}s")
    titles = dp.load_titles(cfg.DATA_DIR)
    df = dp.smart_subset(df, cfg.MIN_MOVIE_RATINGS, cfg.MIN_USER_RATINGS)

    # ---- EDA figures -------------------------------------------------------
    _eda_figures(df, cfg)

    df, n_users, n_movies, idx_to_movie = dp.remap_indices(df)
    print(f"   n_users={n_users:,}  n_movies={n_movies:,}")
    train, test = dp.temporal_split(df, cfg.TEST_FRACTION)

    tr_u, tr_i = train["u"].values, train["i"].values
    tr_r = train["Rating"].values.astype(np.float32)
    te_u, te_i = test["u"].values, test["i"].values
    te_r = test["Rating"].values.astype(np.float32)

    # ---- 2. MODEL TRAINING -------------------------------------------------
    print("\n>> Training models...")
    wm = M.WeightedMean(n_movies, damping=1000).fit(tr_u, tr_i, tr_r)
    wm_rmse = E.rmse(te_r, wm.predict(te_u, te_i))
    print(f"   Weighted-Mean       RMSE = {wm_rmse:.4f}")

    mf = M.MatrixFactorization(n_users, n_movies, cfg.MF_FACTORS, cfg.MF_EPOCHS,
                               cfg.MF_LR, cfg.MF_REG, cfg.SEED)
    mf.fit(tr_u, tr_i, tr_r, val=(te_u, te_i, te_r))
    mf_rmse = E.rmse(te_r, mf.predict(te_u, te_i))
    print(f"   MatrixFactorization RMSE = {mf_rmse:.4f}")

    knn = M.ItemKNN(n_users, n_movies, cfg.KNN_K).fit(tr_u, tr_i, tr_r)
    knn_rmse = E.evaluate_rmse_sample(knn.predict, te_u, te_i, te_r)
    print(f"   Item-kNN            RMSE = {knn_rmse:.4f} (sample)")

    # ---- 3. EVALUATION -----------------------------------------------------
    print("\n>> Computing MAP@10...")
    wm_map = E.evaluate_map(wm.predict, test, cfg.TOP_K, cfg.RELEVANCE_THRESH,
                            cfg.EVAL_USERS, cfg.SEED, "Weighted-Mean")
    mf_map = E.evaluate_map(mf.predict, test, cfg.TOP_K, cfg.RELEVANCE_THRESH,
                            cfg.EVAL_USERS, cfg.SEED, "Matrix-Fact")
    knn_map = E.evaluate_map(knn.predict, test, cfg.TOP_K, cfg.RELEVANCE_THRESH,
                             cfg.EVAL_USERS, cfg.SEED, "Item-kNN")

    results = pd.DataFrame({
        "Model": ["Weighted Mean", "Matrix Factorization", "Item-based kNN"],
        "Test RMSE": [wm_rmse, mf_rmse, knn_rmse],
        "MAP@10": [wm_map, mf_map, knn_map],
    })
    results.to_csv(cfg.RESULTS_DIR / "results_table.csv", index=False)
    print("\n" + "=" * 52)
    print(results.round(4).to_string(index=False))
    print("=" * 52)
    _results_figure(results, cfg)

    # ---- 4. RECOMMENDATION GENERATION -------------------------------------
    print("\n>> Generating explained recommendations...")
    mname = R.make_title_lookup(titles, idx_to_movie)
    R.print_sample_recommendations(test, train, mf, knn, mname)

    R.export_dashboard_data(
        cfg.DASHBOARD_JSON, test, train, mf, knn, mname,
        results_records=results.round(4).to_dict("records"),
        n_users_total=n_users, n_movies_total=n_movies, n_ratings_total=len(df),
    )
    print("\nDone. Artifacts in results/ and dashboard/.")


def _eda_figures(df, cfg):
    """Three EDA charts (rating distribution, long tails, temporal)."""
    # rating distribution
    data = df["Rating"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(data.index, data.values, color=NETFLIX_RED)
    ax.set_title(f"Distribution of {len(df):,} Ratings")
    ax.set_xlabel("Star Rating"); ax.set_ylabel("Count")
    plt.tight_layout(); plt.savefig(cfg.FIG_DIR / "eda_rating_distribution.png", bbox_inches="tight"); plt.close()

    # long tails
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    mc = df.groupby("Movie")["Rating"].count()
    ax[0].hist(mc.clip(upper=mc.quantile(0.99)).values, bins=50, color=NETFLIX_RED)
    ax[0].set_title("Ratings per Movie"); ax[0].set_yscale("log")
    uc = df.groupby("User")["Rating"].count()
    ax[1].hist(uc.clip(upper=uc.quantile(0.99)).values, bins=50, color="#7a0000")
    ax[1].set_title("Ratings per User"); ax[1].set_yscale("log")
    plt.tight_layout(); plt.savefig(cfg.FIG_DIR / "eda_long_tails.png", bbox_inches="tight"); plt.close()

    # temporal
    daily = df.set_index("Date").resample("ME").size()
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(daily.index, daily.values, color=NETFLIX_RED)
    ax.set_title("Rating Volume Over Time")
    plt.tight_layout(); plt.savefig(cfg.FIG_DIR / "eda_temporal.png", bbox_inches="tight"); plt.close()


def _results_figure(results, cfg):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    colors = [NETFLIX_RED, "#a30000", "#7a0000"]
    ax[0].bar(results["Model"], results["Test RMSE"], color=colors)
    ax[0].set_title("RMSE (lower = better)"); ax[0].tick_params(axis="x", rotation=15)
    ax[1].bar(results["Model"], results["MAP@10"], color=colors)
    ax[1].set_title("MAP@10 (higher = better)"); ax[1].tick_params(axis="x", rotation=15)
    plt.tight_layout(); plt.savefig(cfg.FIG_DIR / "results_comparison.png", bbox_inches="tight"); plt.close()


if __name__ == "__main__":
    main()

"""
data_processing.py
==================
Deliverable component: DATA PROCESSING PIPELINE

Handles everything from raw Netflix Prize files to a clean, model-ready table:
  1. parse_combined_file  - parse the raw combined_data_*.txt format
  2. load_titles          - load movie_titles.csv (handles encoding quirks)
  3. smart_subset         - filter to active users / popular movies
  4. remap_indices        - compress sparse IDs to contiguous integers

This is the exact pipeline used in the project notebook, refactored into
importable functions. See notebooks/netflix_recommender.ipynb for the executed
run with real outputs.

Real run reference (combined_data_1.txt):
  24,053,764 ratings parsed -> 1,730,647 after subset
  7,198 users x 419 movies, sparsity 42.62%
"""

import gc
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd


def parse_combined_file(filepath):
    """Parse one raw combined_data_*.txt file into a tidy DataFrame.

    The raw format is NOT a CSV. A line ending in ':' is a movie-ID header, and
    every row beneath it (until the next header) is `CustomerID,Rating,Date` for
    that movie:

        1:
        1488844,3,2005-09-06
        822109,5,2005-05-13
        2:
        ...

    Reading it as a flat CSV would attach every rating to the wrong movie. We
    locate header rows (where Rating parses as NaN), pair each header's index
    with the next header's index using deque.rotate, and slice each movie's block
    in a single vectorized pass.

    Parameters
    ----------
    filepath : str or Path
        Path to a combined_data_*.txt file.

    Returns
    -------
    pandas.DataFrame
        Columns: User (int32), Rating (int8), Date (datetime64), Movie (int16).
    """
    # Read flat. Header rows ("1:") land in 'User' with a NaN Rating.
    df_raw = pd.read_csv(filepath, header=None,
                         names=["User", "Rating", "Date"], usecols=[0, 1, 2])

    # Index + movie-id of each header row ("1:" -> 1).
    tmp = df_raw[df_raw["Rating"].isna()]["User"].reset_index()
    movie_indices = [[idx, int(m[:-1])] for idx, m in tmp.values]

    # Rotate by one to get each block's [start, end) boundaries.
    shifted = deque(movie_indices)
    shifted.rotate(-1)

    blocks = []
    for [start, movie_id], [end, _next] in zip(movie_indices, shifted):
        blk = df_raw.loc[start + 1:end - 1] if start < end else df_raw.loc[start + 1:]
        blk = blk.copy()
        blk["Movie"] = movie_id           # stamp the movie id onto its rows
        blocks.append(blk)

    df = pd.concat(blocks)
    del df_raw, tmp, movie_indices, shifted, blocks
    gc.collect()

    # Compact dtypes: ~3x smaller than defaults, critical on a 16GB kernel.
    df["User"]   = df["User"].astype(np.int32)
    df["Movie"]  = df["Movie"].astype(np.int16)
    df["Rating"] = df["Rating"].astype(np.int8)
    df["Date"]   = pd.to_datetime(df["Date"])
    return df


def load_ratings(data_dir, rating_files):
    """Parse and concatenate one or more combined_data_*.txt files.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing the rating files.
    rating_files : list[str]
        Filenames to parse, e.g. ["combined_data_1.txt"].

    Returns
    -------
    pandas.DataFrame
    """
    data_dir = Path(data_dir)
    frames = []
    for f in rating_files:
        frames.append(parse_combined_file(data_dir / f))
    df = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()
    return df


def load_titles(data_dir):
    """Load movie_titles.csv -> DataFrame indexed by movie Id.

    Uses ISO-8859-1 encoding (some titles contain non-UTF-8 characters) and
    tolerant column parsing (some titles contain commas). Movie IDs are
    sequential 1..17770 and do NOT correspond to IMDb/Netflix IDs.

    Returns
    -------
    pandas.DataFrame
        Indexed by 'Id', columns ['Year', 'Name'].
    """
    return pd.read_csv(Path(data_dir) / "movie_titles.csv", encoding="ISO-8859-1",
                       header=None, names=["Id", "Year", "Name"],
                       usecols=[0, 1, 2]).set_index("Id")


def smart_subset(df, min_movie_ratings=3000, min_user_ratings=200, verbose=True):
    """Filter to active users and popular movies.

    The full user x movie matrix is enormous and ~99% empty. Subsetting (which
    the problem statement explicitly permits) removes the noisiest cold-start
    tail while keeping the dense, learnable core. Filtering one dimension can
    push the other below threshold, so we iterate a few times until stable.

    Parameters
    ----------
    df : DataFrame with columns User, Movie, Rating, Date.
    min_movie_ratings : int
        Keep movies with at least this many ratings.
    min_user_ratings : int
        Keep users with at least this many ratings.

    Returns
    -------
    pandas.DataFrame
        The filtered subset (index reset).
    """
    before = len(df)
    for _ in range(3):
        keep_m = df["Movie"].value_counts()
        df = df[df["Movie"].isin(keep_m[keep_m >= min_movie_ratings].index)]
        keep_u = df["User"].value_counts()
        df = df[df["User"].isin(keep_u[keep_u >= min_user_ratings].index)]
    df = df.reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("Subset removed all rows; lower the thresholds.")

    if verbose:
        sparsity = 1 - len(df) / (df["User"].nunique() * df["Movie"].nunique())
        print(f"Subset: {before:,} -> {len(df):,} ({100*len(df)/before:.1f}% kept)")
        print(f"  users={df['User'].nunique():,}  movies={df['Movie'].nunique():,}"
              f"  sparsity={100*sparsity:.2f}%")
    return df


def remap_indices(df):
    """Compress large sparse User/Movie IDs to contiguous 0..N-1 integers.

    Adds two columns: 'u' (internal user index) and 'i' (internal movie index),
    so IDs can be used directly as array/matrix positions.

    Returns
    -------
    (df, n_users, n_movies, idx_to_movie)
        df with added 'u','i' columns; counts; and a dict mapping internal movie
        index -> original movie Id (for joining titles back on).
    """
    user_ids = df["User"].unique()
    movie_ids = df["Movie"].unique()
    u_map = {x: i for i, x in enumerate(user_ids)}
    m_map = {x: i for i, x in enumerate(movie_ids)}
    idx_to_movie = {i: x for x, i in m_map.items()}

    df = df.copy()
    df["u"] = df["User"].map(u_map).astype(np.int32)
    df["i"] = df["Movie"].map(m_map).astype(np.int32)
    return df, len(user_ids), len(movie_ids), idx_to_movie


def temporal_split(df, test_fraction=0.20, verbose=True):
    """Leakage-free temporal train/test split.

    For each user, ratings are sorted by date and the most-recent `test_fraction`
    are held out for testing. This mirrors how Netflix built the original
    competition's qualifying set (predict future from past) and avoids the
    look-ahead leakage a random split would cause. Test rows whose user or movie
    is unseen in training are dropped (a model cannot score what it never saw).

    Real run: Train 1,387,301 | Test 343,346.

    Returns
    -------
    (train_df, test_df)
    """
    df = df.sort_values(["u", "Date"]).reset_index(drop=True)
    df["rank"] = df.groupby("u").cumcount()                       # 0 = user's oldest
    df["n"]    = df.groupby("u")["u"].transform("size")
    df["is_test"] = df["rank"] >= df["n"] * (1 - test_fraction)   # newest fraction
    tr, te = df[~df["is_test"]].copy(), df[df["is_test"]].copy()

    tr_u, tr_i = set(tr["u"]), set(tr["i"])
    te = te[te["u"].isin(tr_u) & te["i"].isin(tr_i)]

    if verbose:
        print(f"Train: {len(tr):,}  |  Test: {len(te):,}")
    return tr.reset_index(drop=True), te.reset_index(drop=True)

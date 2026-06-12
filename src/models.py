"""
models.py
=========
Deliverable component: MODEL TRAINING PIPELINE

The three recommendation models compared in this project, spanning three
distinct paradigms (not three variants of one idea):

  1. WeightedMean        - non-personalized baseline (IMDb damped-mean ranking)
  2. MatrixFactorization - latent-factor model, SGD, from scratch
  3. ItemKNN             - neighbourhood model; also the explainability engine

All three are dependency-light (NumPy / SciPy / scikit-learn only) so they run
in any environment without fragile library installs. These are the exact
implementations used in the project notebook.

Real-run results (held-out temporal test set, combined_data_1.txt subset):
  Weighted Mean        : RMSE 0.9661 | MAP@10 0.7784
  Matrix Factorization : RMSE 0.8146 | MAP@10 0.8497   (40 factors, 15 epochs)
  Item-based k-NN      : RMSE 0.8334 | MAP@10 0.8488   (k=40)
"""

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

RANDOM_SEED = 42


# ----------------------------------------------------------------------------
# Model 1 - Weighted-Mean Ranking (non-personalized baseline)
# ----------------------------------------------------------------------------
class WeightedMean:
    """Rank every movie by a damped average rating (the IMDb formula):

        WR = [v/(v+m)] * R  +  [m/(v+m)] * C

    where R = the movie's mean rating, v = its rating count, C = the global mean,
    and m = a damping constant. As v grows the score trusts the movie's own mean;
    when v is small it is pulled toward the global mean. Every user receives the
    same ranked list, so this is the reference personalized models must beat.
    """

    def __init__(self, n_movies, damping=1000):
        self.n_movies = n_movies
        self.damping = damping

    def fit(self, u, i, r):
        self.global_mean = r.mean()                              # C
        item_sum = np.bincount(i, weights=r, minlength=self.n_movies)
        item_cnt = np.bincount(i, minlength=self.n_movies)
        R = item_sum / np.maximum(item_cnt, 1)                   # per-movie mean
        v = item_cnt                                             # per-movie count
        m = self.damping
        self.WR = v / (v + m) * R + m / (v + m) * self.global_mean
        self.counts = v
        return self

    def predict(self, u, i):
        """Predicted rating = the movie's weighted rating (same for all users)."""
        return np.clip(self.WR[i], 1, 5)

    def top_movies(self, k=10):
        """Internal movie indices of the top-k movies by weighted rating."""
        return np.argsort(-self.WR)[:k]


# ----------------------------------------------------------------------------
# Model 2 - Matrix Factorization (SVD via SGD)
# ----------------------------------------------------------------------------
class MatrixFactorization:
    """Latent-factor model trained with regularized stochastic gradient descent.

    Predicted rating:
        r_hat(u,i) = mu + b_u + b_i + p_u . q_i

    where p_u and q_i are latent vectors capturing taste affinity, and the bias
    terms capture user/movie rating tendencies. This is the model family that
    powered the winning Netflix Prize solutions, implemented from scratch.

    Real run: 40 factors, lr=0.005, reg=0.02, 15 epochs, 356.9s.
    Validation RMSE fell monotonically 0.8893 -> 0.8146.
    """

    def __init__(self, n_users, n_movies, n_factors=40, n_epochs=15,
                 lr=0.005, reg=0.02, seed=RANDOM_SEED):
        self.nu, self.ni, self.k = n_users, n_movies, n_factors
        self.epochs, self.lr, self.reg, self.seed = n_epochs, lr, reg, seed

    def fit(self, u, i, r, val=None, verbose=True):
        rng = np.random.default_rng(self.seed)
        self.mu = r.mean()                                       # global mean
        self.bu = np.zeros(self.nu, np.float32)                  # user biases
        self.bi = np.zeros(self.ni, np.float32)                  # item biases
        self.P = rng.normal(0, 0.1, (self.nu, self.k)).astype(np.float32)  # user factors
        self.Q = rng.normal(0, 0.1, (self.ni, self.k)).astype(np.float32)  # item factors

        n = len(r)
        for ep in range(self.epochs):
            for x in rng.permutation(n):                         # shuffle each epoch
                a, b, rui = u[x], i[x], r[x]
                pred = self.mu + self.bu[a] + self.bi[b] + self.P[a] @ self.Q[b]
                e = rui - pred                                   # prediction error
                # bias updates (with L2 regularization)
                self.bu[a] += self.lr * (e - self.reg * self.bu[a])
                self.bi[b] += self.lr * (e - self.reg * self.bi[b])
                # latent-factor updates (use a copy of P[a] for the Q update)
                pa = self.P[a].copy()
                self.P[a] += self.lr * (e * self.Q[b] - self.reg * self.P[a])
                self.Q[b] += self.lr * (e * pa        - self.reg * self.Q[b])
            if val is not None and verbose:
                vp = self.predict(val[0], val[1])
                rmse = np.sqrt(mean_squared_error(val[2], vp))
                print(f"  epoch {ep+1:2d}/{self.epochs}  val RMSE = {rmse:.4f}")
        return self

    def predict(self, u, i):
        dot = np.sum(self.P[u] * self.Q[i], axis=1)
        return np.clip(self.mu + self.bu[u] + self.bi[i] + dot, 1, 5)

    def recommend(self, uidx, known, top_k=10):
        """Top-K unseen movies for one user, ranked by predicted score.

        `known` is the set of internal movie indices the user has already rated;
        these are masked out so we only recommend new items.
        """
        scores = self.mu + self.bu[uidx] + self.bi + self.Q @ self.P[uidx]
        scores[list(known)] = -np.inf
        top = np.argpartition(-scores, top_k)[:top_k]
        return top[np.argsort(-scores[top])]


# ----------------------------------------------------------------------------
# Model 3 - Item-based k-NN (and the explainability engine)
# ----------------------------------------------------------------------------
class ItemKNN:
    """Item-based collaborative filtering with cosine similarity.

    Computes item-item cosine similarity on mean-centered ratings (centering
    removes each movie's popularity offset so similarity reflects co-taste, not
    shared popularity). A user's rating for a movie is predicted as the
    similarity-weighted average of their ratings on that movie's k nearest
    neighbours.

    It is slightly less accurate than Matrix Factorization but interpretable:
    the neighbours driving a prediction are exactly the "because you watched X"
    reasons (see `explain`), which power the recommendation explanations.

    Real run: k=40, RMSE 0.8334 (20k-pair sample), MAP@10 0.8488.
    """

    def __init__(self, n_users, n_movies, k=40):
        self.nu, self.ni, self.k = n_users, n_movies, k

    def fit(self, u, i, r):
        # Mean-center each item's ratings, build sparse item x user matrix.
        self.item_mean = np.bincount(i, weights=r, minlength=self.ni) / \
                         np.maximum(np.bincount(i, minlength=self.ni), 1)
        centered = r - self.item_mean[i]
        IU = csr_matrix((centered, (i, u)), shape=(self.ni, self.nu), dtype=np.float32)

        # Full item-item cosine similarity (dense is fine on the subset).
        self.sim = cosine_similarity(IU, dense_output=True).astype(np.float32)
        np.fill_diagonal(self.sim, 0.0)                          # a movie isn't its own neighbour

        # Keep only top-k neighbours per item (zero the rest): denoise + speed.
        if self.k < self.ni:
            for it in range(self.ni):
                row = self.sim[it]
                row[np.argpartition(-row, self.k)[self.k:]] = 0.0

        # Cache each user's {movie: rating} for fast prediction/explanation.
        self.user_items = {}
        order = np.argsort(u, kind="stable")
        su, si, sr = u[order], i[order], r[order]
        start = 0
        for uidx in range(self.nu):
            end = start
            while end < len(su) and su[end] == uidx:
                end += 1
            if end > start:
                self.user_items[uidx] = dict(zip(si[start:end], sr[start:end]))
            start = end
        return self

    def predict_one(self, a, b):
        rated = self.user_items.get(a, {})
        if not rated:
            return self.item_mean[b]                             # cold user -> item mean
        nbr = self.sim[b]
        num = den = 0.0
        for j, ruj in rated.items():                            # similarity-weighted average
            s = nbr[j]
            if s > 0:
                num += s * (ruj - self.item_mean[j]); den += s
        return self.item_mean[b] if den == 0 else np.clip(self.item_mean[b] + num / den, 1, 5)

    def predict(self, u, i):
        return np.array([self.predict_one(a, b) for a, b in zip(u, i)])

    def explain(self, a, b, top_n=3):
        """Return the user's already-rated movies most responsible for
        recommending movie b: ranked by similarity x rating. These are the
        'because you watched X' reasons.

        Returns list of (movie_index, similarity, user_rating).
        """
        rated = self.user_items.get(a, {})
        contribs = [(j, self.sim[b, j], ruj) for j, ruj in rated.items() if self.sim[b, j] > 0]
        contribs.sort(key=lambda t: -t[1] * t[2])
        return contribs[:top_n]

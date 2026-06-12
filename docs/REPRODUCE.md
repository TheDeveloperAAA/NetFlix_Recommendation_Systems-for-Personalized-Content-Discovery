# Reproducing the Results

This document gives exact, step‑by‑step instructions to reproduce every number,
chart, and recommendation in this project. There are two supported paths: a
**Kaggle notebook** (recommended — no large download) and a **local script run**.

The target outputs you should be able to reproduce:

| Model | Test RMSE | MAP@10 |
|---|---|---|
| Weighted Mean | 0.9661 | 0.7784 |
| Matrix Factorization | 0.8146 | 0.8497 |
| Item‑based k‑NN | 0.8334 | 0.8488 |

Dataset processing: **24,053,764 ratings** parsed → **1,730,647** after subsetting
(**7,198 users × 419 movies**, 42.62% sparsity), split into **1,387,301 train /
343,346 test** by a temporal (newest‑20%‑per‑user) split.

> **Note on exact reproducibility.** Random seeds are fixed (`seed=42`) for the
> models, the MAP@10 user sample, and the k‑NN RMSE sample, so results are
> deterministic given the same data and the same subset thresholds. If you change
> `MIN_MOVIE_RATINGS` / `MIN_USER_RATINGS` or the number of rating files, the
> subset (and therefore the numbers) will differ — this is expected and is noted
> as future work in the report.

---

## Path A — Kaggle notebook (recommended)

**Why Kaggle:** the dataset is ~2 GB; on Kaggle it's mounted for you, so nothing
downloads to your machine, and you get a free 16 GB‑RAM kernel.

1. **Sign in** at kaggle.com (free account).
2. Create a **New Notebook**.
3. In the right sidebar, open the **Input** panel → **+ Add Input** → search
   **"Netflix Prize data"** (publisher **netflix‑inc**) → click **+** to attach it.
4. **Verify the data path.** In the left file browser, expand the attached
   dataset and note its folder path under `/kaggle/input/...`. It may be nested
   (e.g. `/kaggle/input/netflix-prize-data` *or* a deeper path). To be certain,
   run this in a cell and read the output:
   ```python
   import os
   for root, dirs, files in os.walk("/kaggle/input"):
       print(root)
   ```
5. **Open the notebook** `notebooks/netflix_recommender.ipynb` (upload it via
   *File → Import Notebook*, or paste its cells).
6. In the **config cell**, set `DATA_DIR` to the path you confirmed in step 4.
   The first cell prints:
   ```
   Data dir exists: True  |  synthetic fallback: False
   ```
   If it says `False`, the path is wrong — fix `DATA_DIR` and re‑run.
   (If the real data is genuinely absent, the notebook falls back to a small
   synthetic dataset so it still runs — but those numbers are **placeholders**,
   not the real results.)
7. **Run All.** Expected wall‑clock: parsing ~25 s, Matrix Factorization ~6 min
   (15 epochs), everything else seconds. Total well under 10 minutes.

Outputs appear inline (charts, tables, recommendations) and are written to
`/kaggle/working/outputs/` (figures + `dashboard_data.json`).

---

## Path B — Local script run

**Requirements:** Python 3.9+ and the Netflix Prize data downloaded locally.

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Get the data.** Download the Netflix Prize dataset (e.g. from Kaggle) and
   note the folder containing `combined_data_1.txt` and `movie_titles.csv`.
3. **Point the pipeline at it.** Edit `CONFIG.DATA_DIR` near the top of
   `run_pipeline.py`:
   ```python
   class CONFIG:
       DATA_DIR = Path("/path/to/netflix-prize-data")
       RATING_FILES = ["combined_data_1.txt"]
       ...
   ```
4. **Run:**
   ```bash
   python run_pipeline.py
   ```

This prints the same progress and results, and writes:
- `results/results_table.csv` — the final metrics
- `results/figures/*.png` — EDA + comparison charts
- `dashboard/dashboard_data.json` — data for the interactive dashboard

---

## Viewing the interactive dashboard

After a run has produced `dashboard/dashboard_data.json`:

```bash
cd dashboard
python -m http.server 8000
# open http://localhost:8000 in a browser
```

Serving the folder lets the page load the real results. Opened directly via
`file://` with no JSON present, it shows a built‑in sample.

---

## Tuning (optional)

All knobs live in the config block (`CONFIG` in `run_pipeline.py`, or `CFG` in the
notebook):

| Setting | Default | Effect |
|---|---|---|
| `RATING_FILES` | `["combined_data_1.txt"]` | add `_2/_3/_4` for more data (slower) |
| `MIN_MOVIE_RATINGS` | `3000` | higher → fewer movies, smaller/faster |
| `MIN_USER_RATINGS` | `200` | higher → fewer users, smaller/faster |
| `MF_FACTORS` | `40` | latent dimensions for Matrix Factorization |
| `MF_EPOCHS` | `15` | training epochs (largest speed lever) |
| `EVAL_USERS` | `2000` | sample size for MAP@10 |

If a run is too slow or memory‑tight, raise the subset thresholds or lower
`MF_EPOCHS`. If you want a richer catalogue, lower the thresholds (expect longer
training).

---

## Troubleshooting

- **`Data dir exists: False`** — the data path is wrong or the dataset isn't
  attached. Re‑check step 4 (Path A) or `CONFIG.DATA_DIR` (Path B).
- **Out‑of‑memory** — raise `MIN_MOVIE_RATINGS` / `MIN_USER_RATINGS`, or use a
  single rating file.
- **A library fails to install** (e.g. `surprise`) — it isn't required. The three
  models use only NumPy / SciPy / scikit‑learn.
- **Pip can't reach the internet on Kaggle** — Kaggle notebooks have internet
  off by default; the pipeline needs no internet, so this is harmless.

# Netflix Recommendation System for Personalised Content Discovery

> **Open Projects 2026 · AI/ML Track**
> A movie recommendation engine built on the **Netflix Prize dataset** (100M+ ratings) that learns user preferences, generates personalized Top‑K recommendations, and **explains every suggestion** — evaluated on both rating accuracy (RMSE) and ranking quality (MAP@10).

---

## Results at a glance

Trained on `combined_data_1.txt` (24,053,764 ratings → 1,730,647 after subsetting; 7,198 users × 419 movies), using a **leakage‑free temporal split**:

| Model | Test RMSE ↓ | MAP@10 ↑ |
|---|---|---|
| Weighted Mean (baseline) | 0.9661 | 0.7784 |
| **Matrix Factorization** | **0.8146** | **0.8497** |
| Item‑based k‑NN | 0.8334 | 0.8488 |

Matrix Factorization is best on both metrics and **beats Netflix's historical Cinematch benchmark (RMSE 0.9474)**. The headline finding: **rating accuracy and ranking quality are not the same goal** — the non‑personalized baseline stays in the RMSE ballpark yet collapses on MAP@10, because it ranks every user identically.

![Model comparison](results/figures/results_comparison.png)

---

## Repository structure

```
netflix-recommender/
├── README.md                      ← you are here
├── requirements.txt               ← dependencies
├── run_pipeline.py                ← end-to-end reproducible entry point
│
├── src/                           ← source modules (the 4 required components)
│   ├── data_processing.py         ← 1. DATA PROCESSING PIPELINE
│   ├── models.py                  ← 2. MODEL TRAINING PIPELINE
│   ├── evaluation.py              ← 3. EVALUATION SCRIPTS
│   └── recommendation.py          ← 4. RECOMMENDATION GENERATION MODULE
│
├── notebooks/
│   └── recommendation-system.ipynb   ← canonical executed run (real outputs embedded)
│
├── results/
│   ├── results_table.csv          ← final RMSE / MAP@10 numbers
│   └── figures/                   ← EDA + comparison charts (from the real run)
│
├── dashboard/
│   └── index.html                 ← interactive recommendation explorer (bonus)
│
└── docs/
    ├── REPRODUCE.md               ← step-by-step reproduction instructions
    └── METHODOLOGY.md             ← design decisions & model details
```

### How the code maps to the required components

| Required component | Where |
|---|---|
| **Data Processing Pipeline** | [`src/data_processing.py`](src/data_processing.py) — raw‑file parser, title loader, smart subset, index remap, temporal split |
| **Model Training Pipeline** | [`src/models.py`](src/models.py) — `WeightedMean`, `MatrixFactorization`, `ItemKNN` |
| **Evaluation Scripts** | [`src/evaluation.py`](src/evaluation.py) — RMSE, `average_precision_at_k`, `evaluate_map` |
| **Recommendation Generation Module** | [`src/recommendation.py`](src/recommendation.py) — explained Top‑K recommendations + dashboard export |
| **Documentation** | this README + [`docs/`](docs/) |
| **Instructions to reproduce** | [`docs/REPRODUCE.md`](docs/REPRODUCE.md) |

---

## Quick start

### Option A — Kaggle notebook (recommended; no data download)

1. Create a new **Kaggle Notebook**.
2. **Add Input** → search **"Netflix Prize data"** (by `netflix-inc`) → attach it.
3. Open `notebooks/recommendation-system.ipynb `, **verify the data path** in the config cell matches where Kaggle mounted the data (check the left file browser — the mount path can be nested), then **Run All**.

The notebook already contains the executed outputs from the real run, so you can also just read it top‑to‑bottom.

### Option B — Run the pipeline as a script

```bash
pip install -r requirements.txt
# Edit CONFIG.DATA_DIR in run_pipeline.py to point at your data folder, then:
python run_pipeline.py
```

This regenerates `results/results_table.csv`, the figures in `results/figures/`, and `dashboard/dashboard_data.json`.

> Full step‑by‑step instructions, including the exact data path fix, are in **[docs/REPRODUCE.md](docs/REPRODUCE.md)**.

---

## The approach (three paradigms)

1. **Weighted‑Mean ranking** — a non‑personalized baseline (IMDb damped‑mean formula). The bar to beat.
2. **Matrix Factorization** — latent‑factor model trained with regularized SGD, from scratch. Best accuracy.
3. **Item‑based k‑NN** — neighbourhood model; slightly less accurate but **interpretable**, and it powers the "because you watched X" explanations.

**Recommendations** use a hybrid: Matrix Factorization *ranks* the items (best accuracy), Item‑k‑NN *explains* each pick. Example real output:

```
User 0 — top‑rated: Elizabeth, X2, Jaws, Roger Rabbit, Bowling for Columbine
  • Seven Samurai        (pred 4.67)  ← because you liked The Godfather, This Is Spinal Tap
  • North by Northwest   (pred 4.39)  ← because you liked The Odd Couple, The Godfather
  • Evil Dead 2          (pred 4.50)  ← because you liked A Nightmare on Elm Street, 28 Days Later
```

See **[docs/METHODOLOGY.md](docs/METHODOLOGY.md)** for the full design rationale, model equations, and the RMSE‑vs‑MAP@10 analysis.

---

## Interactive dashboard (bonus)

`dashboard/index.html` is a standalone explorer for browsing any sample user's history and their explained recommendations. After running the pipeline (which writes `dashboard/dashboard_data.json`):

```bash
cd dashboard
python -m http.server 8000
# open http://localhost:8000
```

Opened directly with no JSON present, it falls back to a small built‑in sample so it always renders.

---

## Notes & constraints

- Uses **only the provided Netflix Prize dataset** — no external metadata — per the competition rules.
- The "smart subset" (active users / popular movies) is explicitly permitted and keeps the pipeline runnable on a free 16 GB kernel.
- All models are **dependency‑light** (NumPy / SciPy / scikit‑learn), so the notebook runs even where heavier libraries fail to install.
- Movie IDs are sequential 1–17,770 and do **not** map to IMDb/Netflix IDs.

## License

Provided for the Open Projects 2026 submission. Dataset © Netflix, used under the Netflix Prize terms.

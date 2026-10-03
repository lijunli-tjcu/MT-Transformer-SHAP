# MT-Transformer-SHAP

Companion code for the manuscript:

> **A Multi-Task Transformer Framework with SHAP Interpretability for Diagnosing Student Evaluation of Teaching**
> Lijun Li, Tianjin University of Commerce
> Submitted to *Education and Information Technologies* (Springer)

This repository contains the code, intermediate data, and documentation to reproduce the results in the manuscript.

---

## Overview

The framework jointly predicts an overall teaching-quality score and twelve sub-indicator scores from shared encoder representations, and uses SHAP (SHapley Additive exPlanations) for post-hoc interpretability at both global and sample levels.

The pipeline comprises:

- **Input**: 9 background features from student evaluation of teaching (SET) surveys
- **Model**: Multi-task Transformer (shared encoder + 13 task-specific heads)
- **Output**: 1 overall teaching-quality score + 12 sub-indicator diagnostics
- **Interpretability**: SHAP global importance + sample-level waterfall plots

---

## Repository Structure

```
MT-Transformer-SHAP/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
├── scripts/                           # All analysis scripts (24 files)
├── data/                              # (Original data NOT included — see data/README.md)
├── shap/                              # SHAP intermediate results (25 runs)
├── per_run/                           # Per-run predictions and R2 values
├── logs/                              # Training loss curves
├── issa_logs/                         # ISSA hyperparameter search results
├── figures/                           # Generated figures (empty; populated on run)
└── Supplementary/                     # Supplementary tables
    └── shap/
        └── Supplementary_Table_S3_SHAP_Stability.csv
```

---

## Data Availability

The **original survey data are not publicly distributed** in order to protect participant confidentiality. De-identified data are available from the corresponding author upon reasonable request and with institutional ethics approval.

**All intermediate data needed to reproduce the figures and tables in the paper are included** in this repository (see `data/README.md` for details).

---

## Requirements

- **Python 3.10** (tested with 3.10.11)
- **Core packages**: see `requirements.txt`
- **GPU**: optional; all results can be reproduced on a standard CPU
  (computation time: ~24 hours for the full pipeline)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Pipeline Overview

The code is organised into **11 stages**. Run them in the following order.

### Stage 1 — Data preparation

| # | Script | Purpose |
|---|---|---|
| 1 | `check_feature_encoding.py` | Verify input feature dimensionality |
| 2 | `compute_feature_correlations.py` | Pearson/Spearman correlations (Table S4) |
| 3 | `compute_correlations.py` | Overall-to-subindicator correlations (§5.2) |

### Stage 2 — ISSA hyperparameter search

| # | Script | Purpose |
|---|---|---|
| 4 | `run_issa_optimization.py` | ISSA search → `issa_logs/best_params_final.txt` |

### Stage 3 — Main cross-validation experiment

| # | Script | Purpose |
|---|---|---|
| 5 | `run_cross_validation.py` | 25-run repeated stratified 5-fold CV → `per_run/*.npy` |
| 6 | `kfold_cv.py` | Training loss curves → `logs/mt_transformer_loss.csv` |

### Stage 4 — Ablation study

| # | Script | Purpose |
|---|---|---|
| 7 | `run_ablation.py` | Ablation of multi-task and Transformer components (Table 5) |

### Stage 5 — Baseline training

| # | Script | Purpose |
|---|---|---|
| 8 | `train_baselines.py` | XGBoost / LightGBM / MT-MLP baselines |

### Stage 6 — Table 3 generation

| # | Script | Purpose |
|---|---|---|
| 9 | `make_table3_model_comparison.py` | Model comparison table |

### Stage 7 — Table 4 generation

| # | Script | Purpose |
|---|---|---|
| 10 | `make_table4_per_indicator.py` | Per-indicator performance table |

### Stage 8 — Statistical tests

| # | Script | Purpose |
|---|---|---|
| 11 | `run_paired_ttest.py` | Standard paired t-test |
| 12 | `corrected_resampled_ttest.py` | Nadeau-Bengio corrected resampled t-test |
| 13 | `analyze_r2_ceiling.py` | Empirical R² ceiling analysis |

### Stage 9 — SHAP analysis

| # | Script | Purpose |
|---|---|---|
| 14 | `run_shap_all.py` | Calls `run_shap_kernel.py` 25 times → `shap/shap_run*.npy` |
| 15 | `run_shap_kernel.py` | Compute SHAP values for a single run |
| 16 | `prepare_shap_data.py` | Compute `shap_mean.npy` and `X_explain.npy` |
| 17 | `recompute_shap_mean.py` | Recompute mean \|SHAP\| (Method A and Method B) |
| 18 | `check_shap_scale.py` | Check SHAP value ranges and additivity |
| 19 | `analyze_shap_stability.py` | Stability analysis → `Supplementary/` |

### Stage 10 — Figure generation

| # | Script | Purpose |
|---|---|---|
| 20 | `generate_fig1_framework.py` | Fig. 1: Overall framework |
| 21 | `generate_fig2_loss.py` | Fig. 2: Training/validation loss |
| 22 | `generate_fig3_shap_bar.py` | Fig. 3: SHAP feature importance |
| 23 | `generate_fig4_waterfall.py` | Fig. 4: SHAP waterfall plots |

### Stage 11 — ISSA evaluation

| # | Script | Purpose |
|---|---|---|
| 24 | `evaluate_issa.py` | Evaluate ISSA-tuned model (Appendix A) |

---

## Reproducing Specific Results

| Result | Scripts to run |
|---|---|
| **Table 3** (model comparison) | Stage 2 → Stage 5 → Stage 6 |
| **Table 4** (per-indicator) | Stage 3 → Stage 7 |
| **Table 5** (paired comparisons) | Stage 3 → Stage 8 |
| **Fig. 1** (framework) | `generate_fig1_framework.py` |
| **Fig. 2** (loss curves) | Stage 3 (`kfold_cv`) → `generate_fig2_loss.py` |
| **Fig. 3** (SHAP importance) | Stage 9 (`run_shap_all`) → `generate_fig3_shap_bar.py` |
| **Fig. 4** (waterfall) | Stage 9 (`prepare_shap_data`) → `generate_fig4_waterfall.py` |
| **Supplementary Table S3** (SHAP stability) | Stage 9 (`analyze_shap_stability`) |

**Note:** The full pipeline requires the original survey data, which is not included. However, all intermediate results are provided, so **Fig. 1–4, Table 3–5, and Table S3 can be regenerated directly from the included data** without re-running the training.

---

## Reproducing Figures from Included Data

To reproduce all figures without re-running training:

```bash
cd scripts
python generate_fig1_framework.py
python generate_fig2_loss.py
python generate_fig3_shap_bar.py
python generate_fig4_waterfall.py
```

All figures will be saved to `../figures/`.

---

## Citation

If you use this code, please cite:

```bibtex
@article{li2026mt,
  author  = {Li, Lijun},
  title   = {A Multi-Task Transformer Framework with {SHAP} Interpretability for Diagnosing Student Evaluation of Teaching},
  journal = {Education and Information Technologies},
  year    = {2026},
  note    = {Under review}
}
```

---

## License

This project is licensed under the **MIT License** — see `LICENSE` for details.

---

## Contact

**Lijun Li**
School of Foreign Studies, Tianjin University of Commerce
Tianjin 300134, China
Email: frausunny@126.com
ORCID: [0009-0002-8738-7553](https://orcid.org/0009-0002-8738-7553)
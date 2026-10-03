# Data Directory

The original survey data are **not publicly available** in order to protect
participant confidentiality. De-identified data are available from the
corresponding author upon reasonable request and with institutional ethics
approval.

## Why the raw data are not distributed

The dataset consists of student evaluation of teaching (SET) responses
collected from Engineering English courses. Although the data are
anonymised, some combinations of background variables (e.g., class size,
year of study, instructor characteristics) could potentially be used to
re-identify individual respondents. Sharing the raw data publicly would
therefore violate the confidentiality commitments made to participants.

## How to reproduce the results in the paper

All intermediate data needed to reproduce the figures and tables in the
manuscript are included in this repository:

| Directory | Contents | Reproduces |
|---|---|---|
| `shap/` | SHAP values for 25 runs (`shap_run0.npy` – `shap_run24.npy`), `shap_mean.npy`, `X_explain.npy` | Fig. 3, Fig. 4, §3.7, §5.5 |
| `per_run/` | Per-run predictions and R² values | Table 3, Table 4, Table 5 |
| `logs/` | Training and validation loss curves | Fig. 2 |
| `issa_logs/` | ISSA hyperparameter search results | Appendix A |

The scripts that generate these intermediate files
(e.g., `run_cross_validation.py`, `run_shap_all.py`) require the original
survey data, which is **not** included. To re-run those scripts, please
request the de-identified dataset from the corresponding author.

## Requesting the data

Please contact:

**Lijun Li**
School of Foreign Studies
Tianjin University of Commerce
Tianjin 300134, China
Email: frausunny@126.com
ORCID: 0009-0002-8738-7553
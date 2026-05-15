# Leak-Free Tabular Foundation Models for ASD Eye-Tracking Classification
### TabICLv2 · TabPFN · XGBoost · LOOCV · Severity Scoring · Explainability

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aashnav01/asd-eyetracking-tabicl-loocv/blob/main/ASD_TabICLv2_TabPFN_INTEGRATED_v3.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

---

## What this notebook does

A complete, reproducible ML pipeline comparing three classifiers — **TabICLv2**, **TabPFN**, and **XGBoost** — on a 57-participant ASD/TD eye-tracking cohort, under a rigorously leak-free Leave-One-Out Cross-Validation (LOOCV) framework.

---

## Pipeline Steps

| Step | Description |
|------|-------------|
| 0 | Global FDR feature selection (reporting only) |
| 1 | Leave-One-Participant-Out CV — all three models on identical folds |
| 2 | Pairwise McNemar tests |
| 3 | Severity scoring — three-tier P(ASD) bands |
| 4 | Global feature importance (permutation + SHAP) |
| 5 | Per-participant local explanations |
| 6 | Permutation test (label shuffle, N=1000) |
| 7 | ROC curves and calibration plots |
| 8 | One-time gated holdout evaluation |

---

## Requirements

```bash
pip install tabicl[finetune] tabpfn skrub scikit-learn statsmodels shap xgboost matplotlib pandas numpy tqdm joblib
```

**Tokens needed** (set as Colab secrets, never hard-code):
- `HF_TOKEN` — from https://huggingface.co/settings/tokens
- `TABPFN_TOKEN` — from https://ux.priorlabs.ai/account

---

## Usage

1. Upload your `processed_data_fixed.zip` (organised participant CSVs) to Colab
2. Set `HF_TOKEN` and `TABPFN_TOKEN` as Colab secrets (🔑 icon)
3. Run cells sequentially

> The holdout cell (Step 8) is gated and runs exactly once — do not delete the flag file.

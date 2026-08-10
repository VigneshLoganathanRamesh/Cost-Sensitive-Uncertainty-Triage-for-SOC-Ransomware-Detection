# Cost-Sensitive Uncertainty Triage (CS-UT) for Ransomware Detection

Ransomware detection under a constrained analyst budget — an empirical evaluation on behavioural, memory-forensic and static PE datasets.

**MSc Cybersecurity project · National College of Ireland**

---

## 1. Overview

Security Operations Centres (SOCs) receive far more alerts than analysts can review. Most machine-learning ransomware detectors optimise accuracy or F1, which weight a missed infection and a false alarm equally — even though the two differ in cost by orders of magnitude, and even though no SOC can act on every alert.

This project implements and evaluates a **Cost-Sensitive Uncertainty Triage (CS-UT)** framework that reformulates detection as a *budget-constrained decision problem*. Instead of forcing a verdict on every sample, the system:

1. trains cost-sensitive detectors (Focal Loss, XGBoost `scale_pos_weight`),
2. quantifies predictive uncertainty with **Monte Carlo Dropout** (T = 50 stochastic passes),
3. routes the top-*k*% most uncertain samples to a simulated analyst, and auto-decides the rest,
4. explains routed decisions with **SHAP**, and
5. audits probability **calibration** (ECE) across three data-augmentation conditions.

The complete stack is run on three datasets spanning three different feature modalities.

### Research questions

| | Question | Answered? |
|---|---|---|
| **RQ1** | Can uncertainty-aware triage materially reduce the ransomware miss rate while routing ≤ 5–10% of alerts to analysts? | Yes |
| **RQ2** | How does the FN:FP cost ratio (`scale_pos_weight`, Focal-Loss α) affect detection performance? | Yes |
| **RQ3** | Which uncertainty metric — predictive entropy, MC Dropout variance, or margin — best selects analyst-worthy samples? | Yes |
| **RQ4** | (a) Do SHAP explanations reduce analyst decision time? (b) Are SHAP attributions stable across MC Dropout passes? | (a) No — requires human-subject data. (b) Yes |
| **RQ5** | Do SMOTE/CTGAN cause overconfidence, and does temperature scaling correct it? | Yes — hypothesis **not** supported |

---

## 2. Repository structure

```
cs-ut-ransomware-detection/
├── README.md              # this file
├── requirements.txt       # Python dependencies
├── .gitignore             # excludes data files, checkpoints, caches
├── code/
│   ├── The_Code.ipynb     # complete implementation (24 cells, top-to-bottom)
│   └── README.md          # cell-by-cell map of the notebook
└── dataset/
    └── README.md          # dataset sources, expected filenames, download steps
```

> **Note on `dataset/`:** the raw data files are **not committed** to this repository. The three corpora are large and are redistributed under their own terms by their original publishers. `dataset/README.md` gives the exact filenames the notebook expects and where to obtain each one.

---

## 3. Datasets

| Dataset | Role | Modality | Records used | Features used | Source |
|---|---|---|---|---|---|
| **MLRan** (2025) | Primary — train & evaluate | Dynamic behavioural | 4,880 (3,905 train / 975 test) | 483 supplied → 50 after MI + RFE | [arXiv:2505.18613](https://arxiv.org/abs/2505.18613) |
| **Obfuscated-MalMem2022** | Secondary validation | Memory forensics | 58,058 loaded → 58,023 after cleaning | 57 → 50 via mutual information | [CIC, UNB](https://www.unb.ca/cic/datasets/malmem-2022.html) |
| **EMBER-2018** | Tertiary validation | Static PE | 50,000 (a disclosed **5% sample** of 1,000,000) | 2,381 → 50 via mutual information | [github.com/elastic/ember](https://github.com/elastic/ember) |

All three are public research datasets containing no direct personal identifiers. Confirm each publisher's licence and citation requirement before redistributing. See `dataset/README.md` for details.

---

## 4. Setup

### Requirements

- Python 3.10+
- A CUDA-capable GPU is strongly recommended (the notebook runs a 108-configuration CNN grid search and 50-pass MC Dropout inference on three datasets). It will run on CPU, but slowly.
- The notebook was developed and executed in **Google Colab** with a GPU runtime.

### Option A — Google Colab (as originally executed)

1. Upload `code/The_Code.ipynb` to Colab and select a GPU runtime (*Runtime → Change runtime type → GPU*).
2. Place the five data files (see `dataset/README.md`) in a Google Drive folder.
3. Run cell 1 — it mounts Drive and resolves the data directory.
4. If your folder differs from the default, edit `DATA_DIR_CANDIDATES` in cell 1.

### Option B — Local Jupyter

```bash
git clone https://github.com/<your-username>/cs-ut-ransomware-detection.git
cd cs-ut-ransomware-detection

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# place the five data files in dataset/ (see dataset/README.md)
jupyter notebook code/The_Code.ipynb
```

**One required edit for local runs.** Cell 1 defines the search path for data files:

```python
DATA_DIR_CANDIDATES = [
    '/content/drive/MyDrive/Colab Notebooks/Project_Code_&_Dataset',
    '/mnt/user-data/uploads',
    '.',
    '/content',
]
```

Add the `dataset/` folder so the resolver finds the files when the notebook is run from `code/`:

```python
DATA_DIR_CANDIDATES = [
    '../dataset',                 # <-- add this line
    'dataset',
    '/content/drive/MyDrive/Colab Notebooks/Project_Code_&_Dataset',
    '/mnt/user-data/uploads',
    '.',
    '/content',
]
```

The Google Drive mount in cell 1 is Colab-specific; it fails harmlessly outside Colab, and the resolver falls through to the remaining candidates.

---

## 5. How to run

Run all 24 cells **in order** — the notebook is strictly sequential and later cells depend on variables defined earlier. There are no command-line entry points.

| Cells | Stage | Approximate cost |
|---|---|---|
| 0–4 | Setup, data loading, temporal check, EDA | fast |
| 5–6 | Feature selection (MI → RFE), splits, scaling, SMOTE, CTGAN | moderate (CTGAN is the slow part) |
| 7–8 | Random Forest (GridSearchCV) and XGBoost (cost sweep + 100-trial Optuna) | moderate |
| 9–11 | CNN grid search (108 configs), final training, MC Dropout, Focal-Loss α sweep | **slowest — GPU strongly advised** |
| 12 | Budget-constrained triage and the 1–50% cost sweep | fast |
| 13–14 | SHAP explanations and literal top-5 views | moderate |
| 15–17 | Raw-condition models, calibration audit, temperature scaling, McNemar, final table | moderate |
| 18–20 | Secondary validation: MalMem2022, leakage investigation, full CS-UT stack | moderate |
| 21–22 | Tertiary validation: EMBER-2018 and full CS-UT stack | moderate |
| 23 | Cross-dataset generalisation summary | fast |

**Reproducibility.** `SEED = 69` is set for NumPy, PyTorch, scikit-learn, XGBoost, RFE and SMOTE. The Optuna studies and the CTGAN synthesiser are **not** seeded, so exact figures will shift slightly between runs. To make the run fully deterministic, pass a sampler seed to `optuna.create_study(...)` and a seed to `CTGANSynthesizer(...)`.

---

## 6. Main results

### Final model comparison — MLRan sealed test set (n = 975)

| Model | F1 | Recall | Precision | FNR | MCC | ECE | Expected cost |
|---|---|---|---|---|---|---|---|
| Random Forest (GridSearchCV, SMOTE) | 0.9518 | 0.9548 | 0.9487 | 0.0452 | 0.9075 | 0.0202 | $106,681,200 |
| XGBoost (Optuna, cost-sensitive) | 0.9494 | 0.9484 | 0.9504 | 0.0516 | 0.9034 | 0.0275 | $121,921,150 |
| CNN + Focal Loss + MC Dropout | 0.9597 | 0.9720 | 0.9476 | 0.0280 | 0.9223 | 0.0200 | $66,041,250 |
| **CNN + Budget-Constrained Rejection (CS-UT, k = 10%)** | **0.9837** | **0.9883** | **0.9791** | **0.0117** | **0.9681** | **0.0072** | **$25,402,875** |

The CS-UT row is computed on the **auto-decided 90%** only, assuming the analyst resolves routed cases correctly. Expected cost applies a per-incident breach cost ($5.08M, IBM 2025) to each missed test sample — it is a **relative ranking device, not a financial forecast**.

### Triage effectiveness (RQ1)

| Analyst budget | Reviewed | Missed ransomware | Reduction | Expected cost |
|---|---|---|---|---|
| 0% (no triage) | 0 | 13 | — | $66,041,250 |
| 7% | 68 | 6 | 53.8% | $30,482,550 |
| 10% | 97 | 5 | 61.5% | $25,402,875 |
| **15%** | 146 | **1** | **92.3%** | **$5,084,000** ← cost minimum |
| 50% | 487 | 1 | 92.3% | $5,092,375 |

Beyond 15% the miss count stops falling while review cost keeps accruing, so expected cost **rises** — the optimal review capacity is finite and identifiable.

### Uncertainty metric comparison (RQ3)

| Budget | Entropy | MC Variance | Margin |
|---|---|---|---|
| 5% | 0.0176 | **0.0135** | 0.0176 |
| 8% | 0.0137 | **0.0117** | 0.0137 |
| 10% | 0.0117 | **0.0048** | 0.0117 |

MC Dropout variance wins at every budget. Entropy and margin are identical because, for a binary classifier, both are monotone functions of |p̄ − 0.5| and therefore induce the same ranking.

### Calibration audit (RQ5) — ECE by augmentation condition

| Model | Raw | SMOTE | CTGAN |
|---|---|---|---|
| Random Forest | 0.0204 | **0.0202** | 0.0208 |
| XGBoost | 0.0326 | **0.0275** | 0.0312 |
| CNN + MC Dropout | 0.0276 | **0.0200** | 0.0266 |

No model in any condition exceeded 0.033, far below the 0.10 overconfidence threshold. **The oversampling-overconfidence hypothesis was not supported** — augmentation modestly *improved* calibration. Temperature scaling helped XGBoost (0.0275 → 0.0229) but *worsened* Random Forest and the CNN, the expected failure mode when correcting an already-calibrated model.

### Cross-dataset generalisation

| Dataset | Modality | CNN F1 | Miss rate @ k = 10% | Auto-recall @ k = 10% |
|---|---|---|---|---|
| MLRan | Dynamic behaviour | 0.9597 | 0.0117 | 0.9883 |
| MalMem2022 (group-aware split) | Memory forensics | 0.9972 | 0.0000 | 1.0000 |
| EMBER-2018 (5% sample) | Static PE | 0.8666 | 0.0281 | 0.9719 |

Triage reduced the miss rate on every dataset, but its magnitude tracks base-model quality: it reached zero where the detector was near-perfect and only halved the FNR where the detector was weak. **Triage amplifies a competent detector; it does not rescue an inadequate one.**

---

## 7. Honest findings and limitations

Results that ran *against* the project's own hypotheses are reported rather than omitted:

- **Random Forest out-triaged the MC-Dropout CNN.** At k = 10%, RF reached a miss rate of 0.0024 against the CNN's 0.0117 — using nothing more than a probability-margin proxy. Bayesian-approximate uncertainty was not required for effective triage.
- **No significant difference between base models.** McNemar's test found p = 0.8383 (RF vs XGBoost), p = 0.1456 (RF vs CNN) and p = 0.0665 (XGBoost vs CNN). The measured benefit comes from the **triage architecture**, not the choice of classifier.
- **Augmentation improved calibration** rather than degrading it, contradicting RQ5's premise.
- **MalMem2022's near-perfect scores are real.** A group-aware re-split with zero family overlap moved F1 by only 0.0002, and nearest-neighbour distances contradicted the near-duplicate hypothesis. The scores reflect intrinsic separability of memory-forensic features.

Known limitations:

- **No temporal validation** — the supplied MLRan feature files contain no date column, so a chronological split could not be constructed.
- **Test-set involvement in operating-point selection** — the budget sweep and XGBoost cost-ratio sweep were evaluated on test data, making those figures optimistic.
- **EMBER uses a disclosed 5% sample** with a reduced training budget; its results demonstrate transferability, not tuned performance.
- **CTGAN was trained for 10 epochs** against a library default of 300, so the CTGAN arm of RQ5 is under-powered.
- No adversarial, drift, real-time or deployment evaluation; MC Dropout's 50× inference cost was never measured.

---

## 8. Known issues in the current notebook

Four printed messages do not match the executed code. They are cosmetic but misleading, and are listed here for transparency:

| Cell | Message says | Code actually does |
|---|---|---|
| 20, 22 | "lightweight 20-trial Optuna search" | `n_trials=5` |
| 20 | "reduced 40-epoch training budget" | `epochs=5` |
| 22 | "reduced 40-epoch training budget" | `epochs=15` |
| 22 | EMBER-2018 "using the group-aware split" | EMBER has no group-aware split — that procedure applies only to MalMem2022 |

The notebook also prints no library version numbers. Adding a version-printing cell is recommended for reproducibility.

---

## 9. Citation

If you use this work, please cite the underlying datasets:

```bibtex
@article{onwuegbuche2025mlran,
  title  = {MLRan: A Behavioural Dataset for Ransomware Analysis and Detection},
  author = {Onwuegbuche, Faithful Chiagoziem and Olaoluwa, Adelodun and
            Jurcut, Anca Delia and Pasquale, Liliana},
  journal = {arXiv preprint arXiv:2505.18613},
  year   = {2025}
}

@inproceedings{carrier2022malmem,
  title     = {Detecting Obfuscated Malware using Memory Feature Engineering},
  author    = {Carrier, Tristan and Victor, Princy and
               Tekeoglu, Ali and Lashkari, Arash Habibi},
  booktitle = {Proc. 8th Int. Conf. Information Systems Security and Privacy (ICISSP)},
  pages     = {177--188},
  year      = {2022},
  doi       = {10.5220/0010908200003120}
}

@article{anderson2018ember,
  title   = {EMBER: An Open Dataset for Training Static PE Malware Machine Learning Models},
  author  = {Anderson, Hyrum S. and Roth, Phil},
  journal = {arXiv preprint arXiv:1804.04637},
  year    = {2018}
}
```

## 10. Licence and use

This repository contains academic coursework. The code is provided for research and educational purposes. The datasets are **not** included and remain subject to the licences of their original publishers — check each before use or redistribution.

This work is defensive security research. The models and explanations here are intended for detection and triage, not for developing evasion techniques.

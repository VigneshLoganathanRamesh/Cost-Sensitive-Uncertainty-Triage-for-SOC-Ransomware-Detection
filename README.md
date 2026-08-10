# A cost-sensitive uncertainty triage for SOC workload management for ransomware detection.

A ransomware detector that knows when it doesn't know — and hands those cases to a human.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Notebook](https://img.shields.io/badge/format-Jupyter-orange)

---

## What this is

Most ML malware detectors output a verdict for every sample and are scored on accuracy or F1. That doesn't match how a real Security Operations Centre works:

- **Analysts can't review everything.** A typical SOC sees ~3,000 alerts a day and leaves most untouched.
- **The two error types cost wildly different amounts.** A missed ransomware infection runs into millions. A false alarm costs an analyst a few minutes. F1 treats them as equal.

**CS-UT** reframes detection as a budget problem: given that your team can only review *k*% of alerts, which ones should they be?

The system measures how *uncertain* the model is about each prediction, sends the top *k*% most uncertain cases to a human, and auto-decides the rest. The result is fewer missed infections at the same staffing level.

**Headline result:** on the MLRan test set, routing 10% of alerts to a human cut missed ransomware from **13 → 5** (−61.5%). Routing 15% cut it to **1**.

---

## Models and techniques

**Models**
- Random Forest — baseline, tuned with `GridSearchCV` (5-fold, 12 combos)
- XGBoost — cost-sensitive, tuned with Optuna (100 TPE trials)
- 1D CNN — Focal Loss + MC Dropout, tuned over a 108-config grid
- CNN + budget-constrained rejection — the full CS-UT system

**Techniques**
- Feature selection: mutual information (483 → 100) then RFE (100 → 50)
- Class balancing: SMOTE and CTGAN, fitted on training data only
- Uncertainty: predictive entropy, MC Dropout variance, margin
- Calibration: Expected Calibration Error + temperature scaling
- Explainability: SHAP (TreeExplainer for RF, deep explanations for the CNN)
- Significance: McNemar's test on paired predictions
- Leakage check: group-aware re-split by malware family + nearest-neighbour analysis

---

## Datasets

| Dataset | Role | Type | Size used |
|---|---|---|---|
| [MLRan](https://arxiv.org/abs/2505.18613) | Primary | Dynamic behaviour (API calls, registry, file ops) | 4,880 samples, 483 features |
| [Obfuscated-MalMem2022](https://www.unb.ca/cic/datasets/malmem-2022.html) | Validation | Memory forensics | 58,058 rows, 57 features |
| [EMBER-2018](https://github.com/elastic/ember) | Validation | Static PE headers | 50,000 rows (5% sample), 2,381 features |

Three different feature modalities on purpose — if the triage layer only works on one kind of data, it isn't much of a framework.

---

## Repository structure

```
.
├── code/
│   └── The_Code.ipynb
├── dataset/
│   ├── MLRan_labels.csv
│   ├── MLRan_X_test_RFE.csv
│   ├── MLRan_X_train_RFE.csv
│   ├── Obfuscated-MalMem2022.parquet
│   └── README.md
├── requirements.txt
└── README.md
```

---

## Setup

### Colab (recommended — this is how it was built)

1. Upload `code/The_Code.ipynb`, set **Runtime → Change runtime type → GPU**.
2. Put the five data files in a Drive folder.
3. Run cell 1 to mount Drive. If your folder path differs, edit `DATA_DIR_CANDIDATES`.
4. Run all cells.

### Local

```bash
git clone https://github.com/VigneshLoganathanRamesh/Cost-Sensitive-Uncertainty-Triage-for-SOC-Ransomware-Detection.git
cd Cost-Sensitive-Uncertainty-Triage-for-SOC-Ransomware-Detection

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# drop the five data files into dataset/ — see dataset/README.md
jupyter notebook code/The_Code.ipynb
```

**One edit needed for local runs.** In cell 1, add `dataset/` to the search path:

```python
DATA_DIR_CANDIDATES = [
    '../dataset',        # ← add this
    'dataset',           # ← and this
    '/content/drive/MyDrive/Colab Notebooks/Project_Code_&_Dataset',
    '/mnt/user-data/uploads',
    '.',
    '/content',
]
```

The Drive mount fails harmlessly outside Colab and falls through to the next candidate.

---

## Running it

Run all 24 cells **in order**. The notebook is sequential — cell 12 needs variables from cell 10, and so on. No CLI, no entry point scripts.

| Cells | Stage | Speed |
|---|---|---|
| 0–4 | Setup, load MLRan, EDA | fast |
| 5–6 | Feature selection, splits, SMOTE, CTGAN | medium |
| 7–8 | Random Forest + XGBoost (Optuna: 100 trials) | medium |
| 9–11 | **CNN grid search (108 configs) + MC Dropout** | slowest — use a GPU |
| 12 | Triage + budget sweep | fast |
| 13–14 | SHAP | medium |
| 15–17 | Calibration, temperature scaling, McNemar, final table | medium |
| 18–23 | MalMem2022 + EMBER-2018 validation runs | medium |

**Reproducibility:** `SEED = 69` covers NumPy, PyTorch, scikit-learn, XGBoost and SMOTE. Optuna and CTGAN are *not* seeded, so numbers shift slightly between runs. To lock them down, pass `sampler=optuna.samplers.TPESampler(seed=SEED)` to `create_study`.

---

## Results

### Final comparison — MLRan test set (975 samples)

| Model | F1 | Recall | FNR | MCC | ECE | Expected cost |
|---|---|---|---|---|---|---|
| Random Forest | 0.9518 | 0.9548 | 0.0452 | 0.9075 | 0.0202 | $106.7M |
| XGBoost | 0.9494 | 0.9484 | 0.0516 | 0.9034 | 0.0275 | $121.9M |
| CNN + Focal + MC Dropout | 0.9597 | 0.9720 | 0.0280 | 0.9223 | 0.0200 | $66.0M |
| **CS-UT (CNN + triage @ 10%)** | **0.9837** | **0.9883** | **0.0117** | **0.9681** | **0.0072** | **$25.4M** |

### Triage: what each budget buys you

| Analyst budget | Alerts reviewed | Ransomware missed | Expected cost |
|---|---|---|---|
| 0% (no triage) | 0 | 13 | $66.0M |
| 7% | 68 | 6 | $30.5M |
| 10% | 97 | 5 | $25.4M |
| **15%** | 146 | **1** | **$5.1M** ← cheapest |
| 50% | 487 | 1 | $5.1M |

### Which uncertainty signal to use

| Budget | Entropy | **MC Variance** | Margin |
|---|---|---|---|
| 5% | 0.0176 | **0.0135** | 0.0176 |
| 8% | 0.0137 | **0.0117** | 0.0137 |
| 10% | 0.0117 | **0.0048** | 0.0117 |

### Does it transfer to other data?

| Dataset | CNN F1 | Miss rate @ 10% |
|---|---|---|
| MLRan (behaviour) | 0.9597 | 0.0117 |
| MalMem2022 (memory) | 0.9972 | 0.0000 |
| EMBER-2018 (static PE) | 0.8666 | 0.0281 |

---

## How to read these results

**"Expected cost" is a ranking tool, not a dollar forecast.** It's `missed × $5.08M + false_alarms × $50 + reviewed × $25`, using IBM's average ransomware incident cost. Applying a per-*incident* figure to each missed *sample* in a 975-row test set isn't literally meaningful — but comparing models with it is, because it's the only metric here that reflects the real cost asymmetry.

**The CS-UT row isn't apples-to-apples.** Its metrics cover only the 90% of samples the model auto-decided, and assume the analyst gets the routed 10% right. That's the point of the design, but don't read 0.9837 as "better classifier."

**More review capacity isn't always better.** Past 15% the miss count stops falling while review costs keep adding up, so expected cost starts climbing again. There's an actual optimum, and it's finite.

**Triage amplifies a good detector; it can't rescue a bad one.** On MalMem2022 (strong base model) triage drove misses to zero. On EMBER-2018 (weak base model) it only halved the FNR.

**The models are statistically tied.** McNemar's test found no significant difference between RF, XGBoost and the CNN (p = 0.84, 0.15, 0.07). The gain comes from the *triage architecture*, not from picking the right classifier.

---

## Things that didn't go as expected

Worth knowing before you build on this:

- **Random Forest beat the CNN at triage.** At a 10% budget, RF hit a 0.0024 miss rate vs the CNN's 0.0117 — using just a simple probability-margin proxy. You may not need MC Dropout at all.
- **SMOTE and CTGAN *improved* calibration** rather than causing overconfidence. Every ECE came in under 0.033, well below the 0.10 "overconfident" line.
- **Temperature scaling made things worse** for two of three models. It only helps when a model is actually miscalibrated; applied to a well-calibrated one it does damage.
- **MalMem2022's near-perfect scores are genuine.** A group-aware re-split with zero family overlap moved F1 by 0.0002. Those features really are that separable — it's not leakage.

---

## Known limitations

- **No temporal validation** — the MLRan feature files ship without a date column, so a chronological split isn't possible.
- **The triage operating point was picked on test data**, which makes those numbers optimistic. Same for the XGBoost cost-ratio sweep.
- **EMBER results use a 5% sample** and a reduced training budget — a transferability demo, not a tuned benchmark.
- **CTGAN ran for 10 epochs** (library default is 300), so that arm is under-powered.
- No adversarial, drift, or real-time evaluation. MC Dropout's 50× inference cost was never measured.

---

## Credits

Built on three public datasets — please cite them if you use this:

- **MLRan** — Onwuegbuche et al., [arXiv:2505.18613](https://arxiv.org/abs/2505.18613) (2025)
- **CIC-MalMem-2022** — Carrier et al., ICISSP 2022, [doi:10.5220/0010908200003120](https://doi.org/10.5220/0010908200003120)
- **EMBER-2018** — Anderson & Roth, [arXiv:1804.04637](https://arxiv.org/abs/1804.04637) (2018)


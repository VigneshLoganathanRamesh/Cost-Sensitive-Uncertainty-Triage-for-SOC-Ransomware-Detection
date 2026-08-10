# Code

`The_Code.ipynb` is the complete implementation — 24 cells, designed to be run top to bottom in a single session. There are no scripts or CLI entry points; later cells depend on variables created earlier, so partial execution will fail.

## Cell map

| Cell | What it does | Key outputs |
|---|---|---|
| 0 | Installs and imports dependencies; sets `SEED = 69`; selects CUDA device | Device confirmation |
| 1 | Mounts Google Drive (Colab) and defines `resolve_data_path()` over `DATA_DIR_CANDIDATES` | Resolved data directory |
| 2 | Loads MLRan train/test CSVs, excludes 4 metadata columns, cross-checks labels | Shapes, class distribution, 0 label mismatches |
| 3 | Searches for a date column to build a chronological split | Disclosure that no date column exists |
| 4 | Exploratory data analysis | Class-balance plot, missing-value count |
| 5 | Mutual information (top 100) → RFE (top 50); 70/10/20 split; MinMaxScaler; SMOTE | 3,416 / 489 / 975 splits; MI–RFE agreement 56/100 |
| 6 | CTGAN augmentation on ransomware training rows only (10 epochs) | Third training condition (`datasets['ctgan']`) |
| 7 | Random Forest: raw baseline, GridSearchCV (5-fold, 12 combos), CTGAN variant; defines `evaluate_model()` | Best params, confusion matrices |
| 8 | XGBoost: cost-ratio sweep {1,10,20,30,50} + 100-trial Optuna (TPE) | Selected `scale_pos_weight = 11` |
| 9 | CNN architecture, Focal Loss, 108-config grid search, final training | Best config; training curves |
| 10 | Monte Carlo Dropout (T = 50) and the three uncertainty metrics | CNN metrics; uncertainty histograms |
| 11 | Focal-Loss α sensitivity sweep | Validation FNR/F1 vs α |
| 12 | Budget-constrained selective classification; 1–50% extended cost sweep | Miss-rate table; cost curve; best operating point |
| 13 | SHAP for Random Forest and CNN; attribution stability across dropout passes | Global importance; consistency = 0.226 |
| 14 | Literal "top-5" SHAP compliance views | Two plots satisfying the proposal requirement |
| 15 | Trains all three models on the Raw condition | Raw-condition ECE values |
| 16 | Calibration audit, temperature scaling, PR curves, McNemar, final comparison table | Three-way ECE table; final results table |
| 17 | McNemar: full CS-UT vs the base CNN | Significant (p < 0.001) |
| 18 | Secondary validation on Obfuscated-MalMem2022 | Near-perfect RF/XGBoost scores |
| 19 | Leakage investigation: family overlap, nearest-neighbour check, group-aware re-split | Verdict table (naive vs group-aware) |
| 20 | Full CS-UT stack on MalMem2022 (group-aware) | CNN + MC Dropout + SHAP + triage |
| 21 | Tertiary validation on EMBER-2018 (5% sample) | RF and cost-sensitive XGBoost results |
| 22 | Full CS-UT stack on EMBER-2018 | CNN + MC Dropout + SHAP + triage |
| 23 | Cross-dataset generalisation summary | Comparison table and bar chart |

## Configuration you may want to change

| Variable | Cell | Default | Notes |
|---|---|---|---|
| `SEED` | 0 | `69` | Applies to NumPy, PyTorch, scikit-learn, XGBoost, RFE, SMOTE |
| `DATA_DIR_CANDIDATES` | 1 | Colab Drive path first | Add `'../dataset'` for local runs |
| `MI_TOP_N` / `FINAL_TOP_N` | 5 | `100` / `50` | Feature-selection stage sizes |
| `CTGAN_EPOCHS` | 6 | `10` | Library default is 300 — raising this strengthens the RQ5 CTGAN arm |
| `T` | 10 | `50` | MC Dropout forward passes |
| `budgets` | 12 | `[0.05, 0.08, 0.10]` | Analyst review budgets |
| `_FN_COST_LOCAL` / `_FP_COST_LOCAL` / `_REVIEW_COST_LOCAL` | 12 | `5_080_000` / `50` / `25` | Cost model constants |

## Runtime

The CNN grid search (cell 9, 108 configurations) and MC Dropout inference are the dominant costs. A GPU runtime is strongly recommended. Cells 18–23 add three more model-training rounds on much larger datasets, though at deliberately reduced budgets.

## Determinism

`SEED = 69` covers most components, but **Optuna and CTGAN are unseeded**, so re-running will shift some figures. For a fully reproducible run:

```python
study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=SEED))   # add sampler

synthesizer = CTGANSynthesizer(metadata, epochs=CTGAN_EPOCHS)  # SDV: see docs for seed control
```

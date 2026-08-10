# Datasets

The raw data files are **not committed to this repository**. They are large, and each is redistributed under its own publisher's terms. This file lists exactly what the notebook expects and where to obtain it.

## Expected files

Place all five files directly in this `dataset/` folder (or in the Google Drive folder configured in cell 1 of the notebook):

```
dataset/
├── MLRan_X_train_RFE.csv          # MLRan training features (3,905 × 487)
├── MLRan_X_test_RFE.csv           # MLRan test features (975 × 487)
├── MLRan_labels.csv               # MLRan labels, used for an integrity cross-check
├── Obfuscated-MalMem2022.parquet  # CIC-MalMem-2022 (58,058 × 57)
└── ember_2018_50000.parquet       # EMBER-2018, 5% sample (50,000 × 2,382)
```

Filenames must match exactly — the notebook's `resolve_data_path()` helper looks them up by name.

## Sources

### 1. MLRan (primary)

Behavioural ransomware dataset: 4,800+ dynamically analysed samples across 64 families collected 2006–2024, balanced against goodware.

- Paper: https://arxiv.org/abs/2505.18613
- The distributed `*_RFE.csv` files carry 483 behavioural features (already reduced by the dataset authors from 6.4M raw indicators via mutual information then RFE), plus four metadata columns: `sample_id`, `sample_type`, `family_label`, `type_label`.
- `sample_type` is the binary label used here: `1` = ransomware, `0` = goodware.

**Note:** the supplied feature files contain **no date or timestamp column**, which is why no chronological/temporal split could be constructed in this project. If temporally-labelled files become available from the authors, a time-aware evaluation becomes possible.

### 2. Obfuscated-MalMem2022 (secondary)

Memory-forensic dataset built from memory dumps of obfuscated spyware, ransomware and Trojan samples, with features extracted by VolMemLyzer.

- Source: https://www.unb.ca/cic/datasets/malmem-2022.html
- Paper: Carrier et al., ICISSP 2022, doi:10.5220/0010908200003120
- The published release is distributed as CSV (58,596 records, 55 features). The file used here is a **Parquet conversion with 58,058 rows and 57 columns**; the provenance of that difference is not documented in the project evidence and should be recorded before submission.
- The `Category` column is dropped by the notebook as a label-leakage guard, but a family identifier is extracted from it first (32,133 unique values) for the group-aware re-split.

### 3. EMBER-2018 (tertiary)

Static PE malware dataset: features from 1.1 million Windows binaries, 2,381 dimensions.

- Source: https://github.com/elastic/ember
- Paper: https://arxiv.org/abs/1804.04637
- The file used here is a **50,000-row sample (5.00%)** of a 1,000,000-row corpus, stored as Parquet with a `Label` column (`-1` = unlabelled, and the notebook removes such rows). The sampling procedure used to draw these 50,000 rows is not documented in the project evidence and should be recorded.

## Obtaining and converting the data

The two Parquet files are conversions of the publishers' original releases. If you are rebuilding them from source:

```python
import pandas as pd

# MalMem2022: CSV -> Parquet
pd.read_csv("Obfuscated-MalMem2022.csv").to_parquet(
    "Obfuscated-MalMem2022.parquet", index=False)

# EMBER-2018: build features with the official `ember` package first,
# then sample and write. Record the sampling method you use.
df = pd.read_parquet("ember_2018_full.parquet")
df.sample(n=50_000, random_state=69).to_parquet(
    "ember_2018_50000.parquet", index=False)
```

## If you want to commit the data anyway

`.gitignore` excludes `*.csv` and `*.parquet` inside this folder by default. GitHub rejects files above 100 MB and warns above 50 MB, so if you do need the data in the repository, use Git LFS:

```bash
git lfs install
git lfs track "dataset/*.parquet"
git lfs track "dataset/*.csv"
git add .gitattributes
```

Check each dataset's licence before redistributing it — several research corpora permit use but not republication.

## Ethical note

None of these datasets contains direct personal identifiers; they hold behavioural counters, memory-forensic statistics and PE header features. Two residual risks are worth stating rather than assuming away:

1. Goodware samples in a behavioural corpus may embed file paths, usernames or organisational artefacts inside string-derived features. This was not inspected in the current project.
2. A system of this kind deployed on live endpoints *would* process user-identifying telemetry, at which point GDPR obligations around lawful basis, data minimisation and retention apply. Nothing in this offline evaluation transfers that assurance to deployment.

# AgriVision AI - MLflow Experiment Tracking and Data Drift Analysis

Data drift analysis for **AgriVision AI**, a crop disease detection and advisory
platform (farmers upload a leaf photo, a model predicts the disease and the app
returns treatment advice). The project compares a stable *reference* window of
scan telemetry with a shifted *current* window, calculates the **Population
Stability Index (PSI)** per feature, and logs the run to **MLflow**.

> The dataset is **synthetic**. It is generated deterministically (seed 49) and
> contains no real farmer or production data. The results validate the
> monitoring pipeline; they do not describe a real production incident.

## Scenario

| Window | Represents | Image brightness | Sharpness | Confidence | Latency |
| --- | --- | --- | --- | --- | --- |
| Reference | Dry season, good light, sharp photos | mean 128 | mean 210 | mean 0.88 | mean 420 ms |
| Current | Monsoon, low light, blurrier photos | mean 104 | mean 150 | mean 0.79 | mean 470 ms |

`image_size_kb` is deliberately left unchanged as a **control feature**: a sound
drift check should report low change for it.

## Files

| File | Purpose |
| --- | --- |
| `analysis.py` | Generates the data, computes PSI, writes the CSVs and JSON report. Standard library only. |
| `mlflow_tracking.py` | Runs the analysis and logs parameters, metrics and artifacts to MLflow. |
| `noise_floor.py` | Shows why 500 records per window are used (PSI sampling noise). |
| `tests/test_analysis.py` | Unit tests (PSI maths, reproducibility, expected drift pattern). |
| `data/reference.csv`, `data/current.csv` | Generated telemetry, 500 records each. |
| `results/drift-report.json` | PSI per feature, rate changes, overall decision. |
| `requirements.txt` | No third-party packages needed for the analysis. |
| `requirements-mlflow.txt` | `mlflow==3.17.2`, only for `mlflow_tracking.py`. |

## Run the analysis (no installation needed)

Requires Python 3.8 or newer (uses `statistics.quantiles` and `statistics.fmean`; tested on Python 3.12.3). `mlflow_tracking.py` additionally needs MLflow 3.17.2, which requires Python 3.10 or newer.

```bash
python analysis.py            # Windows: python analysis.py
python noise_floor.py         # optional: PSI noise-floor demonstration
python -m unittest discover -s tests -v
```

## Run with MLflow tracking

```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements-mlflow.txt
python mlflow_tracking.py
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open http://127.0.0.1:5000 and select the experiment `agrivision-crop-disease-drift`.
Each run records:

- **Parameters:** `seed`, `records_per_window`, `psi_bins`, `psi_moderate_threshold`, `psi_significant_threshold`
- **Metrics:** `psi_<feature>`, `reference_mean_<feature>`, `current_mean_<feature>`, rates and their change for `expert_referral` and `correct_prediction`, `drifted_feature_count`, `overall_drift_detected`
- **Artifacts:** `data/reference.csv`, `data/current.csv`, `results/drift-report.json`

## Method

PSI uses 10 bins whose edges are the deciles of the reference window:

`PSI = sum((current% - reference%) * ln(current% / reference%))`

| PSI | Interpretation |
| --- | --- |
| < 0.10 | Low change |
| 0.10 to < 0.25 | Moderate change |
| >= 0.25 | Significant drift |

Overall drift is reported when at least one numeric feature reaches 0.25.

### Why 500 records per window

PSI is inflated by sampling noise when windows are small. `noise_floor.py`
compares two samples of the *same* distribution (400 trials, 10 bins):

| Records per window | Mean PSI | 95th percentile | Trials >= 0.25 |
| --- | --- | --- | --- |
| 100 | 0.197 | 0.367 | 24% |
| 200 | 0.093 | 0.169 | 0% |
| 500 | 0.038 | 0.068 | 0% |
| 1000 | 0.019 | 0.037 | 0% |

With 100 records a stable feature would be flagged as "significant drift" about
one time in four, so 500 per window is used.

## Results (seed 49, 500 records per window)

| Indicator | Reference | Current | PSI / change | Decision |
| --- | --- | --- | --- | --- |
| Image brightness mean | 128.21 | 103.96 | PSI 1.1539 | Significant drift |
| Image sharpness mean | 211.37 | 153.75 | PSI 1.7303 | Significant drift |
| Image size KB mean (control) | 322.78 | 318.38 | PSI 0.0319 | Low change |
| Top-1 confidence mean | 0.8749 | 0.7884 | PSI 0.9276 | Significant drift |
| Inference latency mean (ms) | 422.94 | 474.66 | PSI 0.5509 | Significant drift |
| Expert-referral rate | 0.4% | 23.2% | +22.8 percentage points | Degraded |
| Correct-prediction rate | 91.4% | 82.0% | -9.4 percentage points | Degraded |

## Publish to GitHub

The commit in this folder is already configured for user `RushikeshKulkarni07`
(`rushikeshkulkarni472@gmail.com`). Create an empty repository named
`agrivision-mlflow-drift-analysis` on GitHub (no README or .gitignore), then:

```bash
git remote -v        # origin should point to the repository below
git push -u origin main
```

If `origin` is missing:

```bash
git remote add origin https://github.com/RushikeshKulkarni07/agrivision-mlflow-drift-analysis.git
```

GitHub will ask you to sign in (browser login or a personal access token).
Never commit tokens, passwords or other credentials.

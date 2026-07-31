# Diabetes Readmission Risk — End-to-End Azure ML Pipeline

A full data engineering + machine learning pipeline built on Azure, predicting 30-day hospital readmission risk for diabetic patients using the UCI **Diabetes 130-US Hospitals (1999–2008)** dataset. Built as hands-on preparation for the **DP-100 (Designing and Implementing a Data Science Solution on Azure)** certification.

This isn't a notebook-only project — it covers the full path from raw CSV to a live, callable REST API, including the real infrastructure problems (schema bugs, auth failures, quota limits) encountered and resolved along the way.

---

## Architecture

```
Raw CSV (diabetic_data.csv)
        │
        ▼
Azure Data Lake Storage Gen2 (bronze / raw zone)
        │
        ▼  Azure Data Factory pipeline (Managed Identity auth)
        │  Copy Data → Stored Procedure (normalization)
        ▼
Azure SQL Database — normalized schema (silver layer)
  patients | encounters | diagnoses | medications | readmission_labels
        │
        ▼
Azure Machine Learning Workspace
  ├─ Compute Instance   (interactive notebooks, EDA)
  ├─ Compute Cluster    (0 min nodes, autoscale — training jobs)
  ├─ Data Assets        (versioned train/test snapshots)
  ├─ Command Job         (standalone training script, MLflow autologging)
  └─ Model Registry     (versioned model artifact)
        │
        ▼
Managed Online Endpoint (real-time REST API)
  → JSON in, readmission prediction out
```

---

## What was built

**Data layer**
- Data Lake Gen2 (bronze) → Azure Data Factory Copy Data activity → Azure SQL (silver) staging table
- Normalized relational schema (`patients`, `encounters`, `diagnoses`, `medications`, `readmission_labels`) built and populated via a T-SQL stored procedure
- Managed Identity authentication between Data Factory and SQL — no passwords or connection secrets stored anywhere in the pipeline

**ML workspace**
- Azure ML workspace with linked Storage, Key Vault, App Insights, Container Registry
- Compute Instance for interactive development, Compute Cluster (autoscaling, 0 minimum nodes) for training jobs
- Data versioned as registered Data Assets (SDK v2 `MLClient`), iterated as feature set improved

**Training**
- Baseline EDA with MLflow autologging — identified class imbalance (54% / 35% / 11% across `NO` / `>30` / `<30` readmission classes)
- RandomForestClassifier with `class_weight="balanced"`, trained as a standalone Python script and submitted as an Azure ML **Command Job** to the Compute Cluster (not run interactively) — the standard production training pattern
- Model registered to the Azure ML Model Registry with full lineage back to the training job and data version that produced it

**Deployment**
- Model deployed behind a **Managed Online Endpoint** — a live REST API
- Verified with real JSON payloads: differentiates correctly between low-risk and high-risk patient profiles

---

## Real problems debugged along the way

Worth calling out explicitly, since this is where most of the actual learning happened:

- **Primary key violation in the normalization stored procedure.** The raw CSV is encounter-level (a patient can appear multiple times), but `patients.patient_nbr` is a primary key. A naive `SELECT DISTINCT` across all columns didn't deduplicate correctly when non-key fields (like `weight`) varied between a patient's rows. Fixed using `ROW_NUMBER() OVER (PARTITION BY patient_nbr ...)` to deterministically pick one row per patient before insert.
- **Managed Identity / Entra ID authentication chain**, from granting Data Factory's identity `db_owner` rights via `CREATE USER ... FROM EXTERNAL PROVIDER`, to diagnosing why a SQL-authenticated admin session can't create Entra ID users (only an Entra ID admin can).
- **Serverless SQL auto-pause** causing intermittent connection timeouts — understanding when a "connection failed" error is actually just the database waking up, not a real fault.
- **Subscription CPU quota limits** blocking online endpoint deployment while a Compute Instance was also running — resolved by using a smaller instance SKU rather than over-provisioning.
- **A silently-failing pipeline debug run** that ADF reported as "succeeded" while producing zero rows in staging — traced through ADF's Monitor → Debug tab (not the default Triggered tab) to find the real row counts and error trail.

---

## Model performance

3-class classification (`NO`, `>30`, `<30` days to readmission) — a genuinely hard, well-known benchmark dataset:

| Class | Precision | Recall | F1 |
|---|---|---|---|
| `<30` | 0.19 | 0.35 | 0.25 |
| `>30` | 0.44 | 0.30 | 0.36 |
| `NO`  | 0.64 | 0.67 | 0.66 |

**Macro F1: 0.42** · **Accuracy: 51%**

This is a baseline, not a tuned final model. The focus of this project was the end-to-end Azure infrastructure and MLOps pattern rather than squeezing out maximum model accuracy — noted here transparently as a natural next iteration (candidates: XGBoost/LightGBM, diagnosis-code features, binary reframing of the target).

---

## Tech stack

`Azure Data Lake Storage Gen2` · `Azure Data Factory` · `Azure SQL Database` · `Azure Machine Learning (SDK v2)` · `MLflow` · `scikit-learn` · `pandas` · `T-SQL`

---

## Repo contents

- `01_eda_and_registration.ipynb` — EDA, MLflow tracking, data asset registration
- `train_component/train.py` — standalone training script (Command Job entry point)
- `deploy_component/score.py` — scoring script for the Managed Online Endpoint

---

## Next steps

- Expand features with diagnosis codes (`diag_1/2/3`) and try gradient-boosted models (XGBoost/LightGBM)
- Add a Responsible AI dashboard (fairness metrics across age/race/gender)
- Batch endpoint for nightly scoring, wired back into Azure SQL via Data Factory
- Data drift monitoring and a scheduled retraining pipeline

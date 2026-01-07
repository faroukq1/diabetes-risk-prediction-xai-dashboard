# TP #2 (Extension) — TODO (English)

## Setup & organization

- [ ] Create the project folders (data/, notebooks/, src/, reports/, dashboard/).
- [ ] Export/load the initial TP data warehouse (schema + data).
- [ ] Define the variable dictionary (BMI, age, fasting glucose, HbA1c, class, etc.).
- [ ] Define the deliverables plan (report, code, dashboard, presentation).

## Part 1 — Data Warehouse enrichment

## New dimension: Risk Factors

- [ ] Define attributes for `dim_risk_factors` (sedentary lifestyle, diet, family history, smoking, etc.).
- [ ] Choose the star schema model: link the dimension to the main fact table (e.g., `fact_patient_measures`).
- [ ] Create the dimension table + surrogate keys + constraints.
- [ ] Update the ETL (Talend/SQL/Pandas) to populate the new dimension.
- [ ] Fill it with consistent simulated data (document the simulation rules).

## Decision-making queries (OLAP)

- [ ] Write queries to analyze correlation/association between risk factors ↔ diabetes (by factor, age range, sex if available).
- [ ] Add segmentation queries (e.g., diabetes vs high sedentary level + BMI).
- [ ] Save all queries in a versioned SQL file.

## Bonus (optional): Simulated external data

- [ ] Choose simulated external variables (pollution, access to care, education, income…).
- [ ] Create related dimensions (e.g., `dim_environment`, `dim_socioeconomic`) or dedicated columns.
- [ ] Document the simulation method and link to the cube (geography/time level if available).
- [ ] Add 2–3 decision queries: “external impact ↔ diabetes prevalence”.

## Part 2 — Advanced Machine Learning

## Preparation & baseline

- [ ] Build the ML dataset from the DWH (clean joins, features, label).
- [ ] Clean data (missing values, outliers, encoding, scaling if needed).
- [ ] Train/validation/test split + cross-validation strategy.
- [ ] Handle class imbalance (e.g., class_weight / SMOTE if needed).
- [ ] Implement baselines: Random Forest + XGBoost.

## Hybrid model (Deep Learning + ensemble)

- [ ] Build an autoencoder (TensorFlow/PyTorch) for dimensionality reduction.
- [ ] Train the autoencoder + extract the latent representation.
- [ ] Train XGBoost on the latent features (hybrid).
- [ ] Compare “raw XGBoost” vs “Autoencoder → XGBoost”.

## Model explainability (XAI)

- [ ] Apply SHAP on XGBoost (global + local explanations).
- [ ] Explain 2 patient cases (one true positive and one false positive).
- [ ] (Optional) Use LIME to compare with SHAP.
- [ ] Save plots/figures + written interpretation.

## Anomaly detection

- [ ] Define the goal: atypical profiles / data collection errors / undiagnosed diabetes.
- [ ] Train Isolation Forest (unsupervised) on clinical variables/risk factors.
- [ ] Train One-Class SVM (for comparison).
- [ ] Inspect top anomalies + explain why (extreme values, inconsistencies).
- [ ] (Optional) Explain anomalies with SHAP (if feasible in your pipeline).

## Part 3 — Interactive visualization

## Dashboard (Power BI / Tableau / Plotly Dash)

- [ ] Choose the tool (Power BI recommended if initial TP was BI-focused).
- [ ] Build the data model (relationships + measures).
- [ ] Page 1: Patient distribution by BMI, age, and class (healthy/diabetic).
- [ ] Page 2: Time evolution of clinical indicators (e.g., fasting glucose, HbA1c) with filters.
- [ ] Page 3: Correlations between risk factors and diabetes (heatmap/bar + slicers).
- [ ] Add filters (sex, age group, factors, time period, area if external data exists).
- [ ] Export the dashboard file (or Dash code) + screenshots for the report.

## Part 4 — Evaluation & comparison

## Model benchmark

- [ ] Define the experiment table (RF, XGBoost, Hybrid, + anomalies as appendix).
- [ ] Compute metrics: accuracy, recall, F1-score, AUC-ROC.
- [ ] Measure training/inference time for each model.
- [ ] Produce the final comparison table + comments.

## Results analysis

- [ ] Explain pros/cons: performance, interpretability, complexity, runtime.
- [ ] Suggest improvements: more data, hyperparameter tuning, new features, better simulation, external validation.

## Deliverables

## Additional technical report

- [ ] Describe the enriched DWH (schema, dimensions, ETL, OLAP queries).
- [ ] Describe the models (baselines + hybrid + XAI + anomalies).
- [ ] Add results (metrics tables + SHAP/LIME figures + dashboard screenshots).
- [ ] Discussion: added value compared to the initial TP.

## Source code

- [ ] Python scripts: preprocessing, training, SHAP/LIME, anomaly detection.
- [ ] requirements.txt / environment.yml + run instructions.
- [ ] Clean notebook(s) for reproducibility.

## Presentation

- [ ] Slides: context, data, DWH, models, dashboard, results, limitations.
- [ ] Quick dashboard demo + 1 SHAP patient explanation.

## Obsidian checklist (syntax reminder)

- [ ] Tasks use the format `- [ ]` / `- [x]` for checkboxes.

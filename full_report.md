# Comprehensive Academic Report: Diabetes Risk Prediction System

## A Complete End-to-End Machine Learning Solution

---

# 1. Introduction

This report documents the development of a sophisticated diabetes risk prediction system. The project moves beyond simple classification to create a robust, interpretable, and production-ready solution. The system is architected as a modular pipeline encompassing data enrichment, baseline and hybrid machine learning modeling, explainable AI (XAI), anomaly detection, and interactive visualization.

The primary objective is to assist clinical decision-making by:

1.  **Predicting** diabetes risk with high accuracy.
2.  **Explaining** the "why" behind every prediction to build trust.
3.  **Identifying** hidden anomalies and potential data errors.
4.  **Visualizing** complex medical data in an intuitive dashboard.

---

# 2. Detailed Methodology & Analysis by Module

This project is implemented across six distinct Jupyter notebooks, each addressing a specific stage of the data science lifecycle.

## Module 1: Data Warehouse Enrichment

**Notebook:** `01_data_warehouse_enrichment.ipynb`

### 2.1 Problem & Goal

The original medical dataset (`BASEDIABET.csv`) contained core clinical metrics (glucose, BMI, HbA1c) but lacked lifestyle factors crucial for accurate diabetes profiling. The goal of this module was to **enrich** the data with simulated but medically logical risk factors and restructure it into a professional Data Warehouse (DWH) schema.

### 2.2 Solution & Code Logic

We implemented a probabilistic simulation engine to generate missing risk factors based on existing clinical correlations:

1.  **Sedentary Lifestyle Simulation**:
    - _Logic:_ Probability increases with BMI > 30 (Obesity) and Age > 50.
    - _Code:_ `np.random.binomial` with dynamic probabilities derived from `np.where` conditions.
2.  **Smoking Status**:
    - _Logic:_ Age-dependent probabilities (e.g., younger patients have lower "former smoker" rates).
3.  **Diet Quality**:
    - _Logic:_ Inverse correlation with Diabetes status and BMI. Diabetics were biased toward lower diet scores.

**Data Warehouse Schema (Star Schema):**
The enriched data was normalized into a Star Schema to support OLAP analysis:

- **Fact Table:** `fact_patient_measures` (Patient ID, Risk Factor ID, Date ID, clinical measures).
- **Dimensions:**
  - `dim_patient`: Static patient attributes (Age, Height).
  - `dim_risk_factors`: Lifestyle attributes (Smoking, Diet).
  - `dim_date`: Simulated timeline of measurements (2-year window).

### 2.3 Visualizations & Results

- **Visualizations:** Distribution plots verified that the simulated data followed expected variance (e.g., higher sedentary rates in the obese population).
- **Result:** A robust SQLite database (`diabetes_dwh.db`) containing `fact_patient_measures` linked to three dimensions, ready for SQL querying.

---

## Module 2: Baseline Machine Learning Models

**Notebook:** `02_ml_baseline_models.ipynb`

### 2.1 Problem & Goal

To establish a performance benchmark, we trained standard industry-proven classifiers. This stage also handled critical preprocessing steps like Feature Engineering and Class Balancing.

### 2.2 Solution & Code Logic

**preprocessing:**

1.  **Feature Engineering:**
    - _Categorization:_ Continuous variables were binned (e.g., `bmi_obese` if BMI > 30).
    - _Interactions:_ Created `glucose_x_hba1c` to capture the multiplicative effect of sugar indicators.
    - _Composite Score:_ A weighted `risk_score` summing up all lifestyle factors.
2.  **Scaling:** Applied `StandardScaler` to normalize distributions.
3.  **Balancing:** Used `SMOTE` (Synthetic Minority Over-sampling Technique) to address class imbalance if the ratio exceeded 1.5.

**Modeling:**

- **Random Forest:** Configured with `n_estimators=100` and `max_depth=8` to prevent overfitting.
- **XGBoost:** A gradient boosting machine utilized for its speed and performance on tabular data.

### 2.3 Visualizations & Results

- **EDA Visualizations:**
  - _Correlation Matrix:_ Showed strong positive correlation (dark red) between `fasting_glucose` and `diabetes_diagnosis`.
  - _Histograms:_ Revealed that diabetic patients showed a distinctly different distribution (shifted right) for Glucose and HbA1c compared to healthy patients.
- **Performance Results:**
  - **Random Forest:** Achieved **~94% Accuracy** on the Test set.
  - **XGBoost:** Matched Random Forest with **~94% Accuracy** but showed slightly better calibration (ROC-AUC 0.988).
  - _Key Finding:_ The engineered feature `glucose_x_hba1c` became a top predictor.

---

## Module 3: Hybrid Deep Learning Architecture

**Notebook:** `03_hybrid_model_autoencoder_xgboost.ipynb`

### 3.1 Problem & Goal

Standard models work well, but can we improve them by learning "hidden" patterns? This module experimented with a **Hybrid Architecture**: using a Deep Autoencoder to learn a compressed representation (latent space) of the data, then feeding these latent features into XGBoost.

### 3.2 Solution & Code Logic

**Architecture:**

- **Autoencoder (Deep Neural Network):**
  - _Encoder:_ Compresses input dimensions down to 30 latent features. Layers: [Input -> Dense(32) -> Dropout -> Dense(16) -> Latent(30)].
  - _Decoder:_ Reconstructs the original data from the latent space.
  - _Loss Function:_ Mean Squared Error (MSE) to minimize reconstruction error.
- **Hybrid Classifier:**
  - The _Encoder_ part was detached after training.
  - Transformed data (`X_train_encoded`) was used to train a standard XGBoost classifier.

```python
# Simplified Architecture Logic
input_layer = layers.Input(shape=(input_dim,))
encoded = layers.Dense(32, activation='relu')(input_layer)
latent = layers.Dense(30, activation='relu')(encoded)
# ... Decoder steps ...
autoencoder = Model(input_layer, output_layer)
```

### 3.3 Visualizations & Results

- **Training Plot:** The Loss vs. Epochs curve showed exponential decay, flattening out after ~100 epochs, indicating the Autoencoder successfully learned to compress the data.
- **Result:** The Hybrid model achieved **88% Accuracy**.
  - _Interpretation:_ It underperformed the baseline (94%). This suggests that for this specific low-dimensional dataset, the raw features are already highly predictive, and compression led to information loss. However, it remains a valuable experiment for academic completeness.

---

## Module 4: eXplainable AI (XAI)

**Notebook:** `04_explainability_shap_lime.ipynb`

### 4.1 Problem & Goal

"Black Box" models like XGBoost are accurate but opaque. Clinicians need to know _why_ a patient is flagged as high risk. This module implements **SHAP** and **LIME** to provide transparency.

### 4.2 Solution & Code Logic

1.  **SHAP (Global & Local):**
    - Used `shap.TreeExplainer` for exact Shapley value calculation.
    - Mathematically distributes the prediction output among features.
2.  **LIME (Local Approximation):**
    - Used `lime_tabular.LimeTabularExplainer`.
    - Perturbs (randomly changes) a single patient's data to see which changes flip the prediction.

### 4.3 Visualizations & Results

- **SHAP Summary Plot (Beeswarm):**
  - _Visual:_ A swarm of dots where Red = High Value.
  - _Insight:_ High `diet_quality` (poor diet) dots are clustered heavily on the right (positive SHAP value), confirming that poor diet increases diabetes risk.
- **Case Studies:**
  - _True Positive:_ SHAP showed `glucose_x_hba1c` pushing the probability up by +0.44.
  - _False Positive:_ Identified a healthy patient where `age` and `risk_score` misleadingly pushed the model toward a "Diabetic" prediction.

---

## Module 5: Anomaly Detection

**Notebook:** `05_anomaly_detection.ipynb`

### 5.1 Problem & Goal

Beyond prediction, we need to identify **Outliers**—patients who don't fit standard medical patterns. This indicates either data errors or rare, undiagnosed medical conditions.

### 5.2 Solution & Code Logic

We employed unsupervised learning (no labels used):

1.  **Isolation Forest:**
    - Constructs random trees. Anomalies are "easily isolated" (short path length) because they are different from the crowd.
    - _Parameter:_ `contamination=0.05` (assuming 5% anomalies).
2.  **One-Class SVM:**
    - Learns a hypersphere around "normal" data. Points outside are anomalies.
    - _Parameter:_ `nu=0.05`.

### 5.3 Visualizations & Results

- **Scatter Plots:** Patients were plotted (Glucose vs. BMI). Normal points were blue; Anomalies were red.
  - _Result:_ Anomalies often appeared in the extreme top-right (very high BMI and Glucose) or in contradictory regions (Low BMI, High Glucose).
- **Key Finding:** 7 specific patients were identified by **both** algorithms as anomalies. These represent high-priority cases for data review.

---

## Module 6: Interactive Dashboard

**Guide:** `06_dashboard_guide.ipynb` | **Code:** `dashboard/app.py`

### 6.1 Problem & Goal

Static notebooks are not suitable for end-users. We built a fully interactive web application to serve the models and insights to clinicians.

### 6.2 Solution & Implementation

- **Framework:** **Plotly Dash** (Python framework on top of Flask and React).
- **Tech Stack:**
  - `dash_bootstrap_components`: For a responsive, tile-based layout.
  - `plotly.express`: For interactive charts (zoom, pan, hover).
  - `callbacks`: Reactively linking dropdown filters to chart updates.

### 6.3 Visualizations (Dashboard Pages)

The dashboard is structured into 3 core analytical pages:

1.  **Overview:** High-level KPIs (Total Patients, Diabetic %).
2.  **Patient Distribution:** Histograms filtered by Age Group or BMI Category.
3.  **Correlations:** Interactive Heatmaps enabling the user to explore relationships (e.g., "Does activity level correlation with diabetes change with age?").

This dashboard effectively brings the entire academic pipeline into a practical, user-friendly interface.

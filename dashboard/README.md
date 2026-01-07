# 🏥 Diabetes Risk Prediction - Interactive Dashboard

## Overview

This is a comprehensive **Plotly Dash** dashboard for analyzing diabetes risk prediction data, integrating advanced machine learning results and explainability techniques (XAI).

## Features

### 📊 Basic Analytics Pages

1. **Overview** - Executive summary with key metrics
2. **Patient Distribution** - Demographics analysis (age, BMI, diabetes status)
3. **Temporal Evolution** - Clinical indicators trends (GAJ, HbA1c)
4. **Risk Correlations** - Correlation heatmaps and scatter plots

### 🤖 Advanced ML Analytics Pages

5. **Models Performance** - Comparison of RF, XGBoost, and Hybrid (Autoencoder + XGBoost)
6. **XAI Explainability** - SHAP and LIME visualizations for model interpretability
7. **Anomaly Detection** - Isolation Forest and One-Class SVM results

## Installation

### 1. Install Dependencies

```bash
pip install -r ../requirements.txt
```

Required packages:

- `plotly`
- `dash`
- `dash-bootstrap-components`
- `pandas`
- `numpy`

### 2. Verify Data Files

Ensure these files exist:

- `../data/BASEDIABET.csv` - Main dataset
- `../reports/all_models_comparison.csv` - ML models metrics
- `../reports/anomaly_detection_results.csv` - Anomaly detection results
- `../reports/*.png` - Visualization images (automatically copied to `assets/`)

## Usage

### Run the Complete Dashboard

```bash
python app_complete.py
```

### Run the Basic Dashboard (without ML results)

```bash
python app.py
```

The dashboard will be available at: **http://localhost:8050**

## Dashboard Structure

```
📁 dashboard/
├── app.py                    # Basic dashboard (3 pages)
├── app_complete.py           # Complete dashboard (7 pages)
├── assets/                   # Images and static files
│   ├── shap_*.png           # SHAP visualizations
│   ├── lime_*.png           # LIME visualizations
│   └── anomaly_*.png        # Anomaly detection plots
└── README.md                # This file
```

## Pages Description

### 1. Overview (/)

- Total patients, diabetic count, best model F1-score
- Dataset summary and ML models summary
- Navigation guide

### 2. Patient Distribution (/distribution)

- **Filters**: Age range, diabetes status, BMI category
- **KPIs**: Total patients, diabetic count, average age/BMI
- **Charts**:
  - Age distribution histogram
  - BMI distribution histogram
  - Age group vs diabetes status
  - BMI category pie chart

### 3. Temporal Evolution (/temporal)

- **Filters**: Diabetes status, age group, aggregation period (daily/weekly/monthly)
- **Charts**:
  - GAJ (Fasting glucose) evolution over time
  - HbA1c evolution over time
  - Combined clinical indicators (GAJ, HbA1c, BMI)

### 4. Risk Correlations (/correlations)

- **Filters**: Diabetes status, age group
- **Charts**:
  - Correlation heatmap (all clinical variables)
  - BMI vs GAJ scatter plot with trendlines
  - Age vs HbA1c scatter plot with trendlines
  - Clinical variables boxplots

### 5. Models Performance (/models)

- **Metrics Table**: All models comparison (Test set)
- **Charts**:
  - Accuracy, F1-Score, ROC-AUC bar charts
  - Radar chart (all metrics)
  - Performance across Train/Validation/Test sets

**Models Compared:**

- Random Forest (baseline)
- XGBoost (baseline)
- Hybrid (Autoencoder + XGBoost)

### 6. XAI Explainability (/explainability)

- **SHAP Analysis**:
  - Feature importance plot
  - Summary plot (beeswarm)
  - Case studies (true positive, false positive)
- **LIME Analysis**:
  - Local explanations for specific cases
  - SHAP vs LIME comparison

### 7. Anomaly Detection (/anomalies)

- **Metrics**:
  - Isolation Forest anomalies count
  - One-Class SVM anomalies count
  - Consensus anomalies (both methods agree)
- **Charts**:
  - BMI vs GAJ scatter (highlighting anomalies)
  - Anomaly score distributions (both methods)
  - PCA visualization
  - Feature distributions

## Interactive Features

### Filters Available:

- **Age Range Slider**: Filter patients by age
- **Diabetes Status**: All / Healthy / Diabetic
- **BMI Category**: All / Underweight / Normal / Overweight / Obese
- **Age Group**: <25, 25-35, 36-45, 46-55, 56-65, 65+
- **Time Period**: Daily / Weekly / Monthly aggregation

### Navigation:

- Sidebar menu for easy page switching
- Clean, modern Bootstrap design
- Font Awesome icons for better UX
- Responsive layout

## Technical Details

### Technologies Used:

- **Framework**: Plotly Dash 2.x
- **UI**: Dash Bootstrap Components
- **Visualizations**: Plotly Express & Plotly Graph Objects
- **Data Processing**: Pandas, NumPy
- **Theme**: Bootstrap (Flatly-inspired)

### Color Scheme:

- **Primary**: #2E86AB (Blue)
- **Secondary**: #A23B72 (Purple)
- **Success**: #06A77D (Green)
- **Danger**: #E63946 (Red)
- **Warning**: #F18F01 (Orange)
- **Healthy**: #06A77D (Green)
- **Diabetic**: #E63946 (Red)
- **Anomaly**: #FF6B6B (Light Red)

## Exporting for Report

### Taking Screenshots:

1. Navigate to each page
2. Use browser screenshot tool or Snipping Tool
3. Save screenshots for your report

### Recommended Screenshots:

- Overview page (KPIs)
- Patient distribution charts
- Temporal evolution (GAJ & HbA1c)
- Correlation heatmap
- Models comparison (radar chart)
- SHAP feature importance
- SHAP vs LIME comparison
- Anomaly detection scatter plot

## Troubleshooting

### Port Already in Use:

```bash
# Change port in app_complete.py (last line)
app.run(debug=True, host='0.0.0.0', port=8051)
```

### Missing Images:

```bash
# Re-copy images from reports
cp ../reports/*.png assets/
```

### Module Not Found:

```bash
# Reinstall dependencies
pip install -r ../requirements.txt
```

## Integration with Homework (TP#2)

This dashboard addresses **Partie 3 - Visualisation Interactive**:

✅ **Page 1**: Patient distribution by IMC, age, class (healthy/diabetic)
✅ **Page 2**: Temporal evolution of GAJ, HbA1c (with filters)
✅ **Page 3**: Risk factors correlations (heatmap + slicers)
✅ **Filters**: Sex (if available), age group, factors, period
✅ **Export**: Code available + take screenshots for report

**Bonus - Advanced ML Integration:**

- ✅ Models comparison (RF, XGBoost, Hybrid)
- ✅ XAI explainability (SHAP + LIME)
- ✅ Anomaly detection visualization

## Contact & Support

For issues or questions about the dashboard, refer to:

- Plotly Dash documentation: https://dash.plotly.com/
- Bootstrap components: https://dash-bootstrap-components.opensource.faculty.ai/

---

**Created for**: TP #2 Extension - Diabetes Risk Prediction Project
**Framework**: Plotly Dash
**Date**: January 2026

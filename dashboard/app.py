"""
Diabetes Risk Prediction - Complete Interactive Dashboard
Plotly Dash Application with ML Results Integration
"""

import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path

# Initialize the Dash app with Bootstrap theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True,
    title="Diabetes Dashboard - Complete"
)

# Load data
BASE_DIR = Path(__file__).parent.parent
df = pd.read_csv(BASE_DIR / 'data' / 'BASEDIABET.csv')
models_comparison = pd.read_csv(BASE_DIR / 'reports' / 'all_models_comparison.csv')
anomaly_results = pd.read_csv(BASE_DIR / 'reports' / 'anomaly_detection_results.csv')

# Add derived columns
df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 55, 65, 100], 
                          labels=['<25', '25-35', '36-45', '46-55', '56-65', '65+'])
df['bmi_category'] = pd.cut(df['bmi']*10000, bins=[0, 18.5, 25, 30, 35, 100],
                             labels=['Underweight', 'Normal', 'Overweight', 'Obese', 'Severely Obese'])
df['diabetes_status'] = df['type_diabete'].map({0: 'Healthy', 1: 'Diabetic'})
df['bmi_scaled'] = df['bmi'] * 10000

# Merge anomaly data
df = df.merge(anomaly_results[['patient_id', 'iso_anomaly', 'iso_score', 'consensus_anomaly']], 
              left_index=True, right_on='patient_id', how='left')

# Create temporal data
df['date'] = pd.date_range(start='2023-01-01', periods=len(df), freq='D')
df['month'] = df['date'].dt.to_period('M').astype(str)

# Color schemes
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'success': '#06A77D',
    'danger': '#E63946',
    'warning': '#F18F01',
    'diabetic': '#E63946',
    'healthy': '#06A77D',
    'anomaly': '#FF6B6B'
}

# ==================== SIDEBAR ====================

sidebar = html.Div(
    [
        html.Div(
            [
                html.H3("🏥 Diabetes", className="text-white mb-1"),
                html.H6("Analytics Dashboard", className="text-white-50 mb-4"),
                html.Hr(className="bg-white"),
                dbc.Nav(
                    [
                        dbc.NavLink(
                            [html.I(className="fas fa-home me-2"), "Overview"],
                            href="/",
                            active="exact",
                            className="text-white mb-2"
                        ),
                        dbc.NavLink(
                            [html.I(className="fas fa-users me-2"), "Patient Distribution"],
                            href="/distribution",
                            active="exact",
                            className="text-white mb-2"
                        ),
                        dbc.NavLink(
                            [html.I(className="fas fa-chart-line me-2"), "Temporal Evolution"],
                            href="/temporal",
                            active="exact",
                            className="text-white mb-2"
                        ),
                        dbc.NavLink(
                            [html.I(className="fas fa-project-diagram me-2"), "Risk Correlations"],
                            href="/correlations",
                            active="exact",
                            className="text-white mb-2"
                        ),
                        html.Hr(className="bg-white-50 my-3"),
                        html.P("🤖 ML Analysis", className="text-white-50 small mb-2 ms-3"),
                        dbc.NavLink(
                            [html.I(className="fas fa-brain me-2"), "Models Performance"],
                            href="/models",
                            active="exact",
                            className="text-white mb-2"
                        ),
                        dbc.NavLink(
                            [html.I(className="fas fa-lightbulb me-2"), "XAI Explainability"],
                            href="/explainability",
                            active="exact",
                            className="text-white mb-2"
                        ),
                        dbc.NavLink(
                            [html.I(className="fas fa-exclamation-triangle me-2"), "Anomaly Detection"],
                            href="/anomalies",
                            active="exact",
                            className="text-white mb-2"
                        ),
                    ],
                    vertical=True,
                    pills=True,
                ),
            ],
            style={
                "padding": "2rem 1rem",
                "background": "linear-gradient(180deg, #2E86AB 0%, #1a4d6b 100%)",
                "height": "100vh",
                "position": "fixed",
                "width": "280px",
                "overflow-y": "auto"
            }
        ),
    ],
    style={"width": "280px"}
)

content = html.Div(
    id="page-content",
    style={
        "margin-left": "280px",
        "padding": "2rem",
        "background-color": "#f8f9fa",
        "min-height": "100vh"
    }
)

app.layout = html.Div([dcc.Location(id='url', refresh=False), sidebar, content])

# ==================== PAGE 0: OVERVIEW ====================

def create_overview():
    total_patients = len(df)
    diabetic = len(df[df['type_diabete'] == 1])
    anomalies = len(df[df['consensus_anomaly'] == 1]) if 'consensus_anomaly' in df.columns else 0
    
    # Get best model
    test_metrics = models_comparison[models_comparison['Set'] == 'Test']
    best_model = test_metrics.loc[test_metrics['f1'].idxmax(), 'Model']
    best_f1 = test_metrics['f1'].max()
    
    return html.Div([
        html.H1("Dashboard Overview", className="mb-4"),
        html.P("Comprehensive Diabetes Risk Analysis with Advanced ML", className="lead text-muted mb-4"),
        
        # KPI Cards
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.I(className="fas fa-users fa-2x text-primary mb-3"),
                            html.H2(f"{total_patients}", className="mb-0"),
                            html.P("Total Patients", className="text-muted mb-0")
                        ], className="text-center")
                    ])
                ], className="shadow-sm")
            ], md=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.I(className="fas fa-heartbeat fa-2x text-danger mb-3"),
                            html.H2(f"{diabetic}", className="mb-0"),
                            html.P(f"Diabetic ({diabetic/total_patients*100:.1f}%)", className="text-muted mb-0")
                        ], className="text-center")
                    ])
                ], className="shadow-sm")
            ], md=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.I(className="fas fa-trophy fa-2x text-success mb-3"),
                            html.H2(f"{best_f1:.2f}", className="mb-0"),
                            html.P(f"Best F1-Score ({best_model})", className="text-muted mb-0")
                        ], className="text-center")
                    ])
                ], className="shadow-sm")
            ], md=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.I(className="fas fa-exclamation-triangle fa-2x text-warning mb-3"),
                            html.H2(f"{anomalies}", className="mb-0"),
                            html.P(f"Anomalies Detected", className="text-muted mb-0")
                        ], className="text-center")
                    ])
                ], className="shadow-sm")
            ], md=3),
        ], className="mb-4"),
        
        # Quick Insights
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([html.I(className="fas fa-chart-bar me-2"), "Dataset Summary"]),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.P([html.Strong("Age Range: "), f"{df['age'].min():.0f} - {df['age'].max():.0f} years"]),
                                html.P([html.Strong("Avg BMI: "), f"{df['bmi_scaled'].mean():.1f}"]),
                                html.P([html.Strong("Avg GAJ: "), f"{df['gaj'].mean():.1f} mg/dL"]),
                            ], md=6),
                            dbc.Col([
                                html.P([html.Strong("Avg HbA1c: "), f"{df['hba1c'].mean():.1f}%"]),
                                html.P([html.Strong("Diabetic Rate: "), f"{diabetic/total_patients*100:.1f}%"]),
                                html.P([html.Strong("Records: "), f"{total_patients}"]),
                            ], md=6),
                        ])
                    ])
                ], className="shadow-sm")
            ], md=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([html.I(className="fas fa-brain me-2"), "ML Models Summary"]),
                    dbc.CardBody([
                        html.P([html.Strong("Models Trained: "), "3 (RF, XGBoost, Hybrid AE+XGB)"]),
                        html.P([html.Strong("Best Model: "), f"{best_model}"]),
                        html.P([html.Strong("Best Test F1: "), f"{best_f1:.4f}"]),
                        html.P([html.Strong("XAI Methods: "), "SHAP, LIME"]),
                        html.P([html.Strong("Anomaly Detection: "), "Isolation Forest, One-Class SVM"]),
                    ])
                ], className="shadow-sm")
            ], md=6),
        ], className="mb-4"),
        
        # Navigation Guide
        dbc.Card([
            dbc.CardHeader([html.I(className="fas fa-compass me-2"), "Dashboard Navigation"]),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H6([html.I(className="fas fa-users me-2 text-primary"), "Patient Distribution"]),
                        html.P("Explore patient demographics by age, BMI, and diabetes status", className="small text-muted")
                    ], md=4),
                    dbc.Col([
                        html.H6([html.I(className="fas fa-chart-line me-2 text-primary"), "Temporal Evolution"]),
                        html.P("Track clinical indicators (GAJ, HbA1c) over time", className="small text-muted")
                    ], md=4),
                    dbc.Col([
                        html.H6([html.I(className="fas fa-project-diagram me-2 text-primary"), "Risk Correlations"]),
                        html.P("Analyze correlations between risk factors and diabetes", className="small text-muted")
                    ], md=4),
                ]),
                html.Hr(),
                dbc.Row([
                    dbc.Col([
                        html.H6([html.I(className="fas fa-brain me-2 text-success"), "Models Performance"]),
                        html.P("Compare ML models: RF, XGBoost, Hybrid", className="small text-muted")
                    ], md=4),
                    dbc.Col([
                        html.H6([html.I(className="fas fa-lightbulb me-2 text-success"), "XAI Explainability"]),
                        html.P("Understand model predictions with SHAP and LIME", className="small text-muted")
                    ], md=4),
                    dbc.Col([
                        html.H6([html.I(className="fas fa-exclamation-triangle me-2 text-success"), "Anomaly Detection"]),
                        html.P("Identify atypical patient profiles", className="small text-muted")
                    ], md=4),
                ])
            ])
        ], className="shadow-sm")
    ])

# ==================== PAGE 1: PATIENT DISTRIBUTION ====================

def create_distribution():
    return html.Div([
        html.H1("Patient Distribution Analysis", className="mb-4"),
        
        dbc.Card([
            dbc.CardBody([
                html.H5("Filters", className="mb-3"),
                dbc.Row([
                    dbc.Col([
                        html.Label("Age Range"),
                        dcc.RangeSlider(
                            id='age-slider',
                            min=df['age'].min(),
                            max=df['age'].max(),
                            value=[df['age'].min(), df['age'].max()],
                            marks={i: str(i) for i in range(int(df['age'].min()), int(df['age'].max())+1, 10)},
                            tooltip={"placement": "bottom", "always_visible": True}
                        )
                    ], md=6),
                    dbc.Col([
                        html.Label("Diabetes Status"),
                        dcc.Dropdown(
                            id='diabetes-filter',
                            options=[
                                {'label': 'All', 'value': 'all'},
                                {'label': 'Healthy', 'value': 0},
                                {'label': 'Diabetic', 'value': 1}
                            ],
                            value='all',
                            clearable=False
                        )
                    ], md=3),
                    dbc.Col([
                        html.Label("BMI Category"),
                        dcc.Dropdown(
                            id='bmi-filter',
                            options=[{'label': 'All', 'value': 'all'}] + 
                                    [{'label': cat, 'value': cat} for cat in df['bmi_category'].dropna().unique()],
                            value='all',
                            clearable=False
                        )
                    ], md=3),
                ]),
            ])
        ], className="mb-4 shadow-sm"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(id='total-patients', className="text-primary"),
                        html.P("Total Patients", className="text-muted")
                    ])
                ], className="shadow-sm")
            ], md=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(id='diabetic-count', className="text-danger"),
                        html.P("Diabetic Patients", className="text-muted")
                    ])
                ], className="shadow-sm")
            ], md=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(id='avg-age', className="text-info"),
                        html.P("Average Age", className="text-muted")
                    ])
                ], className="shadow-sm")
            ], md=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(id='avg-bmi', className="text-warning"),
                        html.P("Average BMI", className="text-muted")
                    ])
                ], className="shadow-sm")
            ], md=3),
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Age Distribution by Diabetes Status"),
                        dcc.Graph(id='age-distribution-chart')
                    ])
                ], className="shadow-sm")
            ], md=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("BMI Distribution by Diabetes Status"),
                        dcc.Graph(id='bmi-distribution-chart')
                    ])
                ], className="shadow-sm")
            ], md=6),
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Age Group vs Diabetes Status"),
                        dcc.Graph(id='age-group-chart')
                    ])
                ], className="shadow-sm")
            ], md=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("BMI Category Distribution"),
                        dcc.Graph(id='bmi-category-chart')
                    ])
                ], className="shadow-sm")
            ], md=6),
        ]),
    ])

# ==================== PAGE 2: TEMPORAL EVOLUTION ====================

def create_temporal():
    return html.Div([
        html.H1("Temporal Evolution of Clinical Indicators", className="mb-4"),
        
        dbc.Card([
            dbc.CardBody([
                html.H5("Filters", className="mb-3"),
                dbc.Row([
                    dbc.Col([
                        html.Label("Diabetes Status"),
                        dcc.Dropdown(
                            id='diabetes-filter-p2',
                            options=[
                                {'label': 'All', 'value': 'all'},
                                {'label': 'Healthy', 'value': 0},
                                {'label': 'Diabetic', 'value': 1}
                            ],
                            value='all',
                            clearable=False
                        )
                    ], md=3),
                    dbc.Col([
                        html.Label("Age Group"),
                        dcc.Dropdown(
                            id='age-group-filter-p2',
                            options=[{'label': 'All', 'value': 'all'}] + 
                                    [{'label': str(ag), 'value': str(ag)} for ag in df['age_group'].dropna().unique()],
                            value='all',
                            clearable=False
                        )
                    ], md=3),
                    dbc.Col([
                        html.Label("Aggregation Period"),
                        dcc.Dropdown(
                            id='period-filter',
                            options=[
                                {'label': 'Daily', 'value': 'D'},
                                {'label': 'Weekly', 'value': 'W'},
                                {'label': 'Monthly', 'value': 'M'}
                            ],
                            value='M',
                            clearable=False
                        )
                    ], md=3),
                ]),
            ])
        ], className="mb-4 shadow-sm"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Fasting Glucose (GAJ) Evolution"),
                        dcc.Graph(id='gaj-evolution-chart')
                    ])
                ], className="shadow-sm")
            ], md=12),
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("HbA1c Evolution"),
                        dcc.Graph(id='hba1c-evolution-chart')
                    ])
                ], className="shadow-sm")
            ], md=12),
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Combined Clinical Indicators"),
                        dcc.Graph(id='combined-indicators-chart')
                    ])
                ], className="shadow-sm")
            ], md=12),
        ]),
    ])

# ==================== PAGE 3: RISK CORRELATIONS ====================

def create_correlations():
    return html.Div([
        html.H1("Risk Factor Correlations", className="mb-4"),
        
        dbc.Card([
            dbc.CardBody([
                html.H5("Filters", className="mb-3"),
                dbc.Row([
                    dbc.Col([
                        html.Label("Diabetes Status"),
                        dcc.Dropdown(
                            id='diabetes-filter-p3',
                            options=[
                                {'label': 'All', 'value': 'all'},
                                {'label': 'Healthy', 'value': 0},
                                {'label': 'Diabetic', 'value': 1}
                            ],
                            value='all',
                            clearable=False
                        )
                    ], md=4),
                    dbc.Col([
                        html.Label("Age Group"),
                        dcc.Dropdown(
                            id='age-group-filter-p3',
                            options=[{'label': 'All', 'value': 'all'}] + 
                                    [{'label': str(ag), 'value': str(ag)} for ag in df['age_group'].dropna().unique()],
                            value='all',
                            clearable=False
                        )
                    ], md=4),
                ]),
            ])
        ], className="mb-4 shadow-sm"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Correlation Heatmap - Clinical Variables"),
                        dcc.Graph(id='correlation-heatmap')
                    ])
                ], className="shadow-sm")
            ], md=12),
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("BMI vs GAJ by Diabetes Status"),
                        dcc.Graph(id='bmi-gaj-scatter')
                    ])
                ], className="shadow-sm")
            ], md=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Age vs HbA1c by Diabetes Status"),
                        dcc.Graph(id='age-hba1c-scatter')
                    ])
                ], className="shadow-sm")
            ], md=6),
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Clinical Variables by Diabetes Status"),
                        dcc.Graph(id='clinical-boxplot')
                    ])
                ], className="shadow-sm")
            ], md=12),
        ]),
    ])

# ==================== PAGE 4: MODELS PERFORMANCE ====================

def create_models():
    return html.Div([
        html.H1("Machine Learning Models Performance", className="mb-4"),
        html.P("Comparison of Random Forest, XGBoost, and Hybrid (Autoencoder + XGBoost)", className="lead text-muted mb-4"),
        
        # Model comparison table
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([html.I(className="fas fa-table me-2"), "Models Comparison - Test Set"]),
                    dbc.CardBody([
                        dcc.Graph(id='models-comparison-table')
                    ])
                ], className="shadow-sm")
            ], md=12),
        ], className="mb-4"),
        
        # Metrics comparison charts
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Accuracy Comparison"),
                        dcc.Graph(id='accuracy-comparison')
                    ])
                ], className="shadow-sm")
            ], md=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("F1-Score Comparison"),
                        dcc.Graph(id='f1-comparison')
                    ])
                ], className="shadow-sm")
            ], md=6),
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("ROC-AUC Comparison"),
                        dcc.Graph(id='roc-comparison')
                    ])
                ], className="shadow-sm")
            ], md=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("All Metrics - Test Set"),
                        dcc.Graph(id='all-metrics-radar')
                    ])
                ], className="shadow-sm")
            ], md=6),
        ], className="mb-4"),
        
        # Training vs Test performance
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Model Performance Across Sets"),
                        dcc.Graph(id='performance-across-sets')
                    ])
                ], className="shadow-sm")
            ], md=12),
        ]),
    ])

# ==================== PAGE 5: XAI EXPLAINABILITY ====================

def create_explainability():
    img_dir = BASE_DIR / 'reports'
    
    return html.Div([
        html.H1("Model Explainability (XAI)", className="mb-4"),
        html.P("Understanding model predictions with SHAP and LIME", className="lead text-muted mb-4"),
        
        # SHAP Analysis
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([html.I(className="fas fa-chart-bar me-2"), "SHAP Feature Importance - XGBoost"]),
                    dbc.CardBody([
                        html.Img(src=f'/assets/shap_feature_importance_xgb.png', 
                                style={'width': '100%', 'height': 'auto'})
                    ])
                ], className="shadow-sm")
            ], md=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([html.I(className="fas fa-chart-bar me-2"), "SHAP Summary Plot"]),
                    dbc.CardBody([
                        html.Img(src=f'/assets/shap_summary_plot_xgb.png', 
                                style={'width': '100%', 'height': 'auto'})
                    ])
                ], className="shadow-sm")
            ], md=6),
        ], className="mb-4"),
        
        # Case Studies
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([html.I(className="fas fa-user-check me-2"), "Case 1: True Positive (SHAP)"]),
                    dbc.CardBody([
                        html.Img(src=f'/assets/shap_case1_true_positive.png', 
                                style={'width': '100%', 'height': 'auto'})
                    ])
                ], className="shadow-sm")
            ], md=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([html.I(className="fas fa-user-times me-2"), "Case 2: False Positive (SHAP)"]),
                    dbc.CardBody([
                        html.Img(src=f'/assets/shap_case2_false_positive.png', 
                                style={'width': '100%', 'height': 'auto'})
                    ])
                ], className="shadow-sm")
            ], md=6),
        ], className="mb-4"),
        
        # LIME Analysis
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([html.I(className="fas fa-microscope me-2"), "LIME - Case 1"]),
                    dbc.CardBody([
                        html.Img(src=f'/assets/lime_case1_true_positive.png', 
                                style={'width': '100%', 'height': 'auto'})
                    ])
                ], className="shadow-sm")
            ], md=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([html.I(className="fas fa-microscope me-2"), "LIME - Case 2"]),
                    dbc.CardBody([
                        html.Img(src=f'/assets/lime_case2_false_positive.png', 
                                style={'width': '100%', 'height': 'auto'})
                    ])
                ], className="shadow-sm")
            ], md=6),
        ], className="mb-4"),
        
        # SHAP vs LIME
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([html.I(className="fas fa-balance-scale me-2"), "SHAP vs LIME Comparison"]),
                    dbc.CardBody([
                        html.Img(src=f'/assets/shap_vs_lime_comparison.png', 
                                style={'width': '100%', 'height': 'auto'})
                    ])
                ], className="shadow-sm")
            ], md=12),
        ]),
    ])

# ==================== PAGE 6: ANOMALY DETECTION ====================

def create_anomalies():
    return html.Div([
        html.H1("Anomaly Detection", className="mb-4"),
        html.P("Identifying atypical patient profiles using Isolation Forest and One-Class SVM", 
               className="lead text-muted mb-4"),
        
        # Statistics
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.I(className="fas fa-exclamation-circle fa-2x text-warning mb-3"),
                            html.H2(id='iso-anomalies-count', className="mb-0"),
                            html.P("Isolation Forest Anomalies", className="text-muted mb-0")
                        ], className="text-center")
                    ])
                ], className="shadow-sm")
            ], md=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.I(className="fas fa-shield-alt fa-2x text-danger mb-3"),
                            html.H2(id='ocsvm-anomalies-count', className="mb-0"),
                            html.P("One-Class SVM Anomalies", className="text-muted mb-0")
                        ], className="text-center")
                    ])
                ], className="shadow-sm")
            ], md=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.I(className="fas fa-check-double fa-2x text-success mb-3"),
                            html.H2(id='consensus-anomalies-count', className="mb-0"),
                            html.P("Consensus Anomalies", className="text-muted mb-0")
                        ], className="text-center")
                    ])
                ], className="shadow-sm")
            ], md=4),
        ], className="mb-4"),
        
        # Visualizations
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Anomaly Detection Results"),
                        dcc.Graph(id='anomaly-scatter')
                    ])
                ], className="shadow-sm")
            ], md=12),
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Anomaly Score Distribution - Isolation Forest"),
                        dcc.Graph(id='iso-score-dist')
                    ])
                ], className="shadow-sm")
            ], md=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Anomaly Score Distribution - One-Class SVM"),
                        dcc.Graph(id='ocsvm-score-dist')
                    ])
                ], className="shadow-sm")
            ], md=6),
        ], className="mb-4"),
        
        # Images from reports
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([html.I(className="fas fa-chart-area me-2"), "PCA Visualization"]),
                    dbc.CardBody([
                        html.Img(src=f'/assets/anomaly_detection_pca_visualization.png', 
                                style={'width': '100%', 'height': 'auto'})
                    ])
                ], className="shadow-sm")
            ], md=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([html.I(className="fas fa-chart-bar me-2"), "Feature Distributions"]),
                    dbc.CardBody([
                        html.Img(src=f'/assets/anomaly_feature_distributions.png', 
                                style={'width': '100%', 'height': 'auto'})
                    ])
                ], className="shadow-sm")
            ], md=6),
        ]),
    ])

# ==================== CALLBACKS ====================

@callback(Output('page-content', 'children'), [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/distribution':
        return create_distribution()
    elif pathname == '/temporal':
        return create_temporal()
    elif pathname == '/correlations':
        return create_correlations()
    elif pathname == '/models':
        return create_models()
    elif pathname == '/explainability':
        return create_explainability()
    elif pathname == '/anomalies':
        return create_anomalies()
    else:
        return create_overview()

# ========== DISTRIBUTION PAGE CALLBACKS ==========

@callback(
    [Output('total-patients', 'children'),
     Output('diabetic-count', 'children'),
     Output('avg-age', 'children'),
     Output('avg-bmi', 'children'),
     Output('age-distribution-chart', 'figure'),
     Output('bmi-distribution-chart', 'figure'),
     Output('age-group-chart', 'figure'),
     Output('bmi-category-chart', 'figure')],
    [Input('age-slider', 'value'),
     Input('diabetes-filter', 'value'),
     Input('bmi-filter', 'value')]
)
def update_distribution(age_range, diabetes_status, bmi_category):
    filtered_df = df[(df['age'] >= age_range[0]) & (df['age'] <= age_range[1])]
    
    if diabetes_status != 'all':
        filtered_df = filtered_df[filtered_df['type_diabete'] == diabetes_status]
    
    if bmi_category != 'all':
        filtered_df = filtered_df[filtered_df['bmi_category'] == bmi_category]
    
    total = len(filtered_df)
    diabetic = len(filtered_df[filtered_df['type_diabete'] == 1])
    avg_age = f"{filtered_df['age'].mean():.1f} years"
    avg_bmi = f"{filtered_df['bmi_scaled'].mean():.1f}"
    
    fig1 = px.histogram(
        filtered_df, x='age', color='diabetes_status',
        nbins=30, barmode='overlay',
        color_discrete_map={'Healthy': COLORS['healthy'], 'Diabetic': COLORS['diabetic']},
        opacity=0.7
    )
    fig1.update_layout(xaxis_title="Age", yaxis_title="Count", legend_title="Status", 
                       plot_bgcolor='white', height=300)
    
    fig2 = px.histogram(
        filtered_df, x='bmi_scaled', color='diabetes_status',
        nbins=30, barmode='overlay',
        color_discrete_map={'Healthy': COLORS['healthy'], 'Diabetic': COLORS['diabetic']},
        opacity=0.7
    )
    fig2.update_layout(xaxis_title="BMI", yaxis_title="Count", legend_title="Status", 
                       plot_bgcolor='white', height=300)
    
    age_group_data = filtered_df.groupby(['age_group', 'diabetes_status']).size().reset_index(name='count')
    fig3 = px.bar(
        age_group_data, x='age_group', y='count', color='diabetes_status',
        color_discrete_map={'Healthy': COLORS['healthy'], 'Diabetic': COLORS['diabetic']},
        barmode='stack'
    )
    fig3.update_layout(xaxis_title="Age Group", yaxis_title="Number of Patients", 
                       legend_title="Status", plot_bgcolor='white', height=300)
    
    bmi_counts = filtered_df['bmi_category'].value_counts().reset_index()
    bmi_counts.columns = ['category', 'count']
    fig4 = px.pie(bmi_counts, values='count', names='category',
                  color_discrete_sequence=px.colors.qualitative.Set3)
    fig4.update_layout(height=300)
    
    return total, diabetic, avg_age, avg_bmi, fig1, fig2, fig3, fig4

# ========== TEMPORAL PAGE CALLBACKS ==========

@callback(
    [Output('gaj-evolution-chart', 'figure'),
     Output('hba1c-evolution-chart', 'figure'),
     Output('combined-indicators-chart', 'figure')],
    [Input('diabetes-filter-p2', 'value'),
     Input('age-group-filter-p2', 'value'),
     Input('period-filter', 'value')]
)
def update_temporal(diabetes_status, age_group, period):
    filtered_df = df.copy()
    
    if diabetes_status != 'all':
        filtered_df = filtered_df[filtered_df['type_diabete'] == diabetes_status]
    
    if age_group != 'all':
        filtered_df = filtered_df[filtered_df['age_group'].astype(str) == age_group]
    
    if period == 'D':
        time_col = 'date'
    elif period == 'W':
        filtered_df['period'] = filtered_df['date'].dt.to_period('W').astype(str)
        time_col = 'period'
    else:
        time_col = 'month'
    
    gaj_data = filtered_df.groupby([time_col, 'diabetes_status'])['gaj'].mean().reset_index()
    fig1 = px.line(
        gaj_data, x=time_col, y='gaj', color='diabetes_status',
        color_discrete_map={'Healthy': COLORS['healthy'], 'Diabetic': COLORS['diabetic']},
        markers=True
    )
    fig1.update_layout(xaxis_title="Period", yaxis_title="Average Fasting Glucose (mg/dL)",
                       legend_title="Status", plot_bgcolor='white', height=350)
    
    hba1c_data = filtered_df.groupby([time_col, 'diabetes_status'])['hba1c'].mean().reset_index()
    fig2 = px.line(
        hba1c_data, x=time_col, y='hba1c', color='diabetes_status',
        color_discrete_map={'Healthy': COLORS['healthy'], 'Diabetic': COLORS['diabetic']},
        markers=True
    )
    fig2.update_layout(xaxis_title="Period", yaxis_title="Average HbA1c (%)",
                       legend_title="Status", plot_bgcolor='white', height=350)
    
    combined_data = filtered_df.groupby(time_col).agg({
        'gaj': 'mean', 'hba1c': 'mean', 'bmi': 'mean'
    }).reset_index()
    combined_data['bmi'] = combined_data['bmi'] * 10000
    
    fig3 = make_subplots(rows=1, cols=3, subplot_titles=('Average GAJ', 'Average HbA1c', 'Average BMI'))
    
    fig3.add_trace(go.Scatter(x=combined_data[time_col], y=combined_data['gaj'], 
                              mode='lines+markers', name='GAJ', line=dict(color=COLORS['primary'])),
                   row=1, col=1)
    fig3.add_trace(go.Scatter(x=combined_data[time_col], y=combined_data['hba1c'], 
                              mode='lines+markers', name='HbA1c', line=dict(color=COLORS['secondary'])),
                   row=1, col=2)
    fig3.add_trace(go.Scatter(x=combined_data[time_col], y=combined_data['bmi'], 
                              mode='lines+markers', name='BMI', line=dict(color=COLORS['danger'])),
                   row=1, col=3)
    
    fig3.update_xaxes(title_text="Period", row=1, col=1)
    fig3.update_xaxes(title_text="Period", row=1, col=2)
    fig3.update_xaxes(title_text="Period", row=1, col=3)
    fig3.update_layout(height=350, showlegend=False, plot_bgcolor='white')
    
    return fig1, fig2, fig3

# ========== CORRELATIONS PAGE CALLBACKS ==========

@callback(
    [Output('correlation-heatmap', 'figure'),
     Output('bmi-gaj-scatter', 'figure'),
     Output('age-hba1c-scatter', 'figure'),
     Output('clinical-boxplot', 'figure')],
    [Input('diabetes-filter-p3', 'value'),
     Input('age-group-filter-p3', 'value')]
)
def update_correlations(diabetes_status, age_group):
    filtered_df = df.copy()
    
    if diabetes_status != 'all':
        filtered_df = filtered_df[filtered_df['type_diabete'] == diabetes_status]
    
    if age_group != 'all':
        filtered_df = filtered_df[filtered_df['age_group'].astype(str) == age_group]
    
    corr_cols = ['age', 'taille', 'poids', 'bmi', 'gaj', 'hba1c', 'type_diabete']
    corr_matrix = filtered_df[corr_cols].corr()
    
    fig1 = go.Figure(data=go.Heatmap(
        z=corr_matrix.values, x=corr_cols, y=corr_cols,
        colorscale='RdBu', zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}', textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    fig1.update_layout(height=500, xaxis_title="Variables", yaxis_title="Variables")
    
    # BMI vs GAJ scatter
    fig2 = px.scatter(
        filtered_df, x='bmi_scaled', y='gaj', color='diabetes_status',
        color_discrete_map={'Healthy': COLORS['healthy'], 'Diabetic': COLORS['diabetic']},
        opacity=0.6
    )
    
    # Add manual trendlines for each group
    for status, color in [('Healthy', COLORS['healthy']), ('Diabetic', COLORS['diabetic'])]:
        df_status = filtered_df[filtered_df['diabetes_status'] == status]
        if len(df_status) > 1:
            z = np.polyfit(df_status['bmi_scaled'].dropna(), df_status['gaj'].dropna(), 1)
            p = np.poly1d(z)
            x_trend = np.linspace(df_status['bmi_scaled'].min(), df_status['bmi_scaled'].max(), 100)
            fig2.add_trace(go.Scatter(
                x=x_trend, y=p(x_trend),
                mode='lines', name=f'{status} trend',
                line=dict(color=color, dash='dash'),
                showlegend=False
            ))
    
    fig2.update_layout(xaxis_title="BMI", yaxis_title="Fasting Glucose (mg/dL)",
                       legend_title="Status", plot_bgcolor='white', height=350)
    
    # Age vs HbA1c scatter
    fig3 = px.scatter(
        filtered_df, x='age', y='hba1c', color='diabetes_status',
        color_discrete_map={'Healthy': COLORS['healthy'], 'Diabetic': COLORS['diabetic']},
        opacity=0.6
    )
    
    # Add manual trendlines for each group
    for status, color in [('Healthy', COLORS['healthy']), ('Diabetic', COLORS['diabetic'])]:
        df_status = filtered_df[filtered_df['diabetes_status'] == status]
        if len(df_status) > 1:
            z = np.polyfit(df_status['age'].dropna(), df_status['hba1c'].dropna(), 1)
            p = np.poly1d(z)
            x_trend = np.linspace(df_status['age'].min(), df_status['age'].max(), 100)
            fig3.add_trace(go.Scatter(
                x=x_trend, y=p(x_trend),
                mode='lines', name=f'{status} trend',
                line=dict(color=color, dash='dash'),
                showlegend=False
            ))
    
    fig3.update_layout(xaxis_title="Age (years)", yaxis_title="HbA1c (%)",
                       legend_title="Status", plot_bgcolor='white', height=350)
    
    melted_df = filtered_df.melt(
        id_vars=['diabetes_status'], value_vars=['gaj', 'hba1c'],
        var_name='Indicator', value_name='Value'
    )
    fig4 = px.box(melted_df, x='Indicator', y='Value', color='diabetes_status',
                  color_discrete_map={'Healthy': COLORS['healthy'], 'Diabetic': COLORS['diabetic']})
    fig4.update_layout(xaxis_title="Clinical Indicator", yaxis_title="Value",
                       legend_title="Status", plot_bgcolor='white', height=400)
    
    return fig1, fig2, fig3, fig4

# ========== MODELS PAGE CALLBACKS ==========

@callback(
    [Output('models-comparison-table', 'figure'),
     Output('accuracy-comparison', 'figure'),
     Output('f1-comparison', 'figure'),
     Output('roc-comparison', 'figure'),
     Output('all-metrics-radar', 'figure'),
     Output('performance-across-sets', 'figure')],
    [Input('url', 'pathname')]
)
def update_models(pathname):
    # Table
    test_data = models_comparison[models_comparison['Set'] == 'Test'].copy()
    
    fig1 = go.Figure(data=[go.Table(
        header=dict(values=list(test_data.columns),
                    fill_color='#2E86AB',
                    font=dict(color='white', size=12),
                    align='left'),
        cells=dict(values=[test_data[col] for col in test_data.columns],
                   fill_color='lavender',
                   align='left'))
    ])
    fig1.update_layout(height=250)
    
    # Accuracy comparison
    fig2 = px.bar(models_comparison[models_comparison['Set'] == 'Test'], 
                  x='Model', y='accuracy', color='Model',
                  color_discrete_sequence=px.colors.qualitative.Set2,
                  text='accuracy')
    fig2.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig2.update_layout(showlegend=False, yaxis_title="Accuracy", height=300)
    
    # F1 comparison
    fig3 = px.bar(models_comparison[models_comparison['Set'] == 'Test'], 
                  x='Model', y='f1', color='Model',
                  color_discrete_sequence=px.colors.qualitative.Pastel,
                  text='f1')
    fig3.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig3.update_layout(showlegend=False, yaxis_title="F1-Score", height=300)
    
    # ROC-AUC comparison
    fig4 = px.bar(models_comparison[models_comparison['Set'] == 'Test'], 
                  x='Model', y='roc_auc', color='Model',
                  color_discrete_sequence=px.colors.qualitative.Bold,
                  text='roc_auc')
    fig4.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig4.update_layout(showlegend=False, yaxis_title="ROC-AUC", height=300)
    
    # Radar chart
    test_metrics = models_comparison[models_comparison['Set'] == 'Test']
    fig5 = go.Figure()
    
    for idx, row in test_metrics.iterrows():
        fig5.add_trace(go.Scatterpolar(
            r=[row['accuracy'], row['precision'], row['recall'], row['f1'], row['roc_auc']],
            theta=['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC'],
            fill='toself',
            name=row['Model']
        ))
    
    fig5.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=300
    )
    
    # Performance across sets
    fig6 = px.line(models_comparison, x='Set', y='f1', color='Model', 
                   markers=True, line_shape='linear')
    fig6.update_layout(xaxis_title="Dataset", yaxis_title="F1-Score",
                       yaxis_range=[0.8, 1.0], height=350)
    
    return fig1, fig2, fig3, fig4, fig5, fig6

# ========== ANOMALY PAGE CALLBACKS ==========

@callback(
    [Output('iso-anomalies-count', 'children'),
     Output('ocsvm-anomalies-count', 'children'),
     Output('consensus-anomalies-count', 'children'),
     Output('anomaly-scatter', 'figure'),
     Output('iso-score-dist', 'figure'),
     Output('ocsvm-score-dist', 'figure')],
    [Input('url', 'pathname')]
)
def update_anomalies(pathname):
    iso_count = len(anomaly_results[anomaly_results['iso_anomaly'] == 1])
    ocsvm_count = len(anomaly_results[anomaly_results['ocsvm_anomaly'] == 1])
    consensus_count = len(anomaly_results[anomaly_results['consensus_anomaly'] == 1])
    
    # Scatter plot
    plot_df = df.copy()
    plot_df['anomaly_type'] = 'Normal'
    if 'iso_anomaly' in plot_df.columns:
        plot_df.loc[plot_df['iso_anomaly'] == 1, 'anomaly_type'] = 'Anomaly'
    
    fig1 = px.scatter(plot_df, x='bmi_scaled', y='gaj', color='anomaly_type',
                      color_discrete_map={'Normal': COLORS['primary'], 'Anomaly': COLORS['anomaly']},
                      opacity=0.6, hover_data=['age', 'hba1c'])
    fig1.update_layout(xaxis_title="BMI", yaxis_title="Fasting Glucose (mg/dL)",
                       legend_title="Type", height=400)
    
    # Iso score distribution
    fig2 = px.histogram(anomaly_results, x='iso_score', nbins=50, 
                        color_discrete_sequence=[COLORS['warning']])
    fig2.update_layout(xaxis_title="Isolation Forest Score", yaxis_title="Count", height=300)
    
    # OCSVM score distribution
    fig3 = px.histogram(anomaly_results, x='ocsvm_score', nbins=50,
                        color_discrete_sequence=[COLORS['danger']])
    fig3.update_layout(xaxis_title="One-Class SVM Score", yaxis_title="Count", height=300)
    
    return iso_count, ocsvm_count, consensus_count, fig1, fig2, fig3

# ==================== RUN APP ====================

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)


-- ========================================
-- Diabetes Data Warehouse - OLAP Queries
-- ========================================

-- Query 1: Diabetes prevalence by risk factors

SELECT 
    rf.sedentary_lifestyle,
    rf.family_history,
    rf.smoking_label,
    COUNT(*) as total_patients,
    SUM(f.diabetes_diagnosis) as diabetic_patients,
    ROUND(AVG(f.diabetes_diagnosis) * 100, 2) as diabetes_rate_percent
FROM fact_patient_measures f
JOIN dim_risk_factors rf ON f.risk_factor_id = rf.risk_factor_id
GROUP BY rf.sedentary_lifestyle, rf.family_history, rf.smoking_label
ORDER BY diabetes_rate_percent DESC


-- Query 2: Sedentary lifestyle impact by age group

SELECT 
    p.age_group,
    rf.sedentary_lifestyle,
    COUNT(*) as patient_count,
    SUM(f.diabetes_diagnosis) as diabetic_count,
    ROUND(AVG(f.diabetes_diagnosis) * 100, 2) as diabetes_rate_percent,
    ROUND(AVG(f.fasting_glucose), 2) as avg_fasting_glucose,
    ROUND(AVG(f.hba1c), 2) as avg_hba1c
FROM fact_patient_measures f
JOIN dim_patient p ON f.patient_id = p.patient_id
JOIN dim_risk_factors rf ON f.risk_factor_id = rf.risk_factor_id
GROUP BY p.age_group, rf.sedentary_lifestyle
ORDER BY p.age_group, rf.sedentary_lifestyle


-- Query 3: Combined risk profile analysis

SELECT 
    CASE 
        WHEN rf.sedentary_lifestyle = 1 AND rf.family_history = 1 
             AND rf.smoking_status > 0 AND p.bmi_category = 'Obese'
        THEN 'High Risk'
        WHEN (rf.sedentary_lifestyle = 1 OR rf.family_history = 1) 
             AND p.bmi_category IN ('Overweight', 'Obese')
        THEN 'Moderate Risk'
        ELSE 'Low Risk'
    END as risk_profile,
    COUNT(*) as patient_count,
    SUM(f.diabetes_diagnosis) as diabetic_count,
    ROUND(AVG(f.diabetes_diagnosis) * 100, 2) as diabetes_rate_percent,
    ROUND(AVG(f.fasting_glucose), 2) as avg_glucose,
    ROUND(AVG(f.hba1c), 2) as avg_hba1c
FROM fact_patient_measures f
JOIN dim_patient p ON f.patient_id = p.patient_id
JOIN dim_risk_factors rf ON f.risk_factor_id = rf.risk_factor_id
GROUP BY risk_profile
ORDER BY diabetes_rate_percent DESC


-- Query 4: Diet and physical activity impact

SELECT 
    rf.diet_label,
    rf.activity_label,
    COUNT(*) as patient_count,
    SUM(f.diabetes_diagnosis) as diabetic_count,
    ROUND(AVG(f.diabetes_diagnosis) * 100, 2) as diabetes_rate_percent,
    ROUND(AVG(p.bmi), 2) as avg_bmi
FROM fact_patient_measures f
JOIN dim_patient p ON f.patient_id = p.patient_id
JOIN dim_risk_factors rf ON f.risk_factor_id = rf.risk_factor_id
GROUP BY rf.diet_label, rf.activity_label
ORDER BY diabetes_rate_percent DESC


-- Query 5: Temporal quarterly trends

SELECT 
    d.year,
    d.quarter,
    COUNT(*) as total_measures,
    SUM(f.diabetes_diagnosis) as diabetic_count,
    ROUND(AVG(f.diabetes_diagnosis) * 100, 2) as diabetes_rate_percent,
    ROUND(AVG(f.fasting_glucose), 2) as avg_glucose,
    ROUND(AVG(f.hba1c), 2) as avg_hba1c
FROM fact_patient_measures f
JOIN dim_date d ON f.date_id = d.date_id
GROUP BY d.year, d.quarter
ORDER BY d.year, d.quarter


-- Query 6: Family history correlation

SELECT 
    rf.family_history,
    p.age_group,
    p.bmi_category,
    COUNT(*) as patient_count,
    SUM(f.diabetes_diagnosis) as diabetic_count,
    ROUND(AVG(f.diabetes_diagnosis) * 100, 2) as diabetes_rate_percent
FROM fact_patient_measures f
JOIN dim_patient p ON f.patient_id = p.patient_id
JOIN dim_risk_factors rf ON f.risk_factor_id = rf.risk_factor_id
GROUP BY rf.family_history, p.age_group, p.bmi_category
HAVING patient_count >= 3
ORDER BY diabetes_rate_percent DESC


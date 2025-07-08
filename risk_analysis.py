from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import joblib

# Load the improved models with selected features
kmeans_selected = joblib.load("models/cervical_cancer_kmeans_selected_model.pkl")
selector = joblib.load("pipelines/cervical_cancer_feature_selector.pkl")
full_risk_pipeline = joblib.load("pipelines/cervical_cancer_risk_pipeline.pkl")

# Load selected feature names
with open('selected_feature_names.pkl', 'rb') as f:
    selected_feature_names = pickle.load(f)

app = FastAPI()

class InferenceInput(BaseModel):
    age: float
    number_of_sexual_partners: int
    first_sexual_intercourse: int
    smoking_status: str 
    stds_history: str
    hpv_test_result: str 

# Define risk categories based on probability thresholds
def get_risk_category(risk_probability, hpv_positive):
    """Determine risk category based on probability and HPV status"""
    if hpv_positive == 1:
        return "High Risk", "HPV Positive - Immediate screening required"
    elif risk_probability < 0.3:
        return "Low Risk", "Minimal risk factors - Regular screening recommended"
    elif risk_probability < 0.6:
        return "Moderate Risk", "Some risk factors present - Consider more frequent screening"
    else:
        return "High Risk", "Multiple risk factors - Immediate screening recommended"

def get_detailed_recommendation(risk_probability, hpv_positive, age):
    """Get detailed recommendation based on risk probability"""
    if hpv_positive == 1:
        return {
            "immediate_action": "HPV DNA testing",
            "follow_up": "HPV-positive patients require specialized care",
            "screening_frequency": "As recommended by healthcare provider"
        }
    elif risk_probability < 0.3:
        if age < 25:
            return {
                "immediate_action": "Continue regular preventive care",
                "follow_up": "Begin cervical cancer screening at age 25",
                "screening_frequency": "Every 3 years with Pap smear"
            }
        else:
            return {
                "immediate_action": "Schedule routine Pap smear",
                "follow_up": "Maintain regular screening schedule",
                "screening_frequency": "Every 3 years with Pap smear or every 5 years with HPV co-testing"
            }
    elif risk_probability < 0.6:
        return {
            "immediate_action": "Schedule Pap smear and HPV testing",
            "follow_up": "Discuss risk factors with healthcare provider",
            "screening_frequency": "Consider annual screening or as recommended"
        }
    else:
        return {
            "immediate_action": "Schedule immediate screening - Pap smear and HPV testing",
            "follow_up": "Discuss comprehensive risk management with healthcare provider",
            "screening_frequency": "Annual screening recommended"
        }

@app.post("/risk_predict")
def risk_prediction(data: InferenceInput):
    smoking_status_map = {
        "No": 0, "N": 0,
        "Yes": 1, "Y": 1,
    }
    stds_map = {
        "No": 0, "N": 0,
        "Yes": 1, "Y": 1,
    }
    hpv_map = {
        "Negative": 0, "N": 0,
        "Positive": 1, "Y": 1,
    }
    
    smoking_status_num = smoking_status_map.get(data.smoking_status.strip().upper(), 0)
    stds_history_num = stds_map.get(data.stds_history.strip().upper(), 0)
    hpv_results_num = hpv_map.get(data.hpv_test_result.strip().upper(), 0)

    # feature engineered features
    years_sexually_active = data.age - data.first_sexual_intercourse
    smokes_and_has_stds = smoking_status_num * stds_history_num
    sexual_partner_and_years_active = data.number_of_sexual_partners * years_sexually_active
    log_sexual_partners = np.log1p(data.number_of_sexual_partners)
    years_sexually_active_squared = years_sexually_active ** 2
    
    # HPV engineered features
    hpv_and_stds = hpv_results_num * stds_history_num
    hpv_and_smoking = hpv_results_num * smoking_status_num
    high_risk_score = 1 if (data.number_of_sexual_partners / 5 + years_sexually_active / 30 + smoking_status_num + stds_history_num) > 2.5 else 0
    early_sexual_activity = 1 if data.first_sexual_intercourse < 18 else 0

    # Risk score
    risk_score = (
        data.number_of_sexual_partners / 5 +
        (years_sexually_active / 30) +
        smoking_status_num +
        stds_history_num
    )

    # input
    input_data = pd.DataFrame({
        'Age': [data.age],
        'Sexual Partners': [data.number_of_sexual_partners],
        'First Sexual Activity Age': [data.first_sexual_intercourse],
        'Smoking Status': [data.smoking_status],
        'STDs History': [data.stds_history],
        'HPV Test Result': [data.hpv_test_result],
        'Years_Sexually_Active': [years_sexually_active],
        'Smoking_Status_Num': [smoking_status_num],
        'STDs_History_Num': [stds_history_num],
        'HPV_Test_Result_Num': [hpv_results_num],
        'Risk_Score': [risk_score],
        'Smokes_and_Has_STDs': [smokes_and_has_stds],
        'Sexual_Partner_and_Years_Active': [sexual_partner_and_years_active],
        'Log_Sexual_Partners': [log_sexual_partners],
        'Years_Sexually_Active_Squared': [years_sexually_active_squared],
        'HPV_and_STDs': [hpv_and_stds],
        'HPV_and_Smoking': [hpv_and_smoking],
        'High_Risk_Score': [high_risk_score],
        'Early_Sexual_Activity': [early_sexual_activity]
    })
    
    # age group
    if data.age < 25:
        age_group = '<25'
    elif 25 <= data.age < 36:
        age_group = '25-35'
    elif 36 <= data.age <= 50:
        age_group = '36-50'
    else:
        age_group = '50+'
    
    input_data['Age_Group'] = [age_group]

    try:
        X_transformed = full_risk_pipeline.transform(input_data)
        
        X_selected = selector.transform(X_transformed)
        
        cluster = int(kmeans_selected.predict(X_selected)[0])
        
        cluster_centers = kmeans_selected.cluster_centers_
        distances = np.linalg.norm(X_selected - cluster_centers, axis=1)
        risk_probability = float(distances[cluster] / np.max(distances))
        
        risk_category, risk_description = get_risk_category(risk_probability, hpv_results_num)
        detailed_recommendation = get_detailed_recommendation(risk_probability, hpv_results_num, data.age)
        
        return {
            "risk_probability": risk_probability,
            "risk_category": risk_category,
            "risk_description": risk_description,
            "recommendations": detailed_recommendation,
            "risk_factors": {
                "hpv_positive": hpv_results_num == 1,
                "smoking": smoking_status_num == 1,
                "stds_history": stds_history_num == 1,
                "early_sexual_activity": early_sexual_activity == 1,
                "multiple_partners": data.number_of_sexual_partners > 2,
                "age_group": age_group
            },
            "selected_features": selected_feature_names
        }
        
    except Exception as e:
        return {
            "error": f"Prediction failed: {str(e)}",
            "risk_probability": 0.0,
            "risk_category": "Error",
            "risk_description": "Unable to assess risk"
        }

@app.get("/model_info")
def get_model_info():
    return {
        "model_type": "K-Means Clustering with Feature Selection",
        "n_clusters": 2,
        "selected_features": selected_feature_names,
        "feature_selection_method": "SelectKBest with f_classif",
        "includes_hpv": True,
        "risk_thresholds": {
            "low_risk": "< 0.3",
            "moderate_risk": "0.3 - 0.6",
            "high_risk": "> 0.6"
        }
    }


@app.get("/risk_thresholds")
def get_risk_thresholds():
    return {
        "low_risk": {
            "threshold": "< 0.3",
            "description": "Minimal risk factors present",
            "action": "Regular screening recommended"
        },
        "moderate_risk": {
            "threshold": "0.3 - 0.6",
            "description": "Some risk factors present",
            "action": "Consider more frequent screening, possibly annual Pap smear and HPV testing"
        },
        "high_risk": {
            "threshold": "> 0.6",
            "description": "Multiple risk factors present",
            "action": "Immediate screening recommended, that is Pap smear and HPV DNA testing"
        },
        "hpv_positive": {
            "threshold": "Any probability",
            "description": "HPV positive status",
            "action": "Immediate specialized care required, including HPV DNA testing and follow-up with healthcare provider"
        }
    }
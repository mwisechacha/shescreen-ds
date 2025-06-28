from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import warnings
import joblib
import sklearn

warnings.filterwarnings("ignore", category=FutureWarning)
print(f"Current scikit-learn version: {sklearn.__version__}")

# Load model
try:
    model = joblib.load("models/cervical_cancer_rf_model.pkl")
    pipeline = joblib.load("pipelines/cervical_cancer_full_pipeline.pkl")
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    model = None
    pipeline = None

app = FastAPI()

class InferenceInput(BaseModel):
    age: float
    number_of_sexual_partners: int
    first_sexual_intercourse: int
    smoking_status: str 
    stds_history: str
    hpv_results: str
    pap_smear_results: str
    screening_type_last: str

@app.post("/recommendation_predict")
def recommendation_prediction(data: InferenceInput):
   if model is None or pipeline is None:
        return {"error": "Models not properly loaded due to version mismatch"}
    
   try:
        smoking_status_map = {"No": 0, "Yes": 1, "N": 0, "Y": 1}
        stds_map = {"No": 0, "Yes": 1, "N": 0, "Y": 1}
        pap_smear_map = {'N': 0, 'Y': 1, 'NEGATIVE': 0, 'POSITIVE': 1}
        hpv_map = {'NEGATIVE': 0, 'POSITIVE': 1, 'N': 0, 'Y': 1}
            
        smoking_status_num = smoking_status_map.get(data.smoking_status.strip().capitalize(), 0)
        stds_history_num = stds_map.get(data.stds_history.strip().capitalize(), 0)
        hpv_results_num = hpv_map.get(data.hpv_results.strip().upper(), 0)
        pap_smear_results_num = pap_smear_map.get(data.pap_smear_results.strip().upper(), 0)

        years_sexually_active = data.age - data.first_sexual_intercourse
        smokes_and_has_stds = smoking_status_num * stds_history_num
        sexual_partner_and_years_active = data.number_of_sexual_partners * years_sexually_active
        log_sexual_partners = np.log1p(data.number_of_sexual_partners)
        years_sexually_active_squared = years_sexually_active ** 2

        risk_score = (
            data.number_of_sexual_partners / 5 +
            (years_sexually_active / 30) +
            smoking_status_num +
            stds_history_num
        )

        if data.age < 25:
            age_group = '<25'
        elif data.age < 35:
            age_group = '25-35'
        elif data.age < 50:
            age_group = '36-50'
        else:
            age_group = '50+'

        feature_data = {
            'Smoking_Status_Num': [smoking_status_num],
            'STDs_History_Num': [stds_history_num], 
            'Years_Sexually_Active': [years_sexually_active],
            'Smokes_and_Has_STDs': [smokes_and_has_stds],
            'Sexual_Partner_and_Years_Active': [sexual_partner_and_years_active],
            'Risk_Score': [risk_score],
            'Log_Sexual_Partners': [log_sexual_partners],
            'Years_Sexually_Active_Squared': [years_sexually_active_squared],
            'Pap_Smear_Result_Num': [pap_smear_results_num],
            'HPV_Test_Result_Num': [hpv_results_num],
            'Age_Group': [age_group],
            'Screening Type Last': [data.screening_type_last.strip().upper()]
        }

        X_df = pd.DataFrame(feature_data)
        X_transformed = pipeline.transform(X_df)
        prediction_proba = model.predict_proba(X_transformed)
        prediction_label = int(np.argmax(prediction_proba[0]))
        
        recommendations = {
            0: "Screening",
            1: "Follow up",
            2: "Diagnostic evaluation and treatment"
        }

        recommendation = recommendations.get(prediction_label, "Unknown")
        return {
            "recommendation": recommendation,
            "prediction_proba": prediction_proba[0].tolist(),
            "risk_score": float(risk_score)
        }
   
   except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import joblib

# load model
model = joblib.load("models/cervical_cancer_kmeans_model.pkl")
imputer = joblib.load("pipelines/cervical_cancer_imputer.pkl")
minmax_scaler = joblib.load("pipelines/cervical_cancer_minmax_scaler.pkl")
std_scaler = joblib.load("pipelines/cervical_cancer_std_scaler.pkl")

app = FastAPI()

class InferenceInput(BaseModel):
    age: float
    number_of_sexual_partners: int
    first_sexual_intercourse: int
    smoking_status: str 
    stds_history: str 

# cluster interpretation
CLUSTER_INFO = {
    0: "Older, high sexual exposure, high STD risk",
    1: "Younger, moderate exposure, low risk",
    2: "Middle-aged, moderate exposure, high behavioral risk",
    3: "Middle-aged, moderate exposure, low risk",
    4: "Young, low exposure, some behavioral risk",
    5: "Older, long exposure, smokes, moderate risk"
}



@app.post("/risk_predict")
def risk_prediction(data: InferenceInput):
    smoking_status_map = {
        "No": 0,
        "Yes": 1,
    }
    stds_map = {
        "No": 0,
        "Yes": 1,
    }
    
    smoking_status_num = smoking_status_map.get(data.smoking_status.strip().capitalize(), 0)
    stds_history_num = stds_map.get(data.stds_history.strip().capitalize(), 0)

    years_sexually_active = data.age - data.first_sexual_intercourse
    smokes_and_has_stds = smoking_status_num * stds_history_num
    sexual_partner_and_years_active = data.number_of_sexual_partners * years_sexually_active
    log_sexual_partners = np.log1p(data.number_of_sexual_partners)
    years_sexually_active_squared = years_sexually_active ** 2

    # Age group one-hot encoding
    age_group_25_35 = 0
    age_group_36_50 = 0
    age_group_50_plus = 0
    age_group_lt25 = 0
    if data.age < 25:
        age_group_lt25 = 1
    elif 25 <= data.age < 36:
        age_group_lt25 = 0
    elif 36 <= data.age <= 50:
        age_group_36_50 = 1
    elif data.age > 50:
        age_group_50_plus = 1

    # Risk score
    risk_score = (
        data.number_of_sexual_partners / 5 +
        (years_sexually_active / 30) +
        smoking_status_num +
        stds_history_num
    )

    risk_score = float(minmax_scaler.transform(np.array([[risk_score]]))[0][0])

    # Prepare input features
    X = np.array([[
        smoking_status_num,        
        stds_history_num, 
        risk_score,
        smokes_and_has_stds,
        sexual_partner_and_years_active,
        log_sexual_partners,
        years_sexually_active_squared,
        age_group_25_35,
        age_group_36_50,
        age_group_50_plus,
        age_group_lt25
    ]])

    X_scaled = std_scaler.transform(X)

    X_imputed = imputer.transform(X_scaled)

    cluster = int(model.predict(X_imputed)[0])
    interpretation = CLUSTER_INFO.get(cluster, "Unknown cluster")
    return {
        "cluster": cluster,
        "interpretation": interpretation
    }
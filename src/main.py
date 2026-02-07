from fastapi import FastAPI
from pydantic import BaseModel
from src.core.model import ChurnModel

app = FastAPI(title="Customer Churn Predictor API")
churn_predictor = ChurnModel()

class CustomerData(BaseModel):
    tenure: int
    monthly_charges: float
    total_charges: float
    support_calls: int
    contract_type: int # 0: Month-to-month, 1: One year, 2: Two year

@app.post("/predict")
async def predict_churn(data: CustomerData):
    # Mapping Pydantic to prediction logic
    features = data.dict()
    probability = churn_predictor.predict(features)
    
    return {
        "churn_probability": probability,
        "risk_level": "High" if probability > 0.7 else "Medium" if probability > 0.3 else "Low",
        "recommendation": "Proactive outreach" if probability > 0.5 else "Standard monitoring"
    }

@app.get("/health")
async def health():
    return {"status": "ok", "model": "RandomForest_v1"}

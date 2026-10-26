from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd


# Load trained model
model = joblib.load("models/crop_recommendation_model.pkl")

app = FastAPI(
    title="Smart Agriculture Crop Recommendation API",
    description="Predicts the most suitable crop based on farm conditions.",
    version="1.0.0"
)


class FarmData(BaseModel):
    nitrogen: float
    phosphorus: float
    potassium: float
    temperature: float
    humidity: float
    soil_ph: float
    rainfall: float
    soil_moisture: float


@app.get("/")
def home():
    return {
        "message": "Smart Agriculture API is running",
        "model": "Random Forest Crop Recommendation"
    }


@app.post("/predict")
def predict_crop(data: FarmData):

    input_data = pd.DataFrame([
        {
            "nitrogen": data.nitrogen,
            "phosphorus": data.phosphorus,
            "potassium": data.potassium,
            "temperature": data.temperature,
            "humidity": data.humidity,
            "soil_ph": data.soil_ph,
            "rainfall": data.rainfall,
            "soil_moisture": data.soil_moisture
        }
    ])

    prediction = model.predict(input_data)

    return {
        "recommended_crop": prediction[0]
    }
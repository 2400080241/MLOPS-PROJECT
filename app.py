
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field


# =========================================================
# 1. PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
TEMPLATE_DIR = BASE_DIR / "templates"


def load_model(filename):
    """Load a trained model from the models directory."""
    path = MODEL_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Missing model file: {path}"
        )

    return joblib.load(path)


# Load all four trained models.
crop_model = load_model("crop_recommendation_model.pkl")
yield_model = load_model("crop_yield_model.pkl")
irrigation_model = load_model("irrigation_model.pkl")
soil_model = load_model("soil_health_model.pkl")


# =========================================================
# 2. FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="Smart Agriculture Assistant",
    description=(
        "Crop recommendation, crop yield prediction, "
        "irrigation advice and soil health prediction."
    ),
    version="2.1.0"
)

templates = Jinja2Templates(
    directory=str(TEMPLATE_DIR)
)


# =========================================================
# 3. INPUT VALIDATION
# =========================================================

class FarmData(BaseModel):
    nitrogen: float = Field(ge=0, le=300)
    phosphorus: float = Field(ge=0, le=300)
    potassium: float = Field(ge=0, le=300)

    temperature: float = Field(ge=-10, le=70)
    humidity: float = Field(ge=0, le=100)
    soil_ph: float = Field(ge=0, le=14)
    rainfall: float = Field(ge=0, le=1000)
    soil_moisture: float = Field(ge=0, le=100)


CROP_FEATURES = [
    "nitrogen",
    "phosphorus",
    "potassium",
    "temperature",
    "humidity",
    "soil_ph",
    "rainfall",
    "soil_moisture"
]

OTHER_FEATURES = [
    "temperature",
    "humidity",
    "soil_moisture",
    "rainfall",
    "soil_ph",
    "nitrogen",
    "phosphorus",
    "potassium"
]


# =========================================================
# 4. PREDICTION HELPERS
# =========================================================

def predict_one(model, values, features):
    """Create input in the exact feature order expected by a model."""
    input_df = pd.DataFrame(
        [{name: values[name] for name in features}],
        columns=features
    )

    prediction = model.predict(input_df)[0]

    return prediction, input_df


def get_confidence(model, input_df):
    """Return class probability when supported by the model."""
    if not hasattr(model, "predict_proba"):
        return None

    probabilities = model.predict_proba(input_df)[0]

    return round(
        float(max(probabilities)) * 100,
        1
    )


# These interpretations are provisional.
# Confirm the original training labels before relying on them.
SOIL_LABELS = {
    0: {
        "status": "Poor",
        "message": (
            "The model places your soil in its lowest "
            "soil-health category. Consider a soil test."
        )
    },
    1: {
        "status": "Moderate",
        "message": (
            "The model places your soil in its middle "
            "soil-health category. Check nutrient levels."
        )
    },
    2: {
        "status": "Good",
        "message": (
            "The model places your soil in its highest "
            "soil-health category. Continue monitoring it."
        )
    }
}

IRRIGATION_LABELS = {
    0: {
        "status": "Water may not be needed",
        "message": (
            "This prediction corresponds to irrigation class 0. "
            "Check actual soil moisture before deciding."
        )
    },
    1: {
        "status": "Water may be needed",
        "message": (
            "This prediction corresponds to irrigation class 1. "
            "Check actual soil moisture before watering."
        )
    }
}


# =========================================================
# 5. WEBSITE HOME PAGE
# =========================================================

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    page = TEMPLATE_DIR / "index.html"

    if not page.exists():
        return HTMLResponse(
            """
            <h1>Smart Agriculture Assistant API is running</h1>
            <p>Create templates/index.html for the website.</p>
            <p><a href="/docs">Open API documentation</a></p>
            """
        )

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


# =========================================================
# 6. ANALYZE ALL FARM CONDITIONS
# =========================================================

@app.post("/analyze")
def analyze_farm(data: FarmData):
    values = data.model_dump()

    try:
        # Crop recommendation
        crop_prediction, crop_input = predict_one(
            crop_model,
            values,
            CROP_FEATURES
        )

        crop_confidence = get_confidence(
            crop_model,
            crop_input
        )

        # Crop yield prediction
        yield_prediction, _ = predict_one(
            yield_model,
            values,
            OTHER_FEATURES
        )

        # Irrigation prediction
        irrigation_prediction, irrigation_input = predict_one(
            irrigation_model,
            values,
            OTHER_FEATURES
        )

        irrigation_confidence = get_confidence(
            irrigation_model,
            irrigation_input
        )

        # Soil health prediction
        soil_prediction, soil_input = predict_one(
            soil_model,
            values,
            OTHER_FEATURES
        )

        soil_confidence = get_confidence(
            soil_model,
            soil_input
        )

        # Convert NumPy integer values to ordinary Python integers.
        irrigation_class = int(irrigation_prediction)
        soil_class = int(soil_prediction)

        irrigation_result = IRRIGATION_LABELS.get(
            irrigation_class,
            {
                "status": "Unknown",
                "message": "The irrigation class is not recognised."
            }
        )

        soil_result = SOIL_LABELS.get(
            soil_class,
            {
                "status": "Unknown",
                "message": "The soil-health class is not recognised."
            }
        )

        return {
            "crop": {
                "name": str(crop_prediction),
                "confidence_percent": crop_confidence
            },

            "yield": {
                "value": round(float(yield_prediction), 2),
                "unit": "unverified",
                "message": (
                    "This is the model's predicted yield value. "
                    "The training dataset does not document its unit."
                )
            },

            "irrigation": {
                "class": irrigation_class,
                "status": irrigation_result["status"],
                "message": irrigation_result["message"],
                "confidence_percent": irrigation_confidence
            },

            "soil_health": {
                "class": soil_class,
                "status": soil_result["status"],
                "message": soil_result["message"],
                "confidence_percent": soil_confidence
            },

            "notice": (
                "Predictions are estimates. Check real field "
                "conditions before making farming decisions. "
                "Soil and irrigation class meanings are provisional, "
                "and the crop-yield unit is unverified."
            )
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Farm analysis failed: {str(error)}"
        )


# =========================================================
# 7. CROP-ONLY ENDPOINT
# =========================================================

@app.post("/predict")
def predict_crop(data: FarmData):
    values = data.model_dump()

    try:
        prediction, _ = predict_one(
            crop_model,
            values,
            CROP_FEATURES
        )

        return {
            "recommended_crop": str(prediction)
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Crop prediction failed: {str(error)}"
        )


# =========================================================
# 8. HEALTH CHECK
# =========================================================

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "models_loaded": {
            "crop_recommendation": True,
            "crop_yield": True,
            "irrigation": True,
            "soil_health": True
        }
    }

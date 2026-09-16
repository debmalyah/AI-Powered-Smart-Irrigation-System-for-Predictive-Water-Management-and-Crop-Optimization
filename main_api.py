import joblib
import pandas as pd
import psycopg2
import json
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from data_eda import engineer_features, TARGET_ORDER
from scheduler_engine import generate_irrigation_schedule

app = FastAPI(
    title="Smart Irrigation Prediction & Scheduling API",
    version="1.0"
)

MODEL_PATH = "rf_irrigation_model.joblib"

# Update credentials to match your local PostgreSQL setup
DB_CONFIG = {
    "dbname": "smart_irrigation",
    "user": "postgres",
    "password": "debmalya",
    "host": "localhost",
    "port": 5432
}


def init_postgres_db():
    """Automatically creates the schedule_logs table if it does not exist."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS schedule_logs (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                crop_type VARCHAR(50),
                growth_stage VARCHAR(50),
                soil_moisture REAL,
                predicted_urgency VARCHAR(50),
                water_amount_mm REAL,
                duration_minutes INT,
                recommended_slot VARCHAR(100),
                full_response TEXT
            );
        """)
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"PostgreSQL initialization error: {e}")


@app.on_event("startup")
def startup_event():
    init_postgres_db()


def save_schedule_to_postgres(input_data: dict, prediction_need: str, schedule: dict):
    """Persists prediction and schedule results to PostgreSQL."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        insert_query = """
            INSERT INTO schedule_logs (
                timestamp, crop_type, growth_stage, soil_moisture,
                predicted_urgency, water_amount_mm, duration_minutes,
                recommended_slot, full_response
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        cursor.execute(insert_query, (
            datetime.now().isoformat(),
            input_data.get("Crop_Type"),
            input_data.get("Growth_Stage"),
            input_data.get("Soil_Moisture"),
            prediction_need,
            schedule.get("water_amount_mm"),
            schedule.get("duration_minutes"),
            schedule.get("recommended_slot"),
            json.dumps({"prediction": prediction_need, "schedule": schedule})
        ))

        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"PostgreSQL logging error: {e}")


try:
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    scaler = artifact["scaler"]
    feature_columns = artifact["feature_columns"]
except Exception:
    model = None


class FieldDataInput(BaseModel):
    Soil_Moisture: float = Field(..., example=32.5)
    Temperature_C: float = Field(..., example=28.4)
    Humidity: float = Field(..., example=65.0)
    Wind_Speed_kmh: float = Field(..., example=12.0)
    Rainfall_mm: float = Field(..., example=0.0)
    Previous_Irrigation_mm: float = Field(..., example=15.0)
    Crop_Type: str = Field(..., example="Wheat")
    Growth_Stage: str = Field(..., example="Vegetative")
    Forecast_Rain_mm: float = Field(0.0, example=2.0)


@app.get("/")
def health_check():
    return {"status": "Online", "model_loaded": model is not None}


@app.post("/predict_and_schedule")
def predict_and_schedule(data: FieldDataInput):
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model artifact missing. Please run 'train_rf.py' or 'train_gb.py' first."
        )

    input_dict = data.model_dump()
    df_input = pd.DataFrame([input_dict])

    df_fe = engineer_features(df_input, soil_moisture_max=100.0, rainfall_max=100.0)
    categorical_cols = df_fe.select_dtypes(include=["object", "string"]).columns
    df_encoded = pd.get_dummies(df_fe, columns=categorical_cols, drop_first=True)
    df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)

    scaled_input = scaler.transform(df_encoded)
    pred_idx = model.predict(scaled_input)[0]
    predicted_need = TARGET_ORDER[pred_idx]

    schedule = generate_irrigation_schedule(
        prediction=predicted_need,
        crop_type=data.Crop_Type,
        growth_stage=data.Growth_Stage,
        soil_moisture=data.Soil_Moisture,
        forecast_rain_mm=data.Forecast_Rain_mm
    )

    save_schedule_to_postgres(input_dict, predicted_need, schedule)

    return {
        "prediction": {
            "irrigation_need": predicted_need,
            "class_index": int(pred_idx)
        },
        "schedule_recommendation": schedule
    }
import os
import joblib
import pandas as pd
import psycopg2
import json
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from data_eda import engineer_features, TARGET_ORDER
from scheduler_engine import generate_irrigation_schedule

# Points at whichever model won the baseline/RF/GB comparison -- run
# select_best_model.py after (re)training to (re)generate this file.
MODEL_PATH = "best_irrigation_model.joblib"

# Read from environment instead of hardcoding -- never commit real credentials
# to source control. Set these in a .env file (loaded by your shell / process
# manager) or your deployment platform's secrets config, e.g.:
#   export IRRIGATION_DB_PASSWORD="your-actual-password"
DB_CONFIG = {
    "dbname": os.environ.get("IRRIGATION_DB_NAME", "smart_irrigation"),
    "user": os.environ.get("IRRIGATION_DB_USER", "postgres"),
    "password": os.environ.get("IRRIGATION_DB_PASSWORD"),
    "host": os.environ.get("IRRIGATION_DB_HOST", "localhost"),
    "port": int(os.environ.get("IRRIGATION_DB_PORT", 5432)),
}
if not DB_CONFIG["password"]:
    print(
        "WARNING: IRRIGATION_DB_PASSWORD is not set. Postgres logging "
        "(schedule_logs) will fail until it's set in the environment; "
        "predictions themselves are unaffected."
    )


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


# Replaces the deprecated @app.on_event("startup") decorator. FastAPI now
# wants a lifespan context manager instead: code before `yield` runs on
# startup, code after `yield` (none needed here) would run on shutdown.
# Behavior is unchanged -- this still just ensures schedule_logs exists
# before the app starts serving requests.
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_postgres_db()
    yield


app = FastAPI(
    title="Smart Irrigation Prediction & Scheduling API",
    version="1.0",
    lifespan=lifespan,
)


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
            input_data.get("Crop_Growth_Stage"),
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
    # Falls back to 100.0 only for artifacts saved before this field existed;
    # retrain to get the real training-time values baked in.
    soil_moisture_max = artifact.get("soil_moisture_max", 100.0)
    rainfall_max = artifact.get("rainfall_max", 100.0)
except Exception:
    model = None


class FieldDataInput(BaseModel):
    # Mirrors every raw column the model was trained on (irrigation_prediction.csv),
    # minus the target. Keeping this in lockstep with the training columns is what
    # avoids the reindex(fill_value=0) silently fabricating zeroed-out feature values
    # for anything the API doesn't collect.
    #
    # Pydantic v2 deprecated the bare `example=` kwarg on Field() in favor of
    # json_schema_extra -- same Swagger "Try it out" example values, just
    # passed the way v2 (and the upcoming v3) actually wants them.
    Soil_Type: str = Field(..., json_schema_extra={"example": "Loamy"})
    Soil_pH: float = Field(..., json_schema_extra={"example": 6.5})
    Soil_Moisture: float = Field(..., json_schema_extra={"example": 32.5})
    Organic_Carbon: float = Field(..., json_schema_extra={"example": 0.9})
    Electrical_Conductivity: float = Field(..., json_schema_extra={"example": 1.8})
    Temperature_C: float = Field(..., json_schema_extra={"example": 28.4})
    Humidity: float = Field(..., json_schema_extra={"example": 65.0})
    Rainfall_mm: float = Field(..., json_schema_extra={"example": 0.0})
    Sunlight_Hours: float = Field(..., json_schema_extra={"example": 7.5})
    Wind_Speed_kmh: float = Field(..., json_schema_extra={"example": 12.0})
    Crop_Type: str = Field(..., json_schema_extra={"example": "Wheat"})
    Crop_Growth_Stage: str = Field(..., json_schema_extra={"example": "Vegetative"})
    Season: str = Field(..., json_schema_extra={"example": "Kharif"})
    Irrigation_Type: str = Field(..., json_schema_extra={"example": "Drip"})
    Water_Source: str = Field(..., json_schema_extra={"example": "Groundwater"})
    Field_Area_hectare: float = Field(..., json_schema_extra={"example": 7.5})
    Mulching_Used: str = Field(..., json_schema_extra={"example": "Yes"})
    Previous_Irrigation_mm: float = Field(..., json_schema_extra={"example": 15.0})
    Region: str = Field(..., json_schema_extra={"example": "North"})
    # Not a training feature -- used only by the scheduler's rain-offset
    # safety check, not fed into the ML model.
    Forecast_Rain_mm: float = Field(0.0, json_schema_extra={"example": 2.0})


@app.get("/")
def health_check():
    return {"status": "Online", "model_loaded": model is not None}


@app.post("/predict_and_schedule")
def predict_and_schedule(data: FieldDataInput):
    if model is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "Model artifact missing. Run 'train_baseline.py', 'train_rf.py' and "
                "'train_gb.py', then 'select_best_model.py' to produce "
                "'best_irrigation_model.joblib'."
            )
        )

    input_dict = data.model_dump()
    df_input = pd.DataFrame([input_dict])

    df_fe = engineer_features(df_input, soil_moisture_max=soil_moisture_max, rainfall_max=rainfall_max)
    categorical_cols = df_fe.select_dtypes(include=["object", "string"]).columns
    df_encoded = pd.get_dummies(df_fe, columns=categorical_cols, drop_first=True)
    df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)

    scaled_input = scaler.transform(df_encoded)
    pred_idx = model.predict(scaled_input)[0]
    predicted_need = TARGET_ORDER[pred_idx]

    schedule = generate_irrigation_schedule(
        prediction=predicted_need,
        crop_type=data.Crop_Type,
        growth_stage=data.Crop_Growth_Stage,
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
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import psycopg2

app = FastAPI(title="Smart Irrigation Data Ingestion API")


# Connect to your local PostgreSQL database
def get_db_connection():
    return psycopg2.connect(
        host="localhost",
        database="smart_irrigation",
        user="postgres",  # UPDATE with your PostgreSQL username
        password="debmalya",  # UPDATE with your PostgreSQL password
        port="5432"
    )


# Pydantic schema for telemetry validation
class TelemetryPayload(BaseModel):
    sensor_id: int
    field_id: int
    soil_moisture: float = Field(..., ge=0.0, le=100.0)
    temperature_c: float = Field(..., ge=-10.0, le=60.0)
    humidity: float = Field(..., ge=0.0, le=100.0)
    rainfall_mm: float = Field(..., ge=0.0)


@app.post("/api/ingest")
def ingest_telemetry(payload: TelemetryPayload):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # 1. Store sensor moisture reading
        cursor.execute(
            "INSERT INTO sensor_readings (sensor_id, soil_moisture_value) VALUES (%s, %s);",
            (payload.sensor_id, payload.soil_moisture)
        )

        # 2. Store weather reading
        cursor.execute(
            "INSERT INTO weather_data (field_id, temperature_c, humidity, rainfall_mm) VALUES (%s, %s, %s, %s);",
            (payload.field_id, payload.temperature_c, payload.humidity, payload.rainfall_mm)
        )

        conn.commit()
        cursor.close()
        conn.close()
        return {"status": "success", "message": "Telemetry successfully ingested"}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Database error: {str(e)}")
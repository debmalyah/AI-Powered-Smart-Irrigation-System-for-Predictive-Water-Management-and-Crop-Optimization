"""
Tests for main_api.py's FastAPI endpoints.

Requires a trained model artifact at MODEL_PATH (best_irrigation_model.joblib) --
run train_baseline.py / train_rf.py / train_gb.py then select_best_model.py first.

Run with pytest:
    pytest test_main_api.py -v

Or standalone, no pytest required:
    python test_main_api.py
"""
from fastapi.testclient import TestClient

from main_api import app, TARGET_ORDER

client = TestClient(app)

VALID_PAYLOAD = {
    "Soil_Type": "Loamy",
    "Soil_pH": 6.5,
    "Soil_Moisture": 32.5,
    "Organic_Carbon": 0.9,
    "Electrical_Conductivity": 1.8,
    "Temperature_C": 28.4,
    "Humidity": 65.0,
    "Rainfall_mm": 0.0,
    "Sunlight_Hours": 7.5,
    "Wind_Speed_kmh": 12.0,
    "Crop_Type": "Wheat",
    "Crop_Growth_Stage": "Vegetative",
    "Season": "Kharif",
    "Irrigation_Type": "Drip",
    "Water_Source": "Groundwater",
    "Field_Area_hectare": 7.5,
    "Mulching_Used": "Yes",
    "Previous_Irrigation_mm": 15.0,
    "Region": "North",
    "Forecast_Rain_mm": 2.0,
}


def test_health_check_reports_model_loaded():
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "Online"
    # If this is False, every test below will fail for the same underlying
    # reason: no best_irrigation_model.joblib on disk. Train + select first.
    assert body["model_loaded"] is True, (
        "No model artifact loaded -- run the training scripts and "
        "select_best_model.py before running these tests."
    )


def test_predict_and_schedule_valid_payload():
    resp = client.post("/predict_and_schedule", json=VALID_PAYLOAD)
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert "prediction" in body and "schedule_recommendation" in body
    assert body["prediction"]["irrigation_need"] in TARGET_ORDER
    assert body["prediction"]["class_index"] in range(len(TARGET_ORDER))

    schedule = body["schedule_recommendation"]
    for key in ("irrigation_required", "reason", "water_amount_mm", "duration_minutes", "recommended_slot"):
        assert key in schedule


def test_high_soil_moisture_skips_irrigation():
    # scheduler_engine's over-watering guard: >=80% soil moisture should
    # always return irrigation_required=False, regardless of what the model predicts.
    payload = {**VALID_PAYLOAD, "Soil_Moisture": 85.0}
    resp = client.post("/predict_and_schedule", json=payload)
    assert resp.status_code == 200, resp.text
    schedule = resp.json()["schedule_recommendation"]
    assert schedule["irrigation_required"] is False
    assert "threshold" in schedule["reason"].lower()


def test_heavy_forecast_rain_skips_irrigation():
    # >=10mm forecast rain should suspend irrigation regardless of prediction.
    payload = {**VALID_PAYLOAD, "Soil_Moisture": 20.0, "Forecast_Rain_mm": 15.0}
    resp = client.post("/predict_and_schedule", json=payload)
    assert resp.status_code == 200, resp.text
    schedule = resp.json()["schedule_recommendation"]
    assert schedule["irrigation_required"] is False
    assert "rain" in schedule["reason"].lower()


def test_missing_required_field_returns_422():
    payload = dict(VALID_PAYLOAD)
    del payload["Soil_Moisture"]
    resp = client.post("/predict_and_schedule", json=payload)
    assert resp.status_code == 422


def test_unknown_category_does_not_crash():
    # A category value the model never saw during training should be
    # absorbed by the one-hot reindex(fill_value=0) -- it should NOT 500.
    payload = {**VALID_PAYLOAD, "Region": "Some_New_Region_Not_In_Training"}
    resp = client.post("/predict_and_schedule", json=payload)
    assert resp.status_code == 200, resp.text
    assert resp.json()["prediction"]["irrigation_need"] in TARGET_ORDER


if __name__ == "__main__":
    tests = [
        test_health_check_reports_model_loaded,
        test_predict_and_schedule_valid_payload,
        test_high_soil_moisture_skips_irrigation,
        test_heavy_forecast_rain_skips_irrigation,
        test_missing_required_field_returns_422,
        test_unknown_category_does_not_crash,
    ]
    passed, failed = 0, 0
    for test_fn in tests:
        try:
            test_fn()
            print(f"PASS  {test_fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {test_fn.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR {test_fn.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")

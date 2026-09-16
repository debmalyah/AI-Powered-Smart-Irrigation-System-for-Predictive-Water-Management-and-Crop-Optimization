import datetime
from typing import Dict, Any

CROP_BASE_WATER_REQ = {
    "Wheat": 25.0,  # mm base
    "Rice": 45.0,
    "Maize": 30.0,
    "Cotton": 35.0,
    "Sugarcane": 50.0,
    "Potato": 32.0,  # PLACEHOLDER estimate -- Potato is present in the actual
                     # dataset but had no entry here (was silently falling back
                     # to the generic 30.0 default). Verify against a real
                     # agronomic reference before relying on this for production.
}

# NOTE: these stage names must match the dataset's actual Crop_Growth_Stage
# values exactly (checked against irrigation_prediction.csv: Sowing,
# Vegetative, Flowering, Harvest). The previous keys (Initial, Mid-Season,
# Late-Season) didn't match ANY of these except "Vegetative" -- 75% of rows
# were silently getting the fallback multiplier of 1.0 instead of a real
# stage-specific value, both as an ML feature and in the schedule math below.
# Values are PLACEHOLDER domain estimates (Sowing/Harvest ~ low demand,
# Flowering ~ peak demand) -- verify against a real agronomic reference.
GROWTH_STAGE_MULTIPLIER = {
    "Sowing": 0.6,
    "Vegetative": 1.0,
    "Flowering": 1.3,
    "Harvest": 0.7,
}


def generate_irrigation_schedule(
        prediction: str,
        crop_type: str,
        growth_stage: str,
        soil_moisture: float,
        forecast_rain_mm: float = 0.0
) -> Dict[str, Any]:
    """Generates precise water quantity, time-slotted recommendations, and over-watering checks."""

    # Safety Check: High soil moisture threshold
    if soil_moisture >= 80.0:
        return {
            "irrigation_required": False,
            "reason": "Soil moisture exceeds safety threshold (>=80%).",
            "water_amount_mm": 0.0,
            "duration_minutes": 0,
            "recommended_slot": None
        }

    # Safety Check: Forecasted heavy rainfall offset
    if forecast_rain_mm >= 10.0:
        return {
            "irrigation_required": False,
            "reason": f"Heavy rain forecasted ({forecast_rain_mm} mm). Irrigation suspended.",
            "water_amount_mm": 0.0,
            "duration_minutes": 0,
            "recommended_slot": None
        }

    if prediction == "Low":
        return {
            "irrigation_required": False,
            "reason": "Sufficient moisture detected; no irrigation required.",
            "water_amount_mm": 0.0,
            "duration_minutes": 0,
            "recommended_slot": None
        }

    # Calculation logic
    base_water = CROP_BASE_WATER_REQ.get(crop_type, 30.0)
    stage_mult = GROWTH_STAGE_MULTIPLIER.get(growth_stage, 1.0)
    pred_mult = 1.0 if prediction == "Medium" else 1.4

    net_water = max(0.0, (base_water * stage_mult * pred_mult) - forecast_rain_mm)
    duration_mins = int(net_water * 4)  # ~4 minutes per mm application rate

    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    slot = f"{tomorrow.isoformat()} 06:00 AM - 08:00 AM"

    return {
        "irrigation_required": True,
        "reason": f"Scheduled for {crop_type} ({growth_stage} stage) under '{prediction}' urgency.",
        "water_amount_mm": round(net_water, 2),
        "duration_minutes": duration_mins,
        "recommended_slot": slot
    }
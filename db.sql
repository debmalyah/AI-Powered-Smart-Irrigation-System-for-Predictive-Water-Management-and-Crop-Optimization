-- 1. CROP (Reference Table)
CREATE TABLE crops (
    crop_id SERIAL PRIMARY KEY,
    crop_name VARCHAR(50) UNIQUE NOT NULL,
    growth_cycle_days INT,
    base_water_need VARCHAR(20) -- e.g., 'Low', 'Medium', 'High'
);

-- 2. FARMER 
CREATE TABLE farmers (
    farmer_id SERIAL PRIMARY KEY,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    phone_number VARCHAR(15) UNIQUE NOT NULL,
    joined_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. FIELD (Links Farmer to Crop)
CREATE TABLE fields (
    field_id SERIAL PRIMARY KEY,
    farmer_id INT NOT NULL REFERENCES farmers(farmer_id) ON DELETE CASCADE,
    crop_id INT NOT NULL REFERENCES crops(crop_id),
    area_hectares FLOAT NOT NULL,
    soil_type VARCHAR(50) NOT NULL,
    gps_coordinates VARCHAR(100)
);

-- 4. SENSOR (Hardware Device Tracking)
CREATE TABLE sensors (
    sensor_id SERIAL PRIMARY KEY,
    field_id INT NOT NULL REFERENCES fields(field_id) ON DELETE CASCADE,
    sensor_type VARCHAR(50) NOT NULL, -- e.g., 'Soil Moisture Node'
    installation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- 5. SENSOR READING (Live Telemetry)
CREATE TABLE sensor_readings (
    reading_id SERIAL PRIMARY KEY,
    sensor_id INT NOT NULL REFERENCES sensors(sensor_id) ON DELETE CASCADE,
    soil_moisture_value FLOAT NOT NULL,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 6. WEATHER DATA (External API Logs)
CREATE TABLE weather_data (
    weather_id SERIAL PRIMARY KEY,
    field_id INT NOT NULL REFERENCES fields(field_id) ON DELETE CASCADE,
    temperature_c FLOAT NOT NULL,
    humidity FLOAT NOT NULL,
    rainfall_mm FLOAT DEFAULT 0.0,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 7. IRRIGATION HISTORY (Pump Activation Logs)
CREATE TABLE irrigation_history (
    history_id SERIAL PRIMARY KEY,
    field_id INT NOT NULL REFERENCES fields(field_id) ON DELETE CASCADE,
    pump_on_time TIMESTAMP NOT NULL,
    pump_off_time TIMESTAMP,
    water_volume_liters FLOAT, -- Calculated volume applied
    triggered_by VARCHAR(50) -- e.g., 'Manual', 'ML Model', 'Schedule'
);
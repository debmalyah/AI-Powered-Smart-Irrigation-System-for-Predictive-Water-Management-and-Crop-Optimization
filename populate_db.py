import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch

# 1. Connect to PostgreSQL
conn = psycopg2.connect(
    host="localhost",
    database="smart_irrigation",
    user="postgres",
    password="debmalya",  # Update with your password
    port="5432"
)
cursor = conn.cursor()

print("Reading Kaggle dataset...")
df = pd.read_csv('data/irrigation_prediction.csv')
df.columns = df.columns.str.strip()

print("Seeding parent tables dynamically from dataset...")

# 2. Insert a default farmer
cursor.execute("""
    INSERT INTO farmers (farmer_id, first_name, last_name, phone_number) 
    VALUES (1, 'Rajesh', 'Kumar', '+919876543210') 
    ON CONFLICT (farmer_id) DO NOTHING;
""")

# 3. Extract and insert ALL unique crops from the CSV
unique_crops = df['Crop_Type'].unique()
crop_id_map = {}

for idx, crop_name in enumerate(unique_crops, start=1):
    cursor.execute("""
        INSERT INTO crops (crop_id, crop_name, growth_cycle_days, base_water_need) 
        VALUES (%s, %s, 120, 'Medium') 
        ON CONFLICT (crop_name) DO UPDATE SET crop_name = EXCLUDED.crop_name
        RETURNING crop_id;
    """, (idx, crop_name))

    # Map crop name to its generated ID
    cursor.execute("SELECT crop_id FROM crops WHERE crop_name = %s;", (crop_name,))
    crop_id_map[crop_name] = cursor.fetchone()[0]

# 4. Create a default field for each crop type so foreign keys match correctly
field_id_map = {}
for crop_name, crop_id in crop_id_map.items():
    cursor.execute("""
        INSERT INTO fields (farmer_id, crop_id, area_hectares, soil_type) 
        VALUES (1, %s, 5.0, 'Loamy')
        RETURNING field_id;
    """, (crop_id,))
    # For simplicity, we can map rows or assign a default field
    field_id_map[crop_name] = cursor.fetchone()[0]

conn.commit()

print("Bulk inserting sensor readings and weather data mapped to respective fields...")

sensor_batch = []
weather_batch = []

for _, row in df.iterrows():
    c_name = row['Crop_Type']
    f_id = field_id_map.get(c_name, 1)  # Fallback to field 1 if missing

    sensor_batch.append((f_id, float(row['Soil_Moisture'])))
    weather_batch.append((f_id, float(row['Temperature_C']), float(row['Humidity']), float(row['Rainfall_mm'])))

# Adjust sensor_readings to accept field_id or map through sensor_id
# (Assuming your sensor table has multiple sensors linked to fields)
# For now, let's bulk insert into weather_data and sensor_readings:
execute_batch(
    cursor,
    "INSERT INTO sensor_readings (sensor_id, soil_moisture_value) VALUES (1, %s);",
    [(s[1],) for s in sensor_batch],
    page_size=1000
)

execute_batch(
    cursor,
    "INSERT INTO weather_data (field_id, temperature_c, humidity, rainfall_mm) VALUES (%s, %s, %s, %s);",
    weather_batch,
    page_size=1000
)

conn.commit()
cursor.close()
conn.close()

print("Successfully synced all unique crops and data into PostgreSQL!")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from data_eda import load_and_clean_data, engineer_features, TARGET_COL, TARGET_MAPPING, DATASET_PATH

TARGET_COL = "Irrigation_Need"
TARGET_ORDER = ["Low", "Medium", "High"]
TARGET_MAPPING = {"Low": 0, "Medium": 1, "High": 2}
DATASET_PATH = "irrigation_prediction.csv"

def load_and_clean_data(path: str = DATASET_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)

    categorical_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()

    for col in categorical_cols:
        df[col] = df[col].astype(str).str.strip()

    if np.isinf(df[numerical_cols]).sum().sum() > 0:
        df[numerical_cols] = df[numerical_cols].replace([np.inf, -np.inf], np.nan)

    if df[numerical_cols].isnull().sum().sum() > 0:
        df = df.dropna().reset_index(drop=True)

    df = df[df[TARGET_COL].isin(TARGET_ORDER)].reset_index(drop=True)
    return df

def engineer_features(X: pd.DataFrame, soil_moisture_max: float, rainfall_max: float) -> pd.DataFrame:
    X = X.copy()
    X["Moisture_Deficit"] = soil_moisture_max - X["Soil_Moisture"]
    X["Temperature_Humidity_Index"] = X["Temperature_C"] * (100 - X["Humidity"])
    X["Heat_Wind_Index"] = X["Temperature_C"] * X["Wind_Speed_kmh"]
    X["Rainfall_Moisture_Stress"] = (
        (rainfall_max - X["Rainfall_mm"]) * (soil_moisture_max - X["Soil_Moisture"])
    )
    X["Previous_Irrigation_Moisture_Ratio"] = (
        X["Previous_Irrigation_mm"] / (X["Soil_Moisture"] + 1)
    )
    return X

def get_preprocessed_data(path: str = DATASET_PATH):
    df = load_and_clean_data(path)
    df["Target_Encoded"] = df[TARGET_COL].map(TARGET_MAPPING)

    X_raw = df.drop(columns=[TARGET_COL, "Target_Encoded"])
    y = df["Target_Encoded"]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.20, random_state=42, stratify=y
    )

    train_soil_moisture_max = X_train_raw["Soil_Moisture"].max()
    train_rainfall_max = X_train_raw["Rainfall_mm"].max()

    X_train_fe = engineer_features(X_train_raw, train_soil_moisture_max, train_rainfall_max)
    X_test_fe = engineer_features(X_test_raw, train_soil_moisture_max, train_rainfall_max)

    categorical_features = X_train_fe.select_dtypes(include=["object", "string"]).columns
    X_train_encoded = pd.get_dummies(X_train_fe, columns=categorical_features, drop_first=True)
    X_test_encoded = pd.get_dummies(X_test_fe, columns=categorical_features, drop_first=True)
    X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_encoded)
    X_test_scaled = scaler.transform(X_test_encoded)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    return (
        X_train_scaled, X_test_scaled, y_train, y_test,
        cv, scaler, X_train_encoded.columns.tolist()
    )
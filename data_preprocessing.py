import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from data_eda import (
    load_and_clean_data, engineer_features,
    TARGET_COL, TARGET_ORDER, TARGET_MAPPING, DATASET_PATH,
)

# NOTE: load_and_clean_data / engineer_features / the constants above are all
# imported from data_eda.py on purpose -- they used to be duplicated here,
# which meant fixes made in data_eda.py (e.g. outlier handling) silently
# never reached the training pipeline. Don't redefine them in this file.

def get_preprocessed_data(path: str = DATASET_PATH, val_size: float = 0.20, test_size: float = 0.20):
    """Returns a 60/20/20 (by default) train/validation/test split, fully leakage-safe:
    every derived value (feature-engineering maxes, one-hot columns, the scaler) is fit
    on the TRAIN split only and then applied to validation and test.

    `cv` (StratifiedKFold) is still returned for use during hyperparameter search
    (RandomizedSearchCV etc.) -- that's a separate concern from the fixed validation
    split, which is for reporting an honest "how's it doing" number before the
    one-shot final test evaluation.
    """
    df = load_and_clean_data(path)
    df["Target_Encoded"] = df[TARGET_COL].map(TARGET_MAPPING)

    X_raw = df.drop(columns=[TARGET_COL, "Target_Encoded"])
    y = df["Target_Encoded"]

    # Step 1: carve off the test set first -- it stays untouched until final evaluation.
    X_trainval_raw, X_test_raw, y_trainval, y_test = train_test_split(
        X_raw, y, test_size=test_size, random_state=42, stratify=y
    )

    # Step 2: split the remainder into train/validation. val_size is expressed as a
    # fraction of the FULL dataset, so it's rescaled relative to what's left here.
    relative_val_size = val_size / (1.0 - test_size)
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_trainval_raw, y_trainval, test_size=relative_val_size, random_state=42, stratify=y_trainval
    )

    train_soil_moisture_max = X_train_raw["Soil_Moisture"].max()
    train_rainfall_max = X_train_raw["Rainfall_mm"].max()

    X_train_fe = engineer_features(X_train_raw, train_soil_moisture_max, train_rainfall_max)
    X_val_fe = engineer_features(X_val_raw, train_soil_moisture_max, train_rainfall_max)
    X_test_fe = engineer_features(X_test_raw, train_soil_moisture_max, train_rainfall_max)

    categorical_features = X_train_fe.select_dtypes(include=["object", "string"]).columns
    X_train_encoded = pd.get_dummies(X_train_fe, columns=categorical_features, drop_first=True)
    X_val_encoded = pd.get_dummies(X_val_fe, columns=categorical_features, drop_first=True)
    X_test_encoded = pd.get_dummies(X_test_fe, columns=categorical_features, drop_first=True)
    X_val_encoded = X_val_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)
    X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_encoded)
    X_val_scaled = scaler.transform(X_val_encoded)
    X_test_scaled = scaler.transform(X_test_encoded)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print(f"\nSplit sizes -> train: {len(X_train_raw)}, validation: {len(X_val_raw)}, test: {len(X_test_raw)}")

    return (
        X_train_scaled, X_val_scaled, X_test_scaled,
        y_train, y_val, y_test,
        cv, scaler, X_train_encoded.columns.tolist(),
        train_soil_moisture_max, train_rainfall_max,
    )
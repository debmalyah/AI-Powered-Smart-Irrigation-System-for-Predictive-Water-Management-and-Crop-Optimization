import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Reused from scheduler_engine.py on purpose: these are the same
# agronomic base-water and growth-stage assumptions the scheduler applies at
# serving time. Importing them (instead of redefining) keeps the model's
# crop/growth-stage features and the scheduler's logic from drifting apart.
from scheduler_engine import CROP_BASE_WATER_REQ, GROWTH_STAGE_MULTIPLIER

TARGET_COL = "Irrigation_Need"
TARGET_ORDER = ["Low", "Medium", "High"]
TARGET_MAPPING = {"Low": 0, "Medium": 1, "High": 2}
DATASET_PATH = "irrigation_prediction.csv"

# Continuous sensor/weather columns eligible for IQR-based outlier treatment.
# (Categorical columns and the target are excluded on purpose.)
OUTLIER_COLS = [
    "Soil_Moisture",
    "Temperature_C",
    "Humidity",
    "Wind_Speed_kmh",
    "Rainfall_mm",
    "Previous_Irrigation_mm",
]

# Per-column IQR multiplier overrides. Rainfall and prior irrigation are
# naturally right-skewed (lots of zeros, occasional heavy values) -- a heavy
# monsoon reading is a real event, not noise -- so they get a wider band than
# the default 1.5x used for the other, more symmetric sensor columns.
OUTLIER_FACTOR_OVERRIDES = {
    "Rainfall_mm": 3.0,
    "Previous_Irrigation_mm": 3.0,
}


# Physically valid ranges for continuous sensor/weather columns. A value
# outside these bounds is a sensor/data-entry error (e.g. Humidity=150%),
# not just an unusually extreme-but-real reading -- that's what OUTLIER_COLS
# above is for. (None means "no bound on that side".)
VALID_RANGES = {
    "Soil_Moisture": (0.0, 100.0),        # percentage
    "Humidity": (0.0, 100.0),             # percentage
    "Temperature_C": (-10.0, 55.0),       # plausible field air temperature
    "Wind_Speed_kmh": (0.0, 150.0),       # can't be negative; 150 covers extreme storms
    "Rainfall_mm": (0.0, None),           # can't be negative; no realistic upper cap
    "Previous_Irrigation_mm": (0.0, None),
}


def correct_invalid_values(df: pd.DataFrame, valid_ranges: dict = None) -> pd.DataFrame:
    """Flags physically impossible sensor/weather readings (out-of-range) as NaN.

    Distinct from handle_outliers(): this checks hard domain bounds
    (Humidity can't be negative or over 100%), not "statistically unusual
    but still physically plausible" values.
    """
    valid_ranges = valid_ranges or VALID_RANGES
    df = df.copy()

    print("\n" + "=" * 70)
    print("INVALID VALUE CHECK (domain range bounds)")
    print("=" * 70)
    for col, (lo, hi) in valid_ranges.items():
        if col not in df.columns:
            continue
        mask = pd.Series(False, index=df.index)
        if lo is not None:
            mask |= df[col] < lo
        if hi is not None:
            mask |= df[col] > hi
        n_invalid = int(mask.sum())
        bound_str = f"[{lo if lo is not None else '-inf'}, {hi if hi is not None else '+inf'}]"
        print(f"{col:30s} valid range {bound_str}: {n_invalid} invalid value(s)")
        if n_invalid > 0:
            df.loc[mask, col] = np.nan

    return df


# ---------------------------------------------------------------------------
# 1. LOAD & CLEAN DATA
# ---------------------------------------------------------------------------

def detect_outliers_iqr(df: pd.DataFrame, cols: list, factor: float = 1.5,
                         factor_overrides: dict = None) -> pd.DataFrame:
    """Returns a boolean DataFrame flagging IQR-based outliers, one column per feature in `cols`."""
    factor_overrides = factor_overrides or {}
    flags = pd.DataFrame(False, index=df.index, columns=cols)
    for col in cols:
        col_factor = factor_overrides.get(col, factor)
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - col_factor * iqr
        upper = q3 + col_factor * iqr
        flags[col] = (df[col] < lower) | (df[col] > upper)
    return flags


def handle_outliers(df: pd.DataFrame, cols: list, factor: float = 1.5, method: str = "cap",
                     factor_overrides: dict = None) -> pd.DataFrame:
    """Detects IQR-based outliers in `cols` and either caps (winsorizes) or removes affected rows.

    method="cap"    -> clip values to [Q1 - factor*IQR, Q3 + factor*IQR] (default; keeps every row)
    method="remove" -> drops any row flagged as an outlier in ANY of `cols`
    """
    factor_overrides = factor_overrides or {}
    df = df.copy()
    cols = [c for c in cols if c in df.columns]
    flags = detect_outliers_iqr(df, cols, factor=factor, factor_overrides=factor_overrides)

    print("\n" + "=" * 70)
    print("OUTLIER DETECTION (IQR)")
    print("=" * 70)
    for col in cols:
        col_factor = factor_overrides.get(col, factor)
        count = int(flags[col].sum())
        pct = (count / len(df) * 100) if len(df) else 0.0
        print(f"{col:30s} (factor={col_factor}): {count:5d} outliers ({pct:.2f}%)")

    if method == "cap":
        for col in cols:
            col_factor = factor_overrides.get(col, factor)
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - col_factor * iqr
            upper = q3 + col_factor * iqr
            df[col] = df[col].clip(lower=lower, upper=upper)
        n_affected = int(flags.any(axis=1).sum())
        print(f"\nCapped values in {n_affected} row(s) across flagged columns (clipped to whisker bounds).")
    elif method == "remove":
        mask_any = flags.any(axis=1)
        n_removed = int(mask_any.sum())
        df = df[~mask_any].reset_index(drop=True)
        print(f"\nRemoved {n_removed} row(s) flagged as outliers.")
    else:
        raise ValueError("method must be 'cap' or 'remove'")

    return df


def load_and_clean_data(path: str = DATASET_PATH, outlier_method: str = "cap") -> pd.DataFrame:
    """Loads CSV, cleans column headers, handles duplicates, invalid values, outliers, and missing target rows."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    print("=" * 70)
    print("DATASET OVERVIEW")
    print("=" * 70)
    print(f"Total Rows    : {df.shape[0]}")
    print(f"Total Columns : {df.shape[1]}")

    # Remove duplicates
    duplicate_count = df.duplicated().sum()
    print(f"\nDuplicate rows found: {duplicate_count}")
    if duplicate_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print("Duplicates removed.")

    # Strip categorical whitespace
    categorical_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    for col in categorical_cols:
        df[col] = df[col].astype(str).str.strip()

    # Handle infinite and null numerical values
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    if np.isinf(df[numerical_cols]).sum().sum() > 0:
        df[numerical_cols] = df[numerical_cols].replace([np.inf, -np.inf], np.nan)

    # Correct physically invalid sensor/weather values (out-of-range -> NaN)
    df = correct_invalid_values(df)

    if df[numerical_cols].isnull().sum().sum() > 0:
        df = df.dropna().reset_index(drop=True)
        print("Dropped invalid/null numerical records.")

    # Detect and handle outliers in the continuous sensor/weather columns
    df = handle_outliers(
        df, OUTLIER_COLS, method=outlier_method, factor_overrides=OUTLIER_FACTOR_OVERRIDES
    )

    # Filter target column validity
    df = df[df[TARGET_COL].isin(TARGET_ORDER)].reset_index(drop=True)

    print(f"Post-cleaning records: {df.shape[0]}")
    return df


# ---------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def engineer_features(X: pd.DataFrame, soil_moisture_max: float, rainfall_max: float) -> pd.DataFrame:
    """Creates derived features for domain-specific water stress, weather interactions,
    and crop/growth-stage water demand."""
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

    # Crop/growth-stage features: turn the same agronomic knowledge the
    # scheduler uses (base water need per crop, demand multiplier per growth
    # stage) into numeric signal for the model, instead of leaving Crop_Type
    # and Growth_Stage as purely arbitrary one-hot categories. Fallback
    # defaults (30.0 / 1.0) match scheduler_engine's .get(..., default).
    X["Crop_Base_Water_Req"] = X["Crop_Type"].map(CROP_BASE_WATER_REQ).fillna(30.0)
    X["Growth_Stage_Multiplier"] = X["Crop_Growth_Stage"].map(GROWTH_STAGE_MULTIPLIER).fillna(1.0)
    X["Expected_Water_Demand"] = X["Crop_Base_Water_Req"] * X["Growth_Stage_Multiplier"]

    # Raw Crop_Type / Growth_Stage columns are left in place -- they still
    # get one-hot encoded downstream (get_dummies) so the model can also pick
    # up on per-category effects the domain formula above doesn't capture.
    return X


# ---------------------------------------------------------------------------
# 2b. HISTORICAL IRRIGATION PATTERN ANALYSIS
# ---------------------------------------------------------------------------

def analyze_historical_irrigation_patterns(df: pd.DataFrame) -> None:
    """Dedicated EDA on Previous_Irrigation_mm: how prior irrigation amounts
    relate to current soil moisture, growth stage, crop type, and the
    Irrigation_Need label.

    Note: the dataset has no timestamp/sequence column, so "historical" here
    means the prior-irrigation reading attached to each record, analyzed
    against current conditions -- not a calendar-time trend.
    """
    print("\n" + "=" * 70)
    print("HISTORICAL IRRIGATION PATTERN ANALYSIS (Previous_Irrigation_mm)")
    print("=" * 70)

    print("\nOverall distribution:")
    print(df["Previous_Irrigation_mm"].describe().round(2))

    print("\nMean Previous_Irrigation_mm by Irrigation_Need:")
    print(df.groupby(TARGET_COL)["Previous_Irrigation_mm"].mean().reindex(TARGET_ORDER).round(2))

    print("\nMean Previous_Irrigation_mm by Growth_Stage:")
    print(df.groupby("Crop_Growth_Stage")["Previous_Irrigation_mm"].mean().round(2))

    print("\nMean Previous_Irrigation_mm by Crop_Type:")
    print(df.groupby("Crop_Type")["Previous_Irrigation_mm"].mean().round(2))

    corr_with_moisture = df["Previous_Irrigation_mm"].corr(df["Soil_Moisture"])
    print(f"\nCorrelation(Previous_Irrigation_mm, Soil_Moisture): {corr_with_moisture:.3f}")

    # Previous irrigation vs current irrigation need
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x=TARGET_COL, y="Previous_Irrigation_mm", order=TARGET_ORDER)
    plt.title("Previous Irrigation Amount vs Current Irrigation Need", fontsize=13, fontweight="bold")
    plt.xlabel("Irrigation Need Level")
    plt.ylabel("Previous Irrigation (mm)")
    plt.tight_layout()
    plt.savefig("previous_irrigation_vs_need.png", dpi=300)
    plt.close()

    # Previous irrigation by growth stage
    known_stage_order = ["Sowing", "Vegetative", "Flowering", "Harvest"]
    stage_order = [s for s in known_stage_order if s in df["Crop_Growth_Stage"].unique()]
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="Crop_Growth_Stage", y="Previous_Irrigation_mm",
                order=stage_order if stage_order else None)
    plt.title("Previous Irrigation Amount by Growth Stage", fontsize=13, fontweight="bold")
    plt.xlabel("Growth Stage")
    plt.ylabel("Previous Irrigation (mm)")
    plt.tight_layout()
    plt.savefig("previous_irrigation_by_growth_stage.png", dpi=300)
    plt.close()

    # Previous irrigation vs current soil moisture, colored by need
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=df, x="Previous_Irrigation_mm", y="Soil_Moisture",
        hue=TARGET_COL, hue_order=TARGET_ORDER, alpha=0.7
    )
    plt.title("Previous Irrigation vs Current Soil Moisture", fontsize=13, fontweight="bold")
    plt.xlabel("Previous Irrigation (mm)")
    plt.ylabel("Soil Moisture (%)")
    plt.tight_layout()
    plt.savefig("previous_irrigation_vs_soil_moisture.png", dpi=300)
    plt.close()

    print("\nHistorical irrigation pattern plots saved: previous_irrigation_vs_need.png, "
          "previous_irrigation_by_growth_stage.png, previous_irrigation_vs_soil_moisture.png")


# ---------------------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS & VISUALIZATION
# ---------------------------------------------------------------------------

def run_eda_and_visualizations(df: pd.DataFrame) -> None:
    """Prints descriptive statistics and generates/saves visual plots."""
    categorical_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()

    print("\n" + "=" * 70)
    print("CATEGORICAL FEATURE BREAKDOWN")
    print("=" * 70)
    for col in categorical_cols:
        print(f"{col}: {df[col].nunique()} unique values -> {df[col].unique()}")

    print("\n" + "=" * 70)
    print("NUMERICAL DESCRIPTIVE STATISTICS")
    print("=" * 70)
    print(df[numerical_cols].describe().T[["mean", "std", "min", "50%", "max"]])

    print("\n" + "=" * 70)
    print("TARGET CLASS DISTRIBUTION (%)")
    print("=" * 70)
    print(df[TARGET_COL].value_counts(normalize=True).mul(100).round(2))

    # Calculate ordinal correlation for EDA overview
    df_corr = df.copy()
    df_corr["Target_Numeric"] = df_corr[TARGET_COL].map(TARGET_MAPPING)
    target_correlations = (
        df_corr.select_dtypes(include=np.number)
        .corr()["Target_Numeric"]
        .drop("Target_Numeric")
        .sort_values(ascending=False)
    )
    print("\nFeature Correlations with Target (Irrigation Need):")
    print(target_correlations.round(3))

    # Plot Settings
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"font.size": 10})

    # 1. Target Class Distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=TARGET_COL, order=TARGET_ORDER)
    plt.title("Target Class Distribution: Irrigation Need", fontsize=13, fontweight="bold")
    plt.xlabel("Irrigation Need Level")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("target_class_distribution.png", dpi=300)
    plt.close()

    # 2. Soil Moisture Boxplot
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x=TARGET_COL, y="Soil_Moisture", order=TARGET_ORDER)
    plt.title("Soil Moisture vs Irrigation Need", fontsize=13, fontweight="bold")
    plt.xlabel("Irrigation Need Level")
    plt.ylabel("Soil Moisture (%)")
    plt.tight_layout()
    plt.savefig("soil_moisture_vs_irrigation.png", dpi=300)
    plt.close()

    # 3. Temperature vs Humidity Scatter
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=df, x="Temperature_C", y="Humidity",
        hue=TARGET_COL, hue_order=TARGET_ORDER, alpha=0.7
    )
    plt.title("Temperature vs Humidity by Irrigation Need", fontsize=13, fontweight="bold")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Humidity (%)")
    plt.tight_layout()
    plt.savefig("temperature_humidity_irrigation.png", dpi=300)
    plt.close()

    # 4. Crop Type Breakdown
    crop_target_props = pd.crosstab(
        df["Crop_Type"], df[TARGET_COL], normalize="index"
    )[TARGET_ORDER]
    plt.figure(figsize=(9, 5))
    crop_target_props.plot(kind="bar", stacked=True, ax=plt.gca(), edgecolor="black")
    plt.title("Irrigation Need Proportion by Crop Type", fontsize=13, fontweight="bold")
    plt.xlabel("Crop Type")
    plt.ylabel("Proportion")
    plt.xticks(rotation=30)
    plt.legend(title="Irrigation Need", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig("crop_type_irrigation_need.png", dpi=300)
    plt.close()

    # 5. Correlation Heatmap
    plt.figure(figsize=(11, 8))
    sns.heatmap(
        df[numerical_cols].corr(), annot=True, fmt=".2f",
        cmap="vlag", center=0, linewidths=0.5
    )
    plt.title("Numerical Feature Correlation Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png", dpi=300)
    plt.close()

    # 6. Historical irrigation pattern analysis (Previous_Irrigation_mm)
    analyze_historical_irrigation_patterns(df)

    print("\nVisualizations saved successfully to root directory.")


if __name__ == "__main__":
    df = load_and_clean_data()
    run_eda_and_visualizations(df)
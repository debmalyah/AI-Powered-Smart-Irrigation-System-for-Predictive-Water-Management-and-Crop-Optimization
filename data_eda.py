import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

TARGET_COL = "Irrigation_Need"
TARGET_ORDER = ["Low", "Medium", "High"]
TARGET_MAPPING = {"Low": 0, "Medium": 1, "High": 2}
DATASET_PATH = "irrigation_prediction.csv"


# ---------------------------------------------------------------------------
# 1. LOAD & CLEAN DATA
# ---------------------------------------------------------------------------

def load_and_clean_data(path: str = DATASET_PATH) -> pd.DataFrame:
    """Loads CSV, cleans column headers, handles duplicates, invalid values, and missing target rows."""
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

    if df[numerical_cols].isnull().sum().sum() > 0:
        df = df.dropna().reset_index(drop=True)
        print("Dropped invalid/null numerical records.")

    # Filter target column validity
    df = df[df[TARGET_COL].isin(TARGET_ORDER)].reset_index(drop=True)

    print(f"Post-cleaning records: {df.shape[0]}")
    return df


# ---------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def engineer_features(X: pd.DataFrame, soil_moisture_max: float, rainfall_max: float) -> pd.DataFrame:
    """Creates derived features for domain-specific water stress and weather interactions."""
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

    print("\nVisualizations saved successfully to root directory.")


if __name__ == "__main__":
    df = load_and_clean_data()
    run_eda_and_visualizations(df)
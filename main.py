"""
Usage:
    python irrigation_prediction_pipeline.py --model both
    python irrigation_prediction_pipeline.py --model gb --skip-eda
    python irrigation_prediction_pipeline.py --model rf
"""

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")

from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold,
    GridSearchCV, RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import randint, uniform


TARGET_COL = "Irrigation_Need"
TARGET_ORDER = ["Low", "Medium", "High"]
TARGET_MAPPING = {"Low": 0, "Medium": 1, "High": 2}

DATASET_PATH = (
    r"C:\Users\Mayukh\PycharmProjects"
    r"\AI-Powered-Smart-Irrigation-System-for-Predictive-Water-Management"
    r"-and-Crop-Optimization\irrigation_prediction.csv"
)


# ---------------------------------------------------------------------------
# 1. LOAD & CLEAN
# ---------------------------------------------------------------------------

def load_and_clean_data(path: str) -> pd.DataFrame:
    """Load the CSV and apply the fixed cleaning steps (cheap, always run)."""

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    print("=" * 70)
    print("DATASET INFORMATION")
    print("=" * 70)
    print(f"Number of rows    : {df.shape[0]}")
    print(f"Number of columns : {df.shape[1]}")

    categorical_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()

    print("\n" + "=" * 70)
    print("DATA CLEANING")
    print("=" * 70)

    # Duplicates
    duplicate_count = df.duplicated().sum()
    print(f"\nDuplicate rows: {duplicate_count}")
    if duplicate_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print("Duplicate rows removed.")

    # Categorical whitespace cleanup
    for col in categorical_cols:
        df[col] = df[col].astype(str).str.strip()

    # Infinite values -> NaN
    infinite_counts = np.isinf(df[numerical_cols]).sum()
    if infinite_counts.sum() > 0:
        print(f"\nInfinite values found:\n{infinite_counts[infinite_counts > 0]}")
        df[numerical_cols] = df[numerical_cols].replace([np.inf, -np.inf], np.nan)
    else:
        print("\nNo infinite values found.")

    # Drop rows with any remaining invalid numerics
    if df[numerical_cols].isnull().sum().sum() > 0:
        print("\nDropping rows with invalid numerical values.")
        df = df.dropna().reset_index(drop=True)
    else:
        print("No invalid numerical values found.")

    # Validate target
    invalid_target_rows = df[~df[TARGET_COL].isin(TARGET_ORDER)]
    if len(invalid_target_rows) > 0:
        df = df[df[TARGET_COL].isin(TARGET_ORDER)].reset_index(drop=True)
        print(f"\nRemoved {len(invalid_target_rows)} invalid target records.")
    else:
        print("\nAll target values are valid.")

    print("\n" + "=" * 70)
    print("POST-CLEANING DATASET")
    print("=" * 70)
    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")
    print(f"\nTarget distribution:\n{df[TARGET_COL].value_counts()}")

    return df


# ---------------------------------------------------------------------------
# 2. EDA (skippable — gated behind --skip-eda since plotting is the slow,
#    non-essential part when you're just iterating on models)
# ---------------------------------------------------------------------------

def run_eda(df: pd.DataFrame) -> None:
    categorical_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()

    print("\n" + "=" * 70)
    print("CATEGORICAL FEATURE ANALYSIS")
    print("=" * 70)
    for col in categorical_cols:
        print(f"{col}: {df[col].nunique()} unique values -> {df[col].unique()}")

    print("\n" + "=" * 70)
    print("NUMERICAL FEATURE ANALYSIS")
    print("=" * 70)
    print(df[numerical_cols].describe().T[["mean", "std", "min", "50%", "max"]])

    print("\n" + "=" * 70)
    print("TARGET VARIABLE ANALYSIS")
    print("=" * 70)
    print("\nTarget Class Distribution (%):")
    print(df[TARGET_COL].value_counts(normalize=True).mul(100).round(2))

    # Correlation with target — EDA-only ordinal encoding, never fed to the model
    df_corr = df.copy()
    df_corr["Target_Numeric"] = df_corr[TARGET_COL].map(TARGET_MAPPING)
    target_correlations = (
        df_corr.select_dtypes(include=np.number)
        .corr()["Target_Numeric"]
        .drop("Target_Numeric")
        .sort_values(ascending=False)
    )
    print("\nCorrelation with Irrigation Need:")
    print(target_correlations.round(3))

    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"font.size": 10})

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=TARGET_COL, order=TARGET_ORDER)
    plt.title("Target Class Distribution: Irrigation Need", fontsize=13, fontweight="bold")
    plt.xlabel("Irrigation Need Level")
    plt.ylabel("Number of Fields")
    plt.tight_layout()
    plt.savefig("target_class_distribution.png", dpi=300)
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x=TARGET_COL, y="Soil_Moisture", order=TARGET_ORDER)
    plt.title("Soil Moisture vs Irrigation Need", fontsize=13, fontweight="bold")
    plt.xlabel("Irrigation Need Level")
    plt.ylabel("Soil Moisture (%)")
    plt.tight_layout()
    plt.savefig("soil_moisture_vs_irrigation.png", dpi=300)
    plt.show()

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
    plt.show()

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
    plt.show()

    plt.figure(figsize=(11, 8))
    sns.heatmap(
        df[numerical_cols].corr(), annot=True, fmt=".2f",
        cmap="vlag", center=0, linewidths=0.5
    )
    plt.title("Numerical Feature Correlation Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png", dpi=300)
    plt.show()


# ---------------------------------------------------------------------------
# 3. PREPROCESSING (split -> feature engineering -> encode -> scale)
#    Split happens BEFORE feature engineering so the engineered features
#    (which use dataset max() values) are built from TRAINING stats only —
#    otherwise test-set information leaks into training features.
# ---------------------------------------------------------------------------

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


def preprocess(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("TARGET ENCODING & TRAIN-TEST SPLIT")
    print("=" * 70)

    df = df.copy()
    df["Target_Encoded"] = df[TARGET_COL].map(TARGET_MAPPING)

    X_raw = df.drop(columns=[TARGET_COL, "Target_Encoded"])
    y = df["Target_Encoded"]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training rows : {X_train_raw.shape[0]}")
    print(f"Testing rows  : {X_test_raw.shape[0]}")

    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING")
    print("=" * 70)

    train_soil_moisture_max = X_train_raw["Soil_Moisture"].max()
    train_rainfall_max = X_train_raw["Rainfall_mm"].max()

    X_train_fe = engineer_features(X_train_raw, train_soil_moisture_max, train_rainfall_max)
    X_test_fe = engineer_features(X_test_raw, train_soil_moisture_max, train_rainfall_max)
    print(f"Feature columns after engineering: {X_train_fe.shape[1]}")

    print("\n" + "=" * 70)
    print("CATEGORICAL ENCODING")
    print("=" * 70)

    categorical_features = X_train_fe.select_dtypes(include=["object", "string"]).columns
    X_train_encoded = pd.get_dummies(X_train_fe, columns=categorical_features, drop_first=True)
    X_test_encoded = pd.get_dummies(X_test_fe, columns=categorical_features, drop_first=True)
    # Reindex test onto training columns so a category seen in only one split
    # can't create a mismatch or a hidden leak.
    X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)
    print(f"Encoded columns: {X_train_encoded.shape[1]}")

    print("\n" + "=" * 70)
    print("FEATURE SCALING")
    print("=" * 70)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_encoded)
    X_test_scaled = scaler.transform(X_test_encoded)
    print(f"Training features shape : {X_train_scaled.shape}")
    print(f"Testing features shape  : {X_test_scaled.shape}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    return X_train_scaled, X_test_scaled, y_train, y_test, cv


# ---------------------------------------------------------------------------
# 4. RANDOM FOREST BRANCH — baseline, SMOTE+CV, GridSearchCV, final eval
# ---------------------------------------------------------------------------

def run_random_forest(X_train_scaled, X_test_scaled, y_train, y_test, cv):
    print("\n" + "=" * 70)
    print("BASELINE MODEL: RANDOM FOREST")
    print("=" * 70)

    rf_model = RandomForestClassifier(
        n_estimators=100, max_depth=10, class_weight="balanced",
        random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    y_pred_rf = rf_model.predict(X_test_scaled)
    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    print(f"Overall Accuracy: {rf_accuracy:.4f}\n")
    print(classification_report(y_test, y_pred_rf, target_names=TARGET_ORDER))

    cm_rf = confusion_matrix(y_test, y_pred_rf)
    disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=TARGET_ORDER)

    print("\n" + "=" * 70)
    print("SMOTE-BALANCED MODEL: RANDOM FOREST")
    print("=" * 70)

    # SMOTE must live INSIDE the pipeline so cross_val_score refits it fresh on
    # each fold's training split only — resampling once beforehand would let
    # synthetic points derived from a validation fold's neighbors leak into
    # the training folds and inflate CV scores.
    smote_pipeline = ImbPipeline([
        ("smote", SMOTE(random_state=42)),
        ("rf", RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42,
            n_jobs=1  # left at 1: this pipeline is also the GridSearchCV
                      # estimator below, and nesting two n_jobs=-1 oversubscribes
        )),
    ])

    cv_scores = cross_val_score(smote_pipeline, X_train_scaled, y_train, cv=cv, scoring="accuracy")
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    smote_pipeline.fit(X_train_scaled, y_train)
    y_pred_smote_test = smote_pipeline.predict(X_test_scaled)
    smote_test_accuracy = accuracy_score(y_test, y_pred_smote_test)
    print(f"SMOTE Model Test Accuracy: {smote_test_accuracy:.4f}\n")
    print(classification_report(y_test, y_pred_smote_test, target_names=TARGET_ORDER))

    cm_smote = confusion_matrix(y_test, y_pred_smote_test)
    disp_smote = ConfusionMatrixDisplay(confusion_matrix=cm_smote, display_labels=TARGET_ORDER)

    fig, (ax_rf, ax_smote) = plt.subplots(1, 2, figsize=(14, 6))
    disp_rf.plot(cmap="Blues", ax=ax_rf, values_format="d", colorbar=False)
    ax_rf.set_title("Baseline (Class-Weighted)", fontsize=12, fontweight="bold")
    disp_smote.plot(cmap="Blues", ax=ax_smote, values_format="d", colorbar=False)
    ax_smote.set_title("SMOTE-Balanced", fontsize=12, fontweight="bold")
    fig.suptitle("Random Forest Confusion Matrices", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("confusion_matrix_comparison.png", dpi=300)
    plt.show()

    # Kept as GridSearchCV: the grid is small (2*4*3*3 = 72 combinations), so
    # exhaustive search is cheap and guarantees the true best combination in
    # the grid rather than a sampled approximation.
    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING: GRIDSEARCHCV")
    print("=" * 70)

    param_grid = {
        "rf__n_estimators": [100, 200],
        "rf__max_depth": [8, 12, 16, None],
        "rf__min_samples_split": [2, 5, 10],
        "rf__min_samples_leaf": [1, 2, 4],
    }
    grid_search = GridSearchCV(
        estimator=smote_pipeline, param_grid=param_grid,
        cv=cv, scoring="accuracy", n_jobs=-1
    )
    grid_search.fit(X_train_scaled, y_train)
    best_rf_model = grid_search.best_estimator_
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best CV Score: {grid_search.best_score_:.4f}")

    print("\n" + "=" * 70)
    print("FINAL EVALUATION: TUNED RANDOM FOREST ON TEST SET")
    print("=" * 70)

    y_pred_best = best_rf_model.predict(X_test_scaled)
    best_test_accuracy = accuracy_score(y_test, y_pred_best)
    print(f"Tuned Model - Test Accuracy: {best_test_accuracy:.4f}\n")
    print(classification_report(y_test, y_pred_best, target_names=TARGET_ORDER))

    cm_best = confusion_matrix(y_test, y_pred_best)
    disp_best = ConfusionMatrixDisplay(confusion_matrix=cm_best, display_labels=TARGET_ORDER)
    plt.figure(figsize=(7, 6))
    disp_best.plot(cmap="Blues", values_format="d", ax=plt.gca())
    plt.title("Tuned Random Forest (Test Set)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("confusion_matrix_tuned.png", dpi=300)
    plt.show()

    print("\nFINAL RANDOM FOREST COMPARISON")
    print(f"Baseline (class-weighted) - Test Accuracy : {rf_accuracy:.4f}")
    print(f"SMOTE-balanced            - Test Accuracy : {smote_test_accuracy:.4f}")
    print(f"Tuned (GridSearchCV)      - Test Accuracy : {best_test_accuracy:.4f}")

    return {"best_model": best_rf_model, "test_accuracy": best_test_accuracy}


# ---------------------------------------------------------------------------
# 5. GRADIENT BOOSTING BRANCH — baseline, SMOTE+CV, RandomizedSearchCV, final eval
#    Uses HistGradientBoostingClassifier + early stopping for speed (see
#    module docstring). A separate, lighter 3-fold CV is used just for the
#    randomized search itself to further cut runtime; the reported CV-score
#    comparisons elsewhere still use the standard 5-fold `cv` for consistency
#    with the Random Forest numbers.
# ---------------------------------------------------------------------------

def make_hgb(random_state=42):
    return HistGradientBoostingClassifier(
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
    )


def run_gradient_boosting(X_train_scaled, X_test_scaled, y_train, y_test, cv):
    print("\n" + "=" * 70)
    print("BASELINE MODEL: GRADIENT BOOSTING (HistGradientBoosting)")
    print("=" * 70)

    gb_model = make_hgb()
    gb_model.fit(X_train_scaled, y_train)
    y_pred_gb = gb_model.predict(X_test_scaled)
    gb_accuracy = accuracy_score(y_test, y_pred_gb)
    print(f"Overall Accuracy: {gb_accuracy:.4f}\n")
    print(classification_report(y_test, y_pred_gb, target_names=TARGET_ORDER))

    print("\n" + "=" * 70)
    print("SMOTE-BALANCED MODEL: GRADIENT BOOSTING")
    print("=" * 70)

    smote_gb_pipeline = ImbPipeline([
        ("smote", SMOTE(random_state=42)),
        ("gb", make_hgb()),
    ])

    gb_cv_scores = cross_val_score(
        smote_gb_pipeline, X_train_scaled, y_train, cv=cv, scoring="accuracy", n_jobs=-1
    )
    print(f"Mean CV Accuracy: {gb_cv_scores.mean():.4f} (+/- {gb_cv_scores.std() * 2:.4f})")

    smote_gb_pipeline.fit(X_train_scaled, y_train)
    y_pred_smote_gb = smote_gb_pipeline.predict(X_test_scaled)
    smote_gb_test_accuracy = accuracy_score(y_test, y_pred_smote_gb)
    print(f"SMOTE Model Test Accuracy: {smote_gb_test_accuracy:.4f}\n")
    print(classification_report(y_test, y_pred_smote_gb, target_names=TARGET_ORDER))

    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING: GRADIENT BOOSTING (RandomizedSearchCV)")
    print("=" * 70)

    # max_iter replaces n_estimators; min_samples_leaf/max_leaf_nodes are
    # HistGradientBoostingClassifier's depth-control knobs (no min_samples_split).
    gb_param_distributions = {
        "gb__max_iter": randint(100, 300),
        "gb__learning_rate": uniform(0.02, 0.28),
        "gb__max_depth": [None, 3, 5, 7, 10],
        "gb__max_leaf_nodes": randint(15, 63),
        "gb__min_samples_leaf": randint(10, 40),
    }

    # Lighter CV just for the search to keep runtime down; early_stopping on
    # the estimator itself also caps each individual fit's tree count.
    cv_search = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    gb_random_search = RandomizedSearchCV(
        estimator=smote_gb_pipeline,
        param_distributions=gb_param_distributions,
        n_iter=20,
        cv=cv_search,
        scoring="accuracy",
        n_jobs=-1,
        random_state=42,
    )

    print("Running RandomizedSearchCV for Gradient Boosting...")
    gb_random_search.fit(X_train_scaled, y_train)
    best_gb_model = gb_random_search.best_estimator_
    print(f"Best Parameters: {gb_random_search.best_params_}")
    print(f"Best CV Score: {gb_random_search.best_score_:.4f}")

    print("\n" + "=" * 70)
    print("FINAL EVALUATION: TUNED GRADIENT BOOSTING ON TEST SET")
    print("=" * 70)

    y_pred_best_gb = best_gb_model.predict(X_test_scaled)
    best_gb_test_accuracy = accuracy_score(y_test, y_pred_best_gb)
    print(f"Tuned Model - Test Accuracy: {best_gb_test_accuracy:.4f}\n")
    print(classification_report(y_test, y_pred_best_gb, target_names=TARGET_ORDER))

    cm_best_gb = confusion_matrix(y_test, y_pred_best_gb)
    disp_best_gb = ConfusionMatrixDisplay(confusion_matrix=cm_best_gb, display_labels=TARGET_ORDER)
    plt.figure(figsize=(7, 6))
    disp_best_gb.plot(cmap="Greens", values_format="d", ax=plt.gca())
    plt.title("Tuned Gradient Boosting (Test Set)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("confusion_matrix_tuned_gb.png", dpi=300)
    plt.show()

    print("\nFINAL GRADIENT BOOSTING COMPARISON")
    print(f"Baseline                  - Test Accuracy : {gb_accuracy:.4f}")
    print(f"SMOTE-balanced            - Test Accuracy : {smote_gb_test_accuracy:.4f}")
    print(f"Tuned (RandomizedSearchCV)- Test Accuracy : {best_gb_test_accuracy:.4f}")

    return {"best_model": best_gb_model, "test_accuracy": best_gb_test_accuracy}


# ---------------------------------------------------------------------------
# 6. ENTRY POINT
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Irrigation need prediction pipeline")
    parser.add_argument(
        "--model", choices=["rf", "gb", "both"], default="both",
        help="Which model branch to run (default: both)"
    )
    parser.add_argument(
        "--skip-eda", action="store_true",
        help="Skip the plotting/EDA section (faster iteration on models)"
    )
    parser.add_argument(
        "--data-path", default=DATASET_PATH,
        help="Path to irrigation_prediction.csv"
    )
    args = parser.parse_args()

    df = load_and_clean_data(args.data_path)

    if not args.skip_eda:
        run_eda(df)

    X_train_scaled, X_test_scaled, y_train, y_test, cv = preprocess(df)

    results = {}
    if args.model in ("rf", "both"):
        results["rf"] = run_random_forest(X_train_scaled, X_test_scaled, y_train, y_test, cv)

    if args.model in ("gb", "both"):
        results["gb"] = run_gradient_boosting(X_train_scaled, X_test_scaled, y_train, y_test, cv)

    if args.model == "both":
        print("\n" + "=" * 70)
        print("OVERALL CHAMPION MODEL COMPARISON")
        print("=" * 70)
        print(f"Tuned Random Forest      - Test Accuracy : {results['rf']['test_accuracy']:.4f}")
        print(f"Tuned Gradient Boosting  - Test Accuracy : {results['gb']['test_accuracy']:.4f}")


if __name__ == "__main__":
    main()
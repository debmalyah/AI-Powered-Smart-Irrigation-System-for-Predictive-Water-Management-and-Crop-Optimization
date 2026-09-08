# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold

# 2. LOAD DATASET

dataset2_path = (
    r'C:\Users\Mayukh\PycharmProjects'
    r'\AI-Powered-Smart-Irrigation-System-for-Predictive-Water-Management'
    r'-and-Crop-Optimization\irrigation_prediction.csv'
)

df = pd.read_csv(dataset2_path)

# Clean column names immediately after loading
df.columns = df.columns.str.strip()

# Track the raw column count now, before any feature engineering,
# so later summaries don't rely on a hardcoded number
original_feature_count = df.shape[1]

# 3. BASIC DATASET INFORMATION

print("=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print(f"Number of rows    : {df.shape[0]}")
print(f"Number of columns : {df.shape[1]}")

print("\nFirst 3 records:")
print(df.head(3).T)

print("\nColumn Information:")
df.info()


# 4. CATEGORICAL FEATURE ANALYSIS

print("\n" + "=" * 70)
print("CATEGORICAL FEATURE ANALYSIS")
print("=" * 70)

categorical_cols = df.select_dtypes(
    include=['object', 'string']
).columns.tolist()

for col in categorical_cols:
    print(
        f"{col}: "
        f"{df[col].nunique()} unique values -> "
        f"{df[col].unique()}"
    )

# 5. NUMERICAL FEATURE ANALYSIS

print("\n" + "=" * 70)
print("NUMERICAL FEATURE ANALYSIS")
print("=" * 70)

numerical_cols = df.select_dtypes(
    include=np.number
).columns.tolist()

print(
    df[numerical_cols]
    .describe()
    .T[['mean', 'std', 'min', '50%', 'max']]
)


# 6. TARGET VARIABLE ANALYSIS

print("\n" + "=" * 70)
print("TARGET VARIABLE ANALYSIS")
print("=" * 70)

target_col = 'Irrigation_Need'

target_distribution = (
    df[target_col]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nTarget Class Distribution (%):")
print(target_distribution)


# 7. TARGET CORRELATION ANALYSIS

# Convert target to numeric only for exploratory correlation analysis
df_corr = df.copy()

df_corr['Target_Numeric'] = (
    df_corr[target_col]
    .map({
        'Low': 0,
        'Medium': 1,
        'High': 2
    })
)

correlation_df = df_corr.select_dtypes(
    include=np.number
)

target_correlations = (
    correlation_df
    .corr()['Target_Numeric']
    .drop('Target_Numeric')
    .sort_values(ascending=False)
)

print("\nCorrelation with Irrigation Need:")
print(target_correlations.round(3))


# 8. DATA VISUALIZATION

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 10})

target_order = ['Low', 'Medium', 'High']


# 8.1 Target Class Distribution
plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x=target_col,
    order=target_order
)

plt.title(
    'Target Class Distribution: Irrigation Need',
    fontsize=13,
    fontweight='bold'
)

plt.xlabel('Irrigation Need Level')
plt.ylabel('Number of Fields')

plt.tight_layout()
plt.savefig(
    'target_class_distribution.png',
    dpi=300
)
plt.show()

# 8.2 Soil Moisture vs Irrigation Need
plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x=target_col,
    y='Soil_Moisture',
    order=target_order
)

plt.title(
    'Soil Moisture vs Irrigation Need',
    fontsize=13,
    fontweight='bold'
)

plt.xlabel('Irrigation Need Level')
plt.ylabel('Soil Moisture (%)')

plt.tight_layout()
plt.savefig(
    'soil_moisture_vs_irrigation.png',
    dpi=300
)
plt.show()

# 8.3 Temperature vs Humidity
plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x='Temperature_C',
    y='Humidity',
    hue=target_col,
    hue_order=target_order,
    alpha=0.7
)

plt.title(
    'Temperature vs Humidity by Irrigation Need',
    fontsize=13,
    fontweight='bold'
)

plt.xlabel('Temperature (°C)')
plt.ylabel('Humidity (%)')

plt.tight_layout()
plt.savefig(
    'temperature_humidity_irrigation.png',
    dpi=300
)
plt.show()

# 8.4 Crop Type vs Irrigation Need
crop_target_props = pd.crosstab(
    df['Crop_Type'],
    df[target_col],
    normalize='index'
)[target_order]

plt.figure(figsize=(9, 5))

crop_target_props.plot(
    kind='bar',
    stacked=True,
    ax=plt.gca(),
    edgecolor='black'
)

plt.title(
    'Irrigation Need Proportion by Crop Type',
    fontsize=13,
    fontweight='bold'
)

plt.xlabel('Crop Type')
plt.ylabel('Proportion')

plt.xticks(rotation=30)
plt.legend(
    title='Irrigation Need',
    bbox_to_anchor=(1.02, 1),
    loc='upper left'
)

plt.tight_layout()
plt.savefig(
    'crop_type_irrigation_need.png',
    dpi=300
)
plt.show()

# 8.5 Numerical Feature Correlation Heatmap
plt.figure(figsize=(11, 8))

correlation_matrix = df[numerical_cols].corr()

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt='.2f',
    cmap='vlag',
    center=0,
    linewidths=0.5
)

plt.title(
    'Numerical Feature Correlation Matrix',
    fontsize=13,
    fontweight='bold'
)

plt.tight_layout()
plt.savefig(
    'correlation_heatmap.png',
    dpi=300
)
plt.show()

# 9. DATA CLEANING

print("\n" + "=" * 70)
print("DATA CLEANING")
print("=" * 70)

# 9.1 Missing Value Check
print("\nMissing Values:")

missing_values = df.isnull().sum()

if missing_values.sum() == 0:
    print("No missing values found.")
else:
    print(missing_values[missing_values > 0])

# 9.2 Duplicate Check
duplicate_count = df.duplicated().sum()

print("\nDuplicate Rows:")
print(f"Number of duplicate rows: {duplicate_count}")

if duplicate_count > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    print("Duplicate rows removed.")
else:
    print("No duplicate rows found.")

# 9.3 Clean Categorical Values
# Reuses categorical_cols from section 4 — columns haven't changed,
# only rows, so recomputing it here was redundant
for col in categorical_cols:
    df[col] = df[col].astype(str).str.strip()

print("\nCategorical values cleaned.")

# 9.4 Infinite Value Check
infinite_counts = np.isinf(
    df[numerical_cols]
).sum()

print("\nInfinite Values:")

if infinite_counts.sum() == 0:
    print("No infinite values found.")
else:
    print(infinite_counts[infinite_counts > 0])

    # Replace infinite values with NaN
    df[numerical_cols] = df[numerical_cols].replace(
        [np.inf, -np.inf],
        np.nan
    )

# 9.5 Handle Invalid Numerical Values
if df[numerical_cols].isnull().sum().sum() > 0:

    print("\nRows containing invalid numerical values:")
    print(
        df[
            df[numerical_cols]
            .isnull()
            .any(axis=1)
        ]
    )

    df = df.dropna().reset_index(drop=True)

else:
    print("\nNo invalid numerical values found.")

# 9.6 Validate Target Variable
valid_targets = ['Low', 'Medium', 'High']

invalid_target_rows = df[
    ~df[target_col].isin(valid_targets)
]

print("\nTarget Validation:")
print(f"Invalid target records: {len(invalid_target_rows)}")

if len(invalid_target_rows) > 0:

    df = df[
        df[target_col].isin(valid_targets)
    ].reset_index(drop=True)

    print("Invalid target records removed.")
else:
    print("All target values are valid.")

# 10. DATASET STATUS AFTER CLEANING

print("\n" + "=" * 70)
print("POST-CLEANING DATASET")
print("=" * 70)

print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

print("\nRemaining missing values:")
print(df.isnull().sum().sum())

print("\nTarget distribution:")
print(df[target_col].value_counts())

# 11. FEATURE ENGINEERING
print("\n" + "=" * 70)
print("FEATURE ENGINEERING")
print("=" * 70)

# 11.1 Moisture Deficit
df['Moisture_Deficit'] = (
    df['Soil_Moisture'].max()
    - df['Soil_Moisture']
)

# 11.2 Temperature-Humidity Interaction
df['Temperature_Humidity_Index'] = (
    df['Temperature_C']
    * (100 - df['Humidity'])
)
# 11.3 Temperature-Wind Interaction

df['Heat_Wind_Index'] = (
    df['Temperature_C']
    * df['Wind_Speed_kmh']
)

# 11.4 Rainfall-Moisture Stress
df['Rainfall_Moisture_Stress'] = (
    (
        df['Rainfall_mm'].max()
        - df['Rainfall_mm']
    )
    *
    (
        df['Soil_Moisture'].max()
        - df['Soil_Moisture']
    )
)

# 11.5 Previous Irrigation / Soil Moisture Ratio
df['Previous_Irrigation_Moisture_Ratio'] = (
    df['Previous_Irrigation_mm']
    / (df['Soil_Moisture'] + 1)
)

# 12. ENGINEERED FEATURE SUMMARY

engineered_features = [
    'Moisture_Deficit',
    'Temperature_Humidity_Index',
    'Heat_Wind_Index',
    'Rainfall_Moisture_Stress',
    'Previous_Irrigation_Moisture_Ratio'
]

print("\nEngineered Features:")

for feature in engineered_features:
    print(f"• {feature}")

print(
    f"\nOriginal number of features : {original_feature_count}"
)

print(
    f"Final number of features    : {df.shape[1]}"
)

print("\nEngineered Feature Statistics:")

print(
    df[engineered_features]
    .describe()
    .T[['mean', 'std', 'min', '50%', 'max']]
)

# 13. FINAL DATASET PREVIEW
print("\n" + "=" * 70)
print("FINAL DATASET PREVIEW")
print("=" * 70)

print(df.head().T)

print("\nFinal columns:")
print(df.columns.tolist())


# 14. MODEL PREPARATION
print("\n" + "=" * 70)
print("MODEL PREPARATION")
print("=" * 70)

# 14.1 Target Encoding
target_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
df['Target_Encoded'] = df['Irrigation_Need'].map(target_mapping)

# Separate Features and Target
X = df.drop(columns=['Irrigation_Need', 'Target_Encoded'])
y = df['Target_Encoded']

# 14.2 Categorical Encoding (One-Hot)
categorical_features = X.select_dtypes(include=['object', 'string']).columns
X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True)

# 14.3 Train-Test Split (80/20 Stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.20, random_state=42, stratify=y
)

# 14.4 Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training features shape : {X_train_scaled.shape}")
print(f"Testing features shape  : {X_test_scaled.shape}")


# 15. BASELINE MODEL - RANDOM FOREST
print("\n" + "=" * 70)
print("BASELINE MODEL: RANDOM FOREST")
print("=" * 70)

# Initialize the model with balanced class weights
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10, # Limits depth to prevent overfitting
    class_weight='balanced',
    random_state=42,
    n_jobs=-1 # Utilizes all CPU cores for faster training
)

# Train the model
rf_model.fit(X_train_scaled, y_train)

# Generate predictions on the unseen test set
y_pred_rf = rf_model.predict(X_test_scaled)

# Evaluate performance
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f"Overall Accuracy: {rf_accuracy:.4f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred_rf, target_names=['Low', 'Medium', 'High']))

# Generate the Confusion Matrix (plotted together with the SMOTE model's below)
cm_rf = confusion_matrix(y_test, y_pred_rf)
disp_rf = ConfusionMatrixDisplay(
    confusion_matrix=cm_rf,
    display_labels=['Low', 'Medium', 'High']
)


# 16. SMOTE-BALANCED MODEL WITH PROPER CROSS-VALIDATION
print("\n" + "=" * 70)
print("SMOTE-BALANCED MODEL: RANDOM FOREST")
print("=" * 70)

# Preview-only: show what SMOTE does to the class balance.This resampled copy is NOT used for training/CV below — it's just for the printed comparison.
X_train_smote_preview, y_train_smote_preview = SMOTE(random_state=42).fit_resample(
    X_train_scaled, y_train
)
print(f"Original training target distribution:\n{y_train.value_counts()}")
print(f"SMOTE training target distribution (preview):\n{y_train_smote_preview.value_counts()}")

# NOTE: SMOTE must be applied *inside* cross-validation, not once on the whole training set beforehand. Resampling first and then doing K-Fold CV
# on the resampled data lets synthetic samples derived from a validation fold's neighbors leak into the training folds, inflating CV scores.
# Wrapping SMOTE + the classifier in an imblearn Pipeline fixes this: cross_val_score re-fits SMOTE fresh on each fold's training split only.
smote_pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('rf', RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    ))
])

# Perform Stratified 5-Fold Cross Validation with leakage-safe SMOTE
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(smote_pipeline, X_train_scaled, y_train, cv=cv, scoring='accuracy')

print(f"\nK-Fold Cross Validation Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Fit the pipeline on the full training set (SMOTE runs once here, only on training data) and evaluate on the untouched, non-SMOTEd test set.
# imblearn pipelines only apply the resampler during .fit(), not .predict(),so this single predict() call is all that's needed.
smote_pipeline.fit(X_train_scaled, y_train)
y_pred_smote_test = smote_pipeline.predict(X_test_scaled)

# Evaluate performance
smote_test_accuracy = accuracy_score(y_test, y_pred_smote_test)
print(f"SMOTE Model Test Accuracy: {smote_test_accuracy:.4f}\n")

print("SMOTE Classification Report:")
print(classification_report(y_test, y_pred_smote_test, target_names=['Low', 'Medium', 'High']))

# Generate the SMOTE Confusion Matrix
cm_smote = confusion_matrix(y_test, y_pred_smote_test)
disp_smote = ConfusionMatrixDisplay(
    confusion_matrix=cm_smote,
    display_labels=['Low', 'Medium', 'High']
)

# 17. CONFUSION MATRIX COMPARISON

print("\n" + "=" * 70)
print("CONFUSION MATRIX COMPARISON: BASELINE vs SMOTE")
print("=" * 70)

# One figure, two subplots side by side, so the two models' error patterns can be scanned at a glance instead of switching between separate figures
fig, (ax_rf, ax_smote) = plt.subplots(1, 2, figsize=(14, 6))

disp_rf.plot(cmap='Blues', ax=ax_rf, values_format='d', colorbar=False)
ax_rf.set_title('Baseline (Class-Weighted)', fontsize=12, fontweight='bold')

disp_smote.plot(cmap='Blues', ax=ax_smote, values_format='d', colorbar=False)
ax_smote.set_title('SMOTE-Balanced', fontsize=12, fontweight='bold')

fig.suptitle('Random Forest Confusion Matrices', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix_comparison.png', dpi=300)
plt.show()

# 18. CONCLUSION

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)

print(f"\nBaseline Random Forest  - Test Accuracy : {rf_accuracy:.4f}")
print(f"SMOTE Random Forest     - Test Accuracy : {smote_test_accuracy:.4f}")
print(f"SMOTE Random Forest     - Mean CV Accuracy : {cv_scores.mean():.4f} "
      f"(+/- {cv_scores.std() * 2:.4f})")

better_model = "SMOTE-balanced" if smote_test_accuracy > rf_accuracy else "baseline (class-weighted)"
print(f"\nBetter-performing model on the held-out test set: {better_model} Random Forest.")

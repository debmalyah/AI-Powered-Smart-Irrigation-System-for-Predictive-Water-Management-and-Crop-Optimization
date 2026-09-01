import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#dataset1_path = r'C:\Users\Mayukh\PycharmProjects\Infosys_Internship\data\cropdata_updated.csv'
dataset2_path = r'C:\Users\Mayukh\PycharmProjects\Infosys_Internship\data\irrigation_prediction.csv'

#df_crop = pd.read_csv(dataset1_path)
df = pd.read_csv(dataset2_path)

#information about the second dataset
print("=== DATASET 2: SECOND DATASET ===")
print(f"Dimensions: {df.shape[0]} rows, {df.shape[1]} columns")
print("\nFirst 3 rows:")
print(df.head(3).T)
print("\nColumn Information:")
df.info()



print("\n=== 1. CATEGORICAL CARDINALITY & UNIQUE VALUES ===")
categorical_cols = df.select_dtypes(include=['object', 'string']).columns
for col in categorical_cols:
    print(f"• {col}: {df[col].nunique()} unique values -> {df[col].unique()[:5]}")

print("\n=== 2. NUMERICAL FEATURE DISTRIBUTIONS ===")
num_cols = df.select_dtypes(include=[np.number]).columns
print(df[num_cols].describe().T[['mean', 'std', 'min', '50%', 'max']])

print("\n=== 3. TARGET CLASS DISTRIBUTION ===")
print(df['Irrigation_Need'].value_counts(normalize=True).round(4) * 100)

# Quick conversion for correlation checking
df_corr = df.copy()
df_corr['Target_Numeric'] = df_corr['Irrigation_Need'].map({'Low': 0, 'Medium': 1, 'High': 2})
numeric_df = df_corr.select_dtypes(include=[np.number])

print("\n=== 4. CORRELATION WITH IRRIGATION NEED ===")
correlations = numeric_df.corr()['Target_Numeric'].drop('Target_Numeric').sort_values(ascending=False)
print(correlations.round(3))



# Set visual style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 10})

# Load Dataset 2
df = pd.read_csv('data/irrigation_prediction.csv')
df.columns = df.columns.str.strip()

# Order for target variable consistency
target_order = ['Low', 'Medium', 'High']

# Create 2x2 Multi-Panel Diagnostic Plot
fig, axes = plt.subplots(2, 2, figsize=(15, 11))

#  Target Class Distribution
sns.countplot(
    data=df, x='Irrigation_Need', order=target_order,
    ax=axes[0, 0], palette='Blues_r'
)
axes[0, 0].set_title('1. Target Class Distribution (Irrigation Need)', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Irrigation Need Level')
axes[0, 0].set_ylabel('Field Count')

# Soil Moisture vs Target Level
sns.boxplot(
    data=df, x='Irrigation_Need', y='Soil_Moisture', order=target_order,
    ax=axes[0, 1], palette='YlGn'
)
axes[0, 1].set_title('2. Soil Moisture Depletion vs. Irrigation Need', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Irrigation Need Level')
axes[0, 1].set_ylabel('Soil Moisture (%)')

#  Temperature vs Humidity Scatter
sns.scatterplot(
    data=df, x='Temperature_C', y='Humidity', hue='Irrigation_Need',
    hue_order=target_order, ax=axes[1, 0], palette='coolwarm', alpha=0.7
)
axes[1, 0].set_title('3. Microclimate: Temperature vs. Humidity', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Temperature (°C)')
axes[1, 0].set_ylabel('Humidity (%)')

#  Crop Type vs Irrigation Requirement Proportions
crop_target_props = pd.crosstab(df['Crop_Type'], df['Irrigation_Need'], normalize='index')[target_order]
crop_target_props.plot(
    kind='bar', stacked=True, ax=axes[1, 1], colormap='Spectral', edgecolor='black'
)
axes[1, 1].set_title('4. Irrigation Need Proportion by Crop Type', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Crop Type')
axes[1, 1].set_ylabel('Proportion')
axes[1, 1].tick_params(axis='x', rotation=30)
axes[1, 1].legend(title='Irrigation Need', bbox_to_anchor=(1.02, 1), loc='upper left')

plt.tight_layout()
plt.savefig('data_exploration_dashboard.png', dpi=300)
plt.show()

#  Feature Correlation Heatmap
plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_df.corr()

sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='vlag', center=0, linewidths=0.5)
plt.title('Numeric Feature Correlation Matrix', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300)
plt.show()
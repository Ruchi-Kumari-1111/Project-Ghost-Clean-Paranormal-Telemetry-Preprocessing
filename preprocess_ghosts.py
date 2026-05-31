import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ==========================================
# 1. IMPORT & EXPLORE THE DATA [cite: 7]
# ==========================================
df = pd.read_csv('haunted_telemetry.csv')
print("--- Initial Spooky Data Exploration ---")
print(df.info())
print("\nMissing Values Count:\n", df.isnull().sum())

# ==========================================
# 2. HANDLE MISSING VALUES (IMPUTATION) [cite: 8]
# ==========================================
# Amityville (LOC_004) has a missing EMF reading. Fill it with the median.
df['EMF_Field_Strength'] = df['EMF_Field_Strength'].fillna(df['EMF_Field_Strength'].median())

# Estes Park (LOC_005) has a missing Ghost Activity Score. Fill it with the mean.
df['Ghost_Activity_Score'] = df['Ghost_Activity_Score'].fillna(df['Ghost_Activity_Score'].mean())

# Alton (LOC_007) has a missing Entity Type. Fill it with a placeholder.
df['Entity_Type'] = df['Entity_Type'].fillna('Unidentified Entity')

# ==========================================
# 3. VISUALIZE & REMOVE OUTLIERS (UPGRADED DASHBOARD) [cite: 11]
# ==========================================
# Create a beautiful, two-row dashboard layout
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
plt.subplots_adjust(hspace=0.5) # Add clear breathing space between charts

# --- Chart 1: Professional Boxplot ---
sns.boxplot(
    x=df['EMF_Field_Strength'], 
    ax=ax1, 
    color='#9b5de5', # Paranormal purple theme
    flierprops={"markerfacecolor": "red", "markersize": 10, "marker": "o"} # Outliers stand out in bright RED!
)
ax1.set_title("⚠️ Ghost Sensor Anomaly Detection (Pre-Cleaning)", fontsize=13, fontweight='bold', pad=10)
ax1.set_xlabel("EMF Field Strength Readings (Microtesla)", fontsize=11)
ax1.grid(axis='x', linestyle='--', alpha=0.5) # Add subtle background grids

# Adding text labels right inside the graph window so it's instantly understandable
ax1.text(15, 0.3, "Normal Sensor Cluster\n(Expected Haunted Signals)", color="purple", fontsize=9, fontweight="bold")
ax1.text(850, -0.15, "🚨 CORRUPTED OUTLIER!\n(Portland Tunnel Glitch)", color="red", fontsize=9, fontweight="bold")

# --- Outlier Removal Logic Execution ---
Q1 = df['EMF_Field_Strength'].quantile(0.25)
Q3 = df['EMF_Field_Strength'].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR

df_cleaned = df[df['EMF_Field_Strength'] <= upper_bound].copy()
print(f"\n[Success] Ghostly outliers purged! Rows reduced from {len(df)} to {len(df_cleaned)}")

# ==========================================
# 4. CATEGORICAL ENCODING [cite: 9]
# ==========================================
# Convert the textual 'Entity_Type' column into numbers for ML processing
encoder = LabelEncoder()
df_cleaned['Entity_Type_Encoded'] = encoder.fit_transform(df_cleaned['Entity_Type'])

# ==========================================
# 5. FEATURE SCALING (STANDARDIZATION) [cite: 10]
# ==========================================
# Standardize numerical features to level out variations in data metrics
scaler = StandardScaler()
df_cleaned[['EMF_Scaled', 'Activity_Scaled']] = scaler.fit_transform(
    df_cleaned[['EMF_Field_Strength', 'Ghost_Activity_Score']]
)

# --- Chart 2: Correlation Heatmap Matrix ---
# Calculate the connection between variables post-cleaning
correlation_matrix = df_cleaned[['EMF_Field_Strength', 'Ghost_Activity_Score', 'Entity_Type_Encoded']].corr()

sns.heatmap(
    correlation_matrix, 
    annot=True, 
    ax=ax2,
    cmap='Purples', 
    fmt=".2f", 
    linewidths=1, 
    cbar=True
)
ax2.set_title("🔮 Spooky Feature Correlation Matrix (Post-Cleaning)", fontsize=13, fontweight='bold', pad=10)

# Save the master analytics visual file to your local workspace
plt.savefig('ghost_analysis_dashboard.png', bbox_inches='tight', dpi=300)
print("[Graphics Status] Dashboard saved to workspace as 'ghost_analysis_dashboard.png'!")

# Display the data tables and launch the active visual graphs
print("\n--- Final Preprocessed Telemetry Dataset ---")
print(df_cleaned[['Location_ID', 'City', 'Entity_Type_Encoded', 'EMF_Scaled', 'Activity_Scaled']])
plt.show()
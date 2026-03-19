import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ==============================================================
# 1. LOAD DATA
# ==============================================================

df = pd.read_excel(
    "C:/Users/Owner/OneDrive/Download1/Research STA199/F1_2024_Full_Dataset_Predictors_DashboardLogic_24Races (1).xlsx"
)

print("Data loaded successfully.")

# ==============================================================
# 2. CLEAN DATA (same preprocessing as model)
# ==============================================================

# Structural zeros
zero_fill = ["TeamStrength", "DriverDNF_Rate", "TeamDNF_Rate"]
df[zero_fill] = df[zero_fill].fillna(0)

# DriverForm imputation (race-level mean)
df["DriverForm"] = (
    df.groupby(["Year", "Round"])["DriverForm"]
      .transform(lambda x: x.fillna(x.mean()))
)

# OvertakeIndex imputation (race-level mean)
df["OvertakeIndex"] = (
    df.groupby(["Year", "Round"])["OvertakeIndex"]
      .transform(lambda x: x.fillna(x.mean()))
)

# Drop missing qualifying
df = df.dropna(subset=["QualifyingPosition"])

print("Cleaning complete.")

# ==============================================================
# 3. CONVERT QUALIFYING TO ORDINAL
# ==============================================================

def bin_quali(pos):
    if pos <= 3:
        return 1   # Front
    elif pos <= 10:
        return 2   # Mid
    else:
        return 3   # Back

df["QualiBinOrdinal"] = df["QualifyingPosition"].apply(bin_quali)

# ==============================================================
# 4. SELECT FEATURES FOR PCA
# ==============================================================

continuous_predictors = [
    "DriverForm",
    "TeamStrength",
    "DriverDNF_Rate",
    "TeamDNF_Rate",
    "OvertakeIndex"
]

pca_features = ["QualiBinOrdinal"] + continuous_predictors

X_raw = df[pca_features].copy()

print("\nMissing values check:")
print(X_raw.isna().sum())

# ==============================================================
# 5. STANDARDIZE (REQUIRED FOR PCA)
# ==============================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

print("\nData standardized for PCA.")

# ==============================================================
# 6. RUN PCA
# ==============================================================

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# ==============================================================
# 7. EXPLAINED VARIANCE
# ==============================================================

explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

print("\nExplained variance per principal component:")
for i, var in enumerate(explained_variance_ratio):
    print(f"PC{i+1}: {var:.4f}")

print("\nCumulative explained variance:")
for i, var in enumerate(cumulative_variance):
    print(f"PC{i+1}: {var:.4f}")

# ==============================================================
# 8. SELECT COMPONENTS (95% THRESHOLD)
# ==============================================================

n_components = np.argmax(cumulative_variance >= 0.95) + 1

print(f"\nNumber of components explaining 95% variance: {n_components}")

# Reduced matrix
X_reduced = X_pca[:, :n_components]

print(f"\nReduced feature matrix shape: {X_reduced.shape}")

print("\nPCA Loadings:")
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f"PC{i+1}" for i in range(len(pca_features))],
    index=pca_features
)
print(loadings)
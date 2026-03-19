import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
from scipy.stats import spearmanr, kendalltau

# ==============================
# 1. LOAD + PREPROCESS
# ==============================

df = pd.read_excel(
    "C:/Users/Owner/OneDrive/Download1/Research STA199/F1_2024_Full_Dataset_Predictors_DashboardLogic_24Races (1).xlsx"
)

zero_fill = ["TeamStrength","DriverDNF_Rate","TeamDNF_Rate"]
df[zero_fill] = df[zero_fill].fillna(0)

df["DriverForm"] = (
    df.groupby(["Year","Round"])["DriverForm"]
      .transform(lambda x: x.fillna(x.mean()))
)

df["OvertakeIndex"] = (
    df.groupby(["Year","Round"])["OvertakeIndex"]
      .transform(lambda x: x.fillna(x.mean()))
)

df = df.dropna(subset=["QualifyingPosition"])

# ==============================
# 2. QUALIFYING ORDINAL (FOR PCA)
# ==============================

def bin_quali(pos):
    if pos <= 3:
        return 1
    elif pos <= 10:
        return 2
    else:
        return 3

df["QualiBinOrdinal"] = df["QualifyingPosition"].apply(bin_quali)

# ==============================
# 3. FEATURES FOR PCA
# ==============================

continuous_predictors = [
    "DriverForm",
    "TeamStrength",
    "DriverDNF_Rate",
    "TeamDNF_Rate",
    "OvertakeIndex"
]

pca_features = ["QualiBinOrdinal"] + continuous_predictors

X_raw = df[pca_features].copy()

# ==============================
# 4. STANDARDIZE
# ==============================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# ==============================
# 5. RUN PCA
# ==============================

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

explained_variance = pca.explained_variance_ratio_
cum_var = np.cumsum(explained_variance)

n_components = np.argmax(cum_var >= 0.95) + 1

print("Components kept:", n_components)

X_reduced = X_pca[:, :n_components]

# Add PCs back to dataframe
for i in range(n_components):
    df[f"PC{i+1}"] = X_reduced[:, i]

pc_predictors = [f"PC{i+1}" for i in range(n_components)]

# ==============================
# 6. BUILD MODEL DATASET
# ==============================

model_df = df[["Year","Round","Driver","RacePosition"] + pc_predictors].copy()

# ==============================
# 7. BUILD RACE STRUCTURE
# ==============================

races = []

for (year, round_), g in model_df.groupby(["Year","Round"]):
    
    g = g.sort_values("RacePosition")
    
    X = g[pc_predictors].astype(float).to_numpy()
    
    order = np.arange(len(g))

    races.append({
        "year": year,
        "round": round_,
        "X": X,
        "order": order
    })

print(f"Total races: {len(races)}")

# ==============================
# 8. TRAIN / TEST SPLIT
# ==============================

train_races, test_races = train_test_split(
    races,
    test_size=0.30,
    random_state=42
)

print(f"Train races: {len(train_races)}")
print(f"Test races: {len(test_races)}")

# ==============================
# 9. PLACKETT-LUCE L2 LIKELIHOOD
# ==============================

def pl_neg_log_likelihood_l2(beta, races, lambda_l2=1.0):

    ll = 0.0
    
    for race in races:
        
        X = race["X"]
        order = race["order"]
        
        theta = X @ beta
        
        remaining = list(order)

        for i in order:
            
            theta_rem = theta[remaining]
            
            theta_rem -= np.max(theta_rem)
            
            ll += theta[i] - np.log(np.sum(np.exp(theta_rem)))
            
            remaining.remove(i)

    penalty = lambda_l2 * np.sum(beta**2)
    
    return -(ll - penalty)

# ==============================
# 10. TRAIN MODEL
# ==============================

p = len(pc_predictors)

beta0 = np.zeros(p)

lambda_l2 = 1.0

res = minimize(
    pl_neg_log_likelihood_l2,
    beta0,
    args=(train_races, lambda_l2),
    method="BFGS"
)

beta_hat = res.x

print("Model trained.")

# ==============================
# 11. TEST EVALUATION
# ==============================

def pl_loglik(beta, races):

    ll = 0.0
    
    for race in races:
        
        X = race["X"]
        order = race["order"]
        
        theta = X @ beta
        
        remaining = list(order)

        for i in order:
            
            theta_rem = theta[remaining]
            
            theta_rem -= np.max(theta_rem)
            
            ll += theta[i] - np.log(np.sum(np.exp(theta_rem)))
            
            remaining.remove(i)

    return ll

test_loglik = pl_loglik(beta_hat, test_races)

print("Test Log-Likelihood:", test_loglik)

# ==============================
# 12. RANK METRICS
# ==============================

spearman_scores = []
kendall_scores = []
top3_scores = []

for race in test_races:

    X = race["X"]
    
    true_order = race["order"]
    
    pred_order = np.argsort(-(X @ beta_hat))

    spearman_scores.append(
        spearmanr(true_order, pred_order).correlation
    )

    kendall_scores.append(
        kendalltau(true_order, pred_order).correlation
    )

    pred_top3 = set(pred_order[:3])
    
    true_top3 = set(true_order[:3])
    
    top3_scores.append(len(pred_top3 & true_top3)/3)

print("Test Mean Spearman:", np.nanmean(spearman_scores))
print("Test Mean Kendall:", np.nanmean(kendall_scores))
print("Test Top-3 Accuracy:", np.mean(top3_scores))
"""
PL mcfadden r2 v2.py
McFadden's adjusted pseudo-R² and coefficient tables for 4 PL model variants:
  Full (18 features), VIF (15 features), BIC (4 features), AIC (5 features).

All fits are on the 2022-2024 reference dataset (no 2025 leakage).

McFadden pseudo-R²     = 1 - log_L_model / log_L_null
McFadden adjusted R²   = 1 - (log_L_model - k) / log_L_null
  log_L_null = -sum_races log(k_i!)   (uniform null: all drivers equally likely)
  k          = number of model parameters

Outputs (saved to src/):
  pl_v2_mcfadden.csv       — R², adj-R², log-likelihood, k per model
  pl_v2_coef_ref_full.csv  — coefficients + p-values, full 18-feature fit
  pl_v2_coef_ref_vif.csv   — coefficients + p-values, VIF 15-feature fit
  pl_v2_coef_ref_bic.csv   — coefficients + p-values, BIC 4-feature fit
  pl_v2_coef_ref_aic.csv   — coefficients + p-values, AIC 5-feature fit
"""
import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler

DATA_PATH = (
    "C:/Users/Owner/OneDrive/Download1/Research STA199/"
    "DEBUGF7_f1_enriched_prediction_dataset_2022_2025.csv"
)

ALL_FEATURES = [
    "QualiPercentile", "QualiGapToPole", "TeammateQualiGap",
    "FP_LongRunPace", "FP_LongRunVar", "DriverForm", "TeamStrength",
    "DriverDNFRate", "TeamDNFRate", "AirTemp", "TrackTemp", "Rainfall",
    "OvertakeIndex", "Quali_x_Overtake", "Reliability_x_Rain",
    "OvertakeOpportunity", "Driver_vs_Team", "GridSpread",
]

LAMBDA_L2 = 1.0


# ── Data loading ─────────────────────────────────────────────────────────────

def load_and_prep(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["FinishPosition"])
    if "QualiPosition" in df.columns:
        n_starters = df.groupby("RaceID")["Driver"].transform("count")
        df["QualiPercentile"] = (df["QualiPosition"] - 1) / (n_starters - 1).clip(lower=1)
        df["Quali_x_Overtake"] = df["QualiPercentile"] * df["OvertakeIndex"]
        df["OvertakeOpportunity"] = (1 - df["QualiPercentile"]) * df["OvertakeIndex"]
    return df


def build_races(df, features):
    races = []
    for race_id, g in df.groupby("RaceID"):
        g = g.sort_values("FinishPosition").copy()
        races.append({
            "race_id": race_id,
            "k": len(g),
            "drivers": g["Driver"].values,
            "X": g[features].astype(float).to_numpy(),
            "order": np.arange(len(g)),
        })
    return races


# ── PL likelihood (vectorized, with analytical gradient) ────────────────────

def _pl_val_grad(beta, races, lambda_l2):
    ll = 0.0
    grad = np.zeros_like(beta)
    for race in races:
        X = race["X"]
        theta = X @ beta
        lse = np.logaddexp.accumulate(theta[::-1])[::-1]
        ll += np.sum(theta - lse)
        inv_Z = np.exp(-lse)
        c = np.exp(theta) * np.cumsum(inv_Z)
        grad += X.T @ (1.0 - c)
    neg_ll = -ll + lambda_l2 * np.sum(beta ** 2)
    neg_grad = -grad + 2.0 * lambda_l2 * beta
    return neg_ll, neg_grad


def pl_loglik(beta, races):
    ll = 0.0
    for race in races:
        theta = race["X"] @ beta
        lse = np.logaddexp.accumulate(theta[::-1])[::-1]
        ll += np.sum(theta - lse)
    return ll


def fit_pl(races, n_features, lambda_l2=LAMBDA_L2):
    res = minimize(
        _pl_val_grad,
        np.zeros(n_features),
        args=(races, lambda_l2),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 400, "ftol": 1e-9},
    )
    return res.x, pl_loglik(res.x, races), res


# ── Null log-likelihood ───────────────────────────────────────────────────────

def null_loglik(races):
    """log L_null = -sum_races log(k_i!) where k_i = number of starters."""
    total = 0.0
    for race in races:
        total -= math.lgamma(race["k"] + 1)   # lgamma(n+1) = log(n!)
    return total


# ── McFadden pseudo-R² ────────────────────────────────────────────────────────

def mcfadden_r2(ll_model, ll_null, k):
    r2     = 1.0 - ll_model / ll_null
    r2_adj = 1.0 - (ll_model - k) / ll_null
    return r2, r2_adj


# ── Unregularized Fisher information matrix ──────────────────────────────────

def pl_fisher_info(beta, races):
    """
    Fisher information = negative Hessian of the UNREGULARIZED log-likelihood.
    Used for valid SE and p-value computation; not affected by L2 penalty.
    For each race, sums the variance of X under the softmax distribution at
    each sequential selection step.
    """
    p = len(beta)
    F = np.zeros((p, p))
    for race in races:
        X = race["X"]            # (k, p)
        theta = X @ beta         # (k,)
        lse = np.logaddexp.accumulate(theta[::-1])[::-1]
        for i in range(len(theta)):
            probs = np.exp(theta[i:] - lse[i])   # softmax over remaining
            Xi = X[i:]
            mu = Xi.T @ probs                     # (p,) expected feature vector
            F += (Xi * probs[:, np.newaxis]).T @ Xi - np.outer(mu, mu)
    return F   # positive semi-definite


# ── Coefficient extraction ───────────────────────────────────────────────────

def extract_coefs(features, beta, races):
    """
    Computes SE from unregularized Fisher information (valid frequentist SEs).
    Falls back gracefully if Fisher info is singular (collinear features).
    """
    F = pl_fisher_info(beta, races)
    rows = []
    try:
        # Use pseudoinverse to handle near-singular cases (collinear features)
        cov = np.linalg.pinv(F)
        diag = np.diag(cov)
        # Negative or near-zero diag entries indicate rank deficiency
        se = np.where(diag > 0, np.sqrt(diag), np.nan)
        z  = np.where(np.isfinite(se), beta / se, np.nan)
        pv = np.where(np.isfinite(z), 2.0 * (1.0 - norm.cdf(np.abs(z))), np.nan)
        sig = []
        for p_val in pv:
            if not np.isfinite(p_val):
                sig.append("")
            elif p_val < 0.001: sig.append("***")
            elif p_val < 0.01:  sig.append("**")
            elif p_val < 0.05:  sig.append("*")
            elif p_val < 0.1:   sig.append(".")
            else:               sig.append("")
        for feat, coef, stderr, z_val, p_val, s in zip(features, beta, se, z, pv, sig):
            rows.append({
                "Feature":  feat,
                "Coef":     round(float(coef),    5),
                "SE":       round(float(stderr),  5) if np.isfinite(stderr) else None,
                "Z":        round(float(z_val),   3) if np.isfinite(z_val)  else None,
                "P_value":  round(float(p_val),   4) if np.isfinite(p_val)  else None,
                "Sig":      s,
            })
    except Exception:
        for feat, coef in zip(features, beta):
            rows.append({"Feature": feat, "Coef": round(float(coef), 5),
                         "SE": None, "Z": None, "P_value": None, "Sig": ""})
    return pd.DataFrame(rows)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    df = load_and_prep(DATA_PATH)
    ref_df = df[df["Year"].isin([2022, 2023, 2024])].copy()
    print(f"Reference data: {len(ref_df)} rows, {ref_df['RaceID'].nunique()} races")

    # Scale features (same as walk-forward: scale all except QualiPercentile)
    scale_cols = [f for f in ALL_FEATURES if f != "QualiPercentile"]
    scaler = StandardScaler()
    ref_scaled = ref_df.copy()
    ref_scaled[scale_cols] = scaler.fit_transform(ref_scaled[scale_cols])

    # Feature sets
    vif_features = list(pd.read_csv("pl_v2_vif_final.csv")["Feature"])
    bic_features = list(pd.read_csv("pl_v2_bic_features.csv")["Feature"])
    aic_features = list(pd.read_csv("pl_v2_aic_features.csv")["Feature"])

    model_defs = [
        ("Full (18 features)", ALL_FEATURES),
        ("VIF (15 features)",  vif_features),
        ("BIC (4 features)",   bic_features),
        ("AIC (5 features)",   aic_features),
    ]

    # Compute null log-likelihood once (same races for all models)
    races_full = build_races(ref_scaled, ALL_FEATURES)
    ll_null = null_loglik(races_full)
    print(f"Null log-likelihood: {ll_null:.3f}\n")

    r2_rows = []
    coef_dfs = {}

    for name, features in model_defs:
        print(f"Fitting: {name}")
        races = build_races(ref_scaled, features)
        beta, ll_model, res = fit_pl(races, len(features))
        r2, r2_adj = mcfadden_r2(ll_model, ll_null, len(features))
        coef_df = extract_coefs(features, beta, races)

        print(f"  log L_model = {ll_model:.3f}  |  R² = {r2:.4f}  |  adj-R² = {r2_adj:.4f}")
        for _, row in coef_df.iterrows():
            sig = row['Sig'] if row['Sig'] else ""
            p_str = f"p={row['P_value']:.4f}" if row['P_value'] is not None else "p=N/A"
            print(f"    {row['Feature']:<25} coef={row['Coef']:+.4f}  {p_str}  {sig}")

        r2_rows.append({
            "Model": name,
            "k": len(features),
            "LogL_model": round(ll_model, 3),
            "LogL_null": round(ll_null, 3),
            "McFadden_R2": round(r2, 4),
            "McFadden_R2_adj": round(r2_adj, 4),
        })
        coef_dfs[name] = coef_df
        print()

    # Save
    pd.DataFrame(r2_rows).to_csv("pl_v2_mcfadden.csv", index=False)
    coef_dfs["Full (18 features)"].to_csv("pl_v2_coef_ref_full.csv", index=False)
    coef_dfs["VIF (15 features)"].to_csv("pl_v2_coef_ref_vif.csv", index=False)
    coef_dfs["BIC (4 features)"].to_csv("pl_v2_coef_ref_bic.csv", index=False)
    coef_dfs["AIC (5 features)"].to_csv("pl_v2_coef_ref_aic.csv", index=False)
    print("Saved: pl_v2_mcfadden.csv + pl_v2_coef_ref_*.csv")


if __name__ == "__main__":
    main()
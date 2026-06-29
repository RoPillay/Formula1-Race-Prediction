import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import norm, spearmanr, kendalltau
from sklearn.metrics import ndcg_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

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

YEARS     = [2022, 2023, 2024, 2025]
LAMBDA_L2 = 1.0


# ==============================================================
# DATA LOADING
# ==============================================================

def load_and_prep(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["FinishPosition"])
    if "QualiPosition" in df.columns:
        n_starters = df.groupby("RaceID")["Driver"].transform("count")
        df["QualiPercentile"] = (df["QualiPosition"] - 1) / (n_starters - 1).clip(lower=1)
        df["Quali_x_Overtake"] = df["QualiPercentile"] * df["OvertakeIndex"]
        df["OvertakeOpportunity"] = (1 - df["QualiPercentile"]) * df["OvertakeIndex"]
    return df


# ==============================================================
# WALK-FORWARD SPLITS
# ==============================================================

def walkforward_splits(df, years):
    """
    5-fold scheme combining rolling single-year and expanding cumulative windows:
      Rolling:    [y] -> y+1  for each consecutive year pair
      Expanding:  [2022..y] -> y+1  for 2-year and 3-year training blocks
    """
    splits = []
    seen = set()

    # Rolling single-year: train on one year, test on the next
    for i in range(len(years) - 1):
        tr_yrs  = [years[i]]
        te_year = years[i + 1]
        key = (tuple(tr_yrs), te_year)
        if key not in seen:
            train_idx = df.index[df["Year"].isin(tr_yrs)].tolist()
            test_idx  = df.index[df["Year"] == te_year].tolist()
            if train_idx and test_idx:
                splits.append((train_idx, test_idx, tr_yrs, te_year))
                seen.add(key)

    # Expanding cumulative: train on all years up to y, test on y+1
    for i in range(2, len(years)):
        tr_yrs  = years[:i]
        te_year = years[i]
        key = (tuple(tr_yrs), te_year)
        if key not in seen:
            train_idx = df.index[df["Year"].isin(tr_yrs)].tolist()
            test_idx  = df.index[df["Year"] == te_year].tolist()
            if train_idx and test_idx:
                splits.append((train_idx, test_idx, tr_yrs, te_year))
                seen.add(key)

    return splits


# ==============================================================
# RACE BUILDING
# ==============================================================

def build_races(df, features):
    races = []
    for race_id, g in df.groupby("RaceID"):
        g = g.sort_values("FinishPosition").copy()
        races.append({
            "race_id": race_id,
            "year": g["Year"].iloc[0],
            "drivers": g["Driver"].values,
            "finish_pos": g["FinishPosition"].values,
            "X": g[features].astype(float).to_numpy(),
            "order": np.arange(len(g)),
        })
    return races


# ==============================================================
# PL LIKELIHOOD  (L-BFGS-B used everywhere — no optimizer inconsistency)
# ==============================================================

def pl_neg_loglik_l2(beta, races, lambda_l2):
    ll = 0.0
    for race in races:
        theta = race["X"] @ beta
        remaining = list(range(len(theta)))
        for i in race["order"]:
            ll += theta[i] - logsumexp(theta[remaining])
            remaining.remove(i)
    return -ll + lambda_l2 * np.sum(beta**2)


def pl_loglik(beta, races):
    ll = 0.0
    for race in races:
        theta = race["X"] @ beta
        remaining = list(range(len(theta)))
        for i in race["order"]:
            ll += theta[i] - logsumexp(theta[remaining])
            remaining.remove(i)
    return ll


def fit_pl(races, n_features, lambda_l2=LAMBDA_L2):
    res = minimize(
        pl_neg_loglik_l2,
        np.zeros(n_features),
        args=(races, lambda_l2),
        method="L-BFGS-B",
        options={"maxiter": 200, "ftol": 1e-6},
    )
    return res.x, pl_loglik(res.x, races), res


# ==============================================================
# AIC / BIC  (n = race count, not row count)
# ==============================================================

def aic_bic(loglik, k, n_races):
    aic = -2 * loglik + 2 * k
    bic = -2 * loglik + k * np.log(n_races)
    return aic, bic


# ==============================================================
# FEATURE SELECTION — METHOD 1: BIC BACKWARD
# ==============================================================

def bic_backward_selection(train_df, features, lambda_l2=LAMBDA_L2):
    current = features.copy()
    n_races = train_df["RaceID"].nunique()

    races = build_races(train_df, current)
    _, loglik, _ = fit_pl(races, len(current), lambda_l2)
    _, best_bic = aic_bic(loglik, len(current), n_races)

    while len(current) > 1:
        candidates = []
        for feat in current:
            trial = [f for f in current if f != feat]
            r = build_races(train_df, trial)
            _, ll, _ = fit_pl(r, len(trial), lambda_l2)
            _, bic = aic_bic(ll, len(trial), n_races)
            candidates.append((feat, bic))

        best_feat, best_bic_candidate = min(candidates, key=lambda x: x[1])
        if best_bic_candidate < best_bic:
            print(f"    [BIC] Removing {best_feat}: {best_bic:.2f} -> {best_bic_candidate:.2f}")
            current.remove(best_feat)
            best_bic = best_bic_candidate
        else:
            break

    return current, best_bic


# ==============================================================
# FEATURE SELECTION — METHOD 2: AIC STEPWISE (BIDIRECTIONAL)
# ==============================================================

def aic_stepwise_selection(train_df, features, lambda_l2=LAMBDA_L2):
    # Forward-only stepwise: greedily add features while AIC improves.
    # Bidirectional is avoided because the flat log-likelihood surface with
    # many features / few races causes infinite oscillation via float noise.
    current = []
    remaining = features.copy()
    n_races = train_df["RaceID"].nunique()
    best_score = np.inf

    while remaining:
        candidates = []
        for feat in remaining:
            trial = current + [feat]
            r = build_races(train_df, trial)
            _, ll, _ = fit_pl(r, len(trial), lambda_l2)
            aic, _ = aic_bic(ll, len(trial), n_races)
            candidates.append((feat, aic))

        best_feat, best_aic = min(candidates, key=lambda x: x[1])
        if best_aic < best_score:
            best_score = best_aic
            print(f"    [AIC] Adding {best_feat}: AIC -> {best_score:.2f}")
            current.append(best_feat)
            remaining.remove(best_feat)
        else:
            break

    return current, best_score


# ==============================================================
# FEATURE SELECTION — METHOD 3: VIF BACKWARD
# ==============================================================

def vif_backward_selection(train_df, features, threshold=5.0):
    """Returns (selected_features, steps_df, final_vif_df)."""
    X = train_df[features].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)

    current = features.copy()
    steps = []
    step_num = 1
    while True:
        vif_vals = [
            (f, variance_inflation_factor(X_scaled[current].values, i))
            for i, f in enumerate(current)
        ]
        vif_df = pd.DataFrame(vif_vals, columns=["Feature", "VIF"]).sort_values(
            "VIF", ascending=False
        )
        worst = vif_df.iloc[0]
        if worst["VIF"] < threshold:
            break
        print(f"    [VIF] Removing {worst['Feature']} (VIF={worst['VIF']:.2f})")
        steps.append({
            "Step": step_num,
            "Feature_Removed": worst["Feature"],
            "VIF_at_Removal": round(float(worst["VIF"]), 3),
            "Features_Remaining": len(current) - 1,
        })
        current.remove(worst["Feature"])
        step_num += 1

    final_vif_df = pd.DataFrame(
        [(f, float(variance_inflation_factor(X_scaled[current].values, i)))
         for i, f in enumerate(current)],
        columns=["Feature", "VIF"]
    ).sort_values("VIF", ascending=False).reset_index(drop=True)

    steps_df = pd.DataFrame(steps) if steps else pd.DataFrame(
        columns=["Step", "Feature_Removed", "VIF_at_Removal", "Features_Remaining"])
    return current, steps_df, final_vif_df


# ==============================================================
# FIT AND EVALUATE A GIVEN FEATURE SET ON ONE FOLD
# ==============================================================

def fit_and_evaluate(train, test, selected, df_for_baseline, test_idx):
    scale_cols = [f for f in selected if f != "QualiPercentile"]
    scaler = StandardScaler()
    tr = train.copy()
    te = test.copy()
    if scale_cols:
        tr[scale_cols] = scaler.fit_transform(tr[scale_cols])
        te[scale_cols] = scaler.transform(te[scale_cols])

    train_races = build_races(tr, selected)
    test_races = build_races(te, selected)
    beta, _, res = fit_pl(train_races, len(selected))

    # Collect coefficients
    coef_rows = []
    try:
        h = res.hess_inv
        hess_arr = np.asarray(h.todense() if hasattr(h, "todense") else h)
        se = np.sqrt(np.diag(hess_arr))
        z = beta / se
        p = 2 * (1 - norm.cdf(np.abs(z)))
        for feat, coef, stderr, z_val, p_val in zip(selected, beta, se, z, p):
            coef_rows.append({
                "Feature": feat, "Coef": coef,
                "SE_approx": stderr, "Z": z_val, "P_approx": p_val,
            })
    except Exception:
        for feat, coef in zip(selected, beta):
            coef_rows.append({"Feature": feat, "Coef": coef})

    # Predictions
    pred_rows = []
    for race in test_races:
        scores = race["X"] @ beta
        for i, drv in enumerate(race["drivers"]):
            pred_rows.append({
                "RaceID": race["race_id"],
                "Driver": drv,
                "FinishPosition": race["finish_pos"][i],
                "PLScore": scores[i],
            })
    pred_df = pd.DataFrame(pred_rows)

    test_raw = df_for_baseline.loc[test_idx, ["RaceID", "Driver", "QualiPosition"]].copy()
    pred_df = pred_df.merge(test_raw, on=["RaceID", "Driver"], how="left")
    # Fill missing QualiPosition (DNS/DSQ) with worst grid + 1 per race
    worst = pred_df.groupby("RaceID")["QualiPosition"].transform(
        lambda x: x.max() + 1
    )
    pred_df["QualiPosition"] = pred_df["QualiPosition"].fillna(worst)
    pred_df["BaselineScore"] = -pred_df["QualiPosition"]

    pl_m = compute_metrics(pred_df, "PLScore")
    bl_m = compute_metrics(pred_df, "BaselineScore")
    return pl_m, bl_m, pd.DataFrame(coef_rows)


# ==============================================================
# EVALUATION
# ==============================================================

def compute_metrics(df, score_col):
    spearmans, kendalls = [], []
    top3s, top5s, winners = [], [], []
    ndcg3s, ndcg5s = [], []

    for _, g in df.groupby("RaceID"):
        if len(g) < 5:
            continue
        pred_rank = g[score_col].rank(ascending=False, method="first")
        actual_rank = g["FinishPosition"].rank(ascending=True, method="first")
        rho, _ = spearmanr(pred_rank, actual_rank)
        tau, _ = kendalltau(pred_rank, actual_rank)
        if np.isfinite(rho):
            spearmans.append(rho)
        if np.isfinite(tau):
            kendalls.append(tau)

        pred_top3 = set(g.nlargest(3, score_col)["Driver"])
        true_top3 = set(g.nsmallest(3, "FinishPosition")["Driver"])
        pred_top5 = set(g.nlargest(5, score_col)["Driver"])
        true_top5 = set(g.nsmallest(5, "FinishPosition")["Driver"])
        top3s.append(len(pred_top3 & true_top3) / 3)
        top5s.append(len(pred_top5 & true_top5) / 5)

        pred_winner = g.nlargest(1, score_col)["Driver"].iloc[0]
        true_winner = g.nsmallest(1, "FinishPosition")["Driver"].iloc[0]
        winners.append(int(pred_winner == true_winner))

        relevance = (21 - g["FinishPosition"]).values.reshape(1, -1)
        scores_2d = g[score_col].values.reshape(1, -1)
        if not np.all(np.isfinite(scores_2d)):
            continue
        ndcg3s.append(ndcg_score(relevance, scores_2d, k=3))
        ndcg5s.append(ndcg_score(relevance, scores_2d, k=5))

    return {
        "Spearman":   np.nanmean(spearmans),
        "KendallTau": np.nanmean(kendalls),
        "Top3 Acc":   np.nanmean(top3s),
        "Top5 Acc":   np.nanmean(top5s),
        "Winner Acc": np.nanmean(winners),
        "NDCG@3":     np.nanmean(ndcg3s),
        "NDCG@5":     np.nanmean(ndcg5s),
        "N_Races":    len(spearmans),
    }


# ==============================================================
# MAIN
# ==============================================================

def main():
    df = load_and_prep(DATA_PATH)
    print(f"Loaded {len(df)} rows, {df['RaceID'].nunique()} races, "
          f"years: {sorted(df['Year'].unique())}")

    splits = walkforward_splits(df, YEARS)
    print(f"\n{len(splits)} folds: " +
          ", ".join(f"{ty}->{t}" for _, _, ty, t in splits))

    all_fold_rows = []
    all_coef_rows = []

    for train_idx, test_idx, train_years, test_year in splits:
        train_label = "+".join(str(y) for y in train_years)
        print(f"\n{'='*70}")
        print(f"FOLD: train=[{train_label}]  ->  test={test_year}")
        train = df.loc[train_idx].copy()
        test  = df.loc[test_idx].copy()
        print(f"  Train: {train['RaceID'].nunique()} races  "
              f"| Test: {test['RaceID'].nunique()} races")

        pl_m, bl_m, coef_df = fit_and_evaluate(
            train, test, ALL_FEATURES, df, test_idx
        )

        print(f"\n  {'Metric':<14} {'Model':>9} {'Baseline':>9} {'Delta':>9}")
        for key in ["Spearman", "KendallTau", "Top3 Acc", "Top5 Acc",
                    "Winner Acc", "NDCG@3", "NDCG@5"]:
            delta = pl_m[key] - bl_m[key]
            sign = "+" if delta >= 0 else ""
            print(f"  {key:<14} {pl_m[key]:>9.4f} {bl_m[key]:>9.4f} "
                  f"{sign}{delta:>8.4f}")

        for label, m in [("Model", pl_m), ("Baseline", bl_m)]:
            all_fold_rows.append({
                "TrainYears": train_label,
                "TestYear":   test_year,
                "Method":     label,
                "N_Features": len(ALL_FEATURES) if label == "Model" else 1,
                **{k: m[k] for k in ["Spearman", "KendallTau", "Top3 Acc",
                                      "Top5 Acc", "Winner Acc", "NDCG@3",
                                      "NDCG@5", "N_Races"]},
            })

        for row in coef_df.to_dict("records"):
            row["TrainYears"] = train_label
            row["TestYear"]   = test_year
            all_coef_rows.append(row)

    summary_df  = pd.DataFrame(all_fold_rows)
    coef_df_all = pd.DataFrame(all_coef_rows)

    print(f"\n{'='*70}")
    print("MEAN ACROSS ALL FOLDS (Model vs Baseline)")
    mean_tbl = (
        summary_df.groupby("Method")[
            ["Spearman", "KendallTau", "Top3 Acc", "Top5 Acc",
             "Winner Acc", "NDCG@3", "NDCG@5"]
        ]
        .mean()
        .round(4)
    )
    print(mean_tbl.to_string())

    summary_df.to_csv("pl_v2_walkforward_summary.csv", index=False)
    coef_df_all.to_csv("pl_v2_coefficients.csv", index=False)

    # ── VIF selection ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("VIF BACKWARD SELECTION (reference: 2022+2023+2024 training data)")
    ref_df = df[df["Year"].isin([2022, 2023, 2024])]
    vif_selected, vif_steps_df, final_vif_df = vif_backward_selection(
        ref_df, ALL_FEATURES, threshold=5.0
    )
    vif_steps_df.to_csv("pl_v2_vif_steps.csv", index=False)
    final_vif_df.to_csv("pl_v2_vif_final.csv", index=False)
    print(f"VIF-selected features ({len(vif_selected)}): {vif_selected}")

    # ── Walk-forward with VIF-selected features ──────────────────────────────
    print(f"\n{'='*70}")
    print(f"WALK-FORWARD WITH VIF-SELECTED FEATURES ({len(vif_selected)})")
    vif_fold_rows = []
    vif_coef_rows = []

    for train_idx, test_idx, train_years, test_year in splits:
        train_label = "+".join(str(y) for y in train_years)
        print(f"\nFold: train=[{train_label}]  ->  test={test_year}")
        train = df.loc[train_idx].copy()
        test  = df.loc[test_idx].copy()

        pl_m, bl_m, coef_df = fit_and_evaluate(
            train, test, vif_selected, df, test_idx
        )
        print(f"  Spearman: {pl_m['Spearman']:.4f}  (BL: {bl_m['Spearman']:.4f})")

        for label, m in [("Model", pl_m), ("Baseline", bl_m)]:
            vif_fold_rows.append({
                "TrainYears": train_label,
                "TestYear":   test_year,
                "Method":     label,
                "N_Features": len(vif_selected) if label == "Model" else 1,
                **{k: m[k] for k in ["Spearman", "KendallTau", "Top3 Acc",
                                      "Top5 Acc", "Winner Acc", "NDCG@3",
                                      "NDCG@5", "N_Races"]},
            })
        for row in coef_df.to_dict("records"):
            row["TrainYears"] = train_label
            row["TestYear"]   = test_year
            vif_coef_rows.append(row)

    vif_summary_df = pd.DataFrame(vif_fold_rows)
    vif_summary_df.to_csv("pl_v2_vif_summary.csv", index=False)
    pd.DataFrame(vif_coef_rows).to_csv("pl_v2_vif_coefficients.csv", index=False)

    print(f"\n{'='*70}")
    print("VIF FOLD MEAN (Model vs Baseline)")
    vif_mean_tbl = (
        vif_summary_df.groupby("Method")[
            ["Spearman", "KendallTau", "Top3 Acc", "Top5 Acc",
             "Winner Acc", "NDCG@3", "NDCG@5"]
        ]
        .mean()
        .round(4)
    )
    print(vif_mean_tbl.to_string())

    print("\nSaved: pl_v2_walkforward_summary.csv, pl_v2_coefficients.csv, "
          "pl_v2_vif_steps.csv, pl_v2_vif_final.csv, pl_v2_vif_summary.csv")


if __name__ == "__main__":
    main()

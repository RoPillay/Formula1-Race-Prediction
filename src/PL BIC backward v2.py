"""
PL BIC backward v2.py
BIC + Backward Selection for the Plackett-Luce v2 model.

Selection is performed on 2022-2024 reference data (no leakage from 2025).
After selection, walk-forward CV is run with the selected feature set.

Outputs (saved to src/):
  pl_v2_bic_steps.csv        — step-by-step BIC removal log
  pl_v2_bic_features.csv     — final selected feature list
  pl_v2_bic_summary.csv      — walk-forward metrics (Model + Baseline rows)
  pl_v2_bic_coefficients.csv — per-fold PL coefficients
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, spearmanr, kendalltau
from sklearn.metrics import ndcg_score
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

YEARS     = [2022, 2023, 2024, 2025]
LAMBDA_L2 = 1.0


# ── Data loading ────────────────────────────────────────────────────────────

def load_and_prep(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["FinishPosition"])
    if "QualiPosition" in df.columns:
        n_starters = df.groupby("RaceID")["Driver"].transform("count")
        df["QualiPercentile"] = (df["QualiPosition"] - 1) / (n_starters - 1).clip(lower=1)
        df["Quali_x_Overtake"] = df["QualiPercentile"] * df["OvertakeIndex"]
        df["OvertakeOpportunity"] = (1 - df["QualiPercentile"]) * df["OvertakeIndex"]
    return df


# ── Walk-forward splits ─────────────────────────────────────────────────────

def walkforward_splits(df, years):
    splits = []
    seen = set()
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


# ── Race building ───────────────────────────────────────────────────────────

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


# ── PL likelihood (vectorized) ───────────────────────────────────────────────
# races are sorted by FinishPosition so order = [0,1,...,k-1].
# logsumexp(theta[i:]) for all i simultaneously via reverse cumulative logaddexp.

def _pl_val_grad(beta, races, lambda_l2):
    """Returns (neg_loglik, neg_grad) — used with jac=True in minimize."""
    ll = 0.0
    grad = np.zeros_like(beta)
    for race in races:
        X = race["X"]                                  # (k, p)
        theta = X @ beta                               # (k,)
        lse = np.logaddexp.accumulate(theta[::-1])[::-1]  # lse[i]=logsumexp(theta[i:])
        ll += np.sum(theta - lse)
        inv_Z = np.exp(-lse)                           # 1 / Z_i
        c = np.exp(theta) * np.cumsum(inv_Z)           # c[l] = exp(theta[l]) * sum_{i<=l} 1/Z_i
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
        options={"maxiter": 200, "ftol": 1e-6},
    )
    return res.x, pl_loglik(res.x, races), res


# ── AIC / BIC  (n = race count, not row count) ──────────────────────────────

def aic_bic(loglik, k, n_races):
    aic = -2 * loglik + 2 * k
    bic = -2 * loglik + k * np.log(n_races)
    return aic, bic


# ── BIC backward selection ──────────────────────────────────────────────────

def bic_backward_selection(train_df, features, lambda_l2=LAMBDA_L2):
    """
    Start with the full feature set. At each step remove the one feature
    whose removal gives the largest BIC decrease. Stop when no removal
    improves BIC. Returns (selected_features, final_bic, steps_df).
    n = number of races (not rows) — corrected from v1.
    """
    current = features.copy()
    n_races = train_df["RaceID"].nunique()

    races = build_races(train_df, current)
    _, loglik, _ = fit_pl(races, len(current), lambda_l2)
    _, best_bic = aic_bic(loglik, len(current), n_races)
    print(f"  Full model BIC ({len(current)} features): {best_bic:.3f}")

    steps = []
    step_num = 1

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
            print(f"  Step {step_num:>2}: Remove {best_feat:<25} "
                  f"BIC {best_bic:.3f} -> {best_bic_candidate:.3f} "
                  f"(d={best_bic_candidate - best_bic:+.3f})")
            steps.append({
                "Step": step_num,
                "Feature_Removed": best_feat,
                "BIC_Before": round(best_bic, 3),
                "BIC_After": round(best_bic_candidate, 3),
                "BIC_Delta": round(best_bic_candidate - best_bic, 3),
                "Features_Remaining": len(current) - 1,
            })
            current.remove(best_feat)
            best_bic = best_bic_candidate
            step_num += 1
        else:
            print(f"  No further BIC improvement. Stopping at {len(current)} features.")
            break

    steps_df = pd.DataFrame(steps) if steps else pd.DataFrame(
        columns=["Step", "Feature_Removed", "BIC_Before", "BIC_After",
                 "BIC_Delta", "Features_Remaining"])
    return current, best_bic, steps_df


# ── Evaluation ──────────────────────────────────────────────────────────────

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


def fit_and_evaluate(train, test, selected, df_for_baseline, test_idx):
    scale_cols = [f for f in selected if f != "QualiPercentile"]
    scaler = StandardScaler()
    tr = train.copy()
    te = test.copy()
    if scale_cols:
        tr[scale_cols] = scaler.fit_transform(tr[scale_cols])
        te[scale_cols] = scaler.transform(te[scale_cols])

    train_races = build_races(tr, selected)
    test_races  = build_races(te, selected)
    beta, _, res = fit_pl(train_races, len(selected))

    coef_rows = []
    try:
        h = res.hess_inv
        hess_arr = np.asarray(h.todense() if hasattr(h, "todense") else h)
        se = np.sqrt(np.diag(hess_arr))
        z = beta / se
        pv = 2 * (1 - norm.cdf(np.abs(z)))
        for feat, coef, stderr, z_val, p_val in zip(selected, beta, se, z, pv):
            coef_rows.append({"Feature": feat, "Coef": coef,
                              "SE_approx": stderr, "Z": z_val, "P_approx": p_val})
    except Exception:
        for feat, coef in zip(selected, beta):
            coef_rows.append({"Feature": feat, "Coef": coef})

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
    worst = pred_df.groupby("RaceID")["QualiPosition"].transform(lambda x: x.max() + 1)
    pred_df["QualiPosition"] = pred_df["QualiPosition"].fillna(worst)
    pred_df["BaselineScore"] = -pred_df["QualiPosition"]

    pl_m = compute_metrics(pred_df, "PLScore")
    bl_m = compute_metrics(pred_df, "BaselineScore")
    return pl_m, bl_m, pd.DataFrame(coef_rows)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    df = load_and_prep(DATA_PATH)
    print(f"Loaded {len(df)} rows, {df['RaceID'].nunique()} races, "
          f"years: {sorted(df['Year'].unique())}")

    splits = walkforward_splits(df, YEARS)
    print(f"{len(splits)} walk-forward folds: " +
          ", ".join(f"{'+'.join(str(y) for y in ty)}->{t}" for _, _, ty, t in splits))

    # ── BIC backward selection on 2022-2024 reference data ──────────────────
    print(f"\n{'='*70}")
    print("BIC BACKWARD SELECTION  (reference: 2022+2023+2024, n = race count)")
    ref_df = df[df["Year"].isin([2022, 2023, 2024])]
    bic_selected, final_bic, bic_steps_df = bic_backward_selection(ref_df, ALL_FEATURES)

    print(f"\nBIC-selected features ({len(bic_selected)}): {bic_selected}")
    print(f"Final BIC: {final_bic:.3f}")

    bic_steps_df.to_csv("pl_v2_bic_steps.csv", index=False)
    pd.DataFrame({"Feature": bic_selected}).to_csv("pl_v2_bic_features.csv", index=False)
    print("Saved: pl_v2_bic_steps.csv, pl_v2_bic_features.csv")

    # ── Walk-forward with BIC-selected features ──────────────────────────────
    print(f"\n{'='*70}")
    print(f"WALK-FORWARD WITH BIC-SELECTED FEATURES ({len(bic_selected)})")

    bic_fold_rows = []
    bic_coef_rows = []

    for train_idx, test_idx, train_years, test_year in splits:
        train_label = "+".join(str(y) for y in train_years)
        print(f"\n  Fold: train=[{train_label}]  ->  test={test_year}")
        train = df.loc[train_idx].copy()
        test  = df.loc[test_idx].copy()

        pl_m, bl_m, coef_df = fit_and_evaluate(
            train, test, bic_selected, df, test_idx
        )
        print(f"    Spearman: {pl_m['Spearman']:.4f}  "
              f"KendallTau: {pl_m['KendallTau']:.4f}  "
              f"Top3: {pl_m['Top3 Acc']:.4f}  "
              f"Winner: {pl_m['Winner Acc']:.4f}")

        for label, m in [("Model", pl_m), ("Baseline", bl_m)]:
            bic_fold_rows.append({
                "TrainYears": train_label,
                "TestYear":   test_year,
                "Method":     label,
                "N_Features": len(bic_selected) if label == "Model" else 1,
                **{k: m[k] for k in ["Spearman", "KendallTau", "Top3 Acc",
                                      "Top5 Acc", "Winner Acc", "NDCG@3",
                                      "NDCG@5", "N_Races"]},
            })
        for row in coef_df.to_dict("records"):
            row["TrainYears"] = train_label
            row["TestYear"]   = test_year
            bic_coef_rows.append(row)

    bic_summary_df = pd.DataFrame(bic_fold_rows)
    bic_summary_df.to_csv("pl_v2_bic_summary.csv", index=False)
    pd.DataFrame(bic_coef_rows).to_csv("pl_v2_bic_coefficients.csv", index=False)

    print(f"\n{'='*70}")
    print("BIC WALK-FORWARD MEAN (Model vs Baseline)")
    bic_mean = (
        bic_summary_df.groupby("Method")[
            ["Spearman", "KendallTau", "Top3 Acc", "Top5 Acc",
             "Winner Acc", "NDCG@3", "NDCG@5"]
        ].mean().round(4)
    )
    print(bic_mean.to_string())
    print("\nSaved: pl_v2_bic_summary.csv, pl_v2_bic_coefficients.csv")


if __name__ == "__main__":
    main()
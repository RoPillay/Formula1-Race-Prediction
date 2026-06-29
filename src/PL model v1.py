import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import ndcg_score
from scipy.special import logsumexp
from scipy.stats import norm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import r2_score
# ==============================
# 1. LOAD + PREPROCESS
# ==============================

df = pd.read_csv(
    "C:/Users/Owner/OneDrive/Download1/Research STA199/DEBUGF7_f1_enriched_prediction_dataset_2022_2025.csv"
)

FEATURES = [
    "QualiPercentile",
    "QualiGapToPole",
    "TeammateQualiGap",
    "FP_LongRunPace",
    "FP_LongRunVar",
    "DriverForm",
    "TeamStrength",
    "DriverDNFRate",
    "TeamDNFRate",
    "AirTemp",
    "TrackTemp",
    "Rainfall",
    "OvertakeIndex",
    "Quali_x_Overtake",
    "Reliability_x_Rain",
    "OvertakeOpportunity",
    "Driver_vs_Team",
    "GridSpread"
]

df = df.dropna(subset=["FinishPosition"])

# ==============================
# 3. BUILD RACE-LEVEL DATAFRAME
# ==============================

predictors = FEATURES

model_df = df[
    ["Year", "Round", "RaceID", "Driver", "Team", "FinishPosition"] + predictors
].copy()



print("Total races:", model_df["RaceID"].nunique())
'''
# VIF CHECK
def compute_vif_pl(df, predictors):
    X = df[predictors].copy()

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=predictors
    )

    vif_data = []

    for i, feature in enumerate(predictors):
        vif = variance_inflation_factor(X_scaled.values, i)
        vif_data.append((feature, vif))

    vif_df = pd.DataFrame(vif_data, columns=["Feature", "VIF"])
    vif_df = vif_df.sort_values("VIF", ascending=False)

    print("\nPL VIF Analysis:")
    print(vif_df)

    return vif_df
'''
# ==============================
# 4. 5-FOLD GROUP CV PL MODEL
# ==============================

from sklearn.model_selection import GroupKFold

def build_races_from_df(fold_df, predictors):
    races = []

    for race_id, g in fold_df.groupby("RaceID"):
        g = g.sort_values("FinishPosition").copy()

        X = g[predictors].astype(float).to_numpy()
        order = np.arange(len(g))

        races.append({
            "race_id": race_id,
            "year": g["Year"].iloc[0],
            "round": g["Round"].iloc[0],
            "drivers": g["Driver"].values,
            "finish_pos": g["FinishPosition"].values,
            "X": X,
            "order": order
        })

    return races


# ==============================
# 5. PL L2 LIKELIHOOD
# ==============================

def pl_neg_log_likelihood_l2(beta, races, lambda_l2=1.0):
    ll = 0.0

    for race in races:
        X = race["X"]
        order = race["order"]
        theta = X @ beta
        remaining = list(range(len(theta)))

        for i in order:
            ll += theta[i] - logsumexp(theta[remaining])
            remaining.remove(i)

    penalty = lambda_l2 * np.sum(beta**2)
    return -ll + penalty


def pl_loglik(beta, races):
    ll = 0.0

    for race in races:
        X = race["X"]
        order = race["order"]
        theta = X @ beta
        remaining = list(range(len(theta)))

        for i in order:
            ll += theta[i] - logsumexp(theta[remaining])
            remaining.remove(i)

    return ll


# ==============================
# 6. EVALUATION FUNCTION
# ==============================

def evaluate_pl_predictions(test_races, beta_hat):
    spearman_scores = []
    kendall_scores = []
    top3_scores = []
    top5_scores = []
    winner_hits = []
    ndcg3_scores = []
    ndcg5_scores = []

    pl_predictions = []

    for race in test_races:
        X = race["X"]
        drivers = race["drivers"]
        finish_pos = race["finish_pos"]

        scores = X @ beta_hat

        race_df = pd.DataFrame({
            "RaceID": race["race_id"],
            "Year": race["year"],
            "Round": race["round"],
            "Driver": drivers,
            "FinishPosition": finish_pos,
            "PLScore": scores
        })

        race_df["PredRank"] = race_df["PLScore"].rank(ascending=False, method="first")
        race_df["ActualRank"] = race_df["FinishPosition"].rank(ascending=True, method="first")

        rho, _ = spearmanr(race_df["PredRank"], race_df["ActualRank"])
        tau, _ = kendalltau(race_df["PredRank"], race_df["ActualRank"])

        if np.isfinite(rho):
            spearman_scores.append(rho)
        if np.isfinite(tau):
            kendall_scores.append(tau)

        pred_top3 = set(race_df.sort_values("PLScore", ascending=False).head(3)["Driver"])
        true_top3 = set(race_df.sort_values("FinishPosition").head(3)["Driver"])

        pred_top5 = set(race_df.sort_values("PLScore", ascending=False).head(5)["Driver"])
        true_top5 = set(race_df.sort_values("FinishPosition").head(5)["Driver"])

        top3_scores.append(len(pred_top3 & true_top3) / 3)
        top5_scores.append(len(pred_top5 & true_top5) / 5)

        pred_winner = race_df.sort_values("PLScore", ascending=False).iloc[0]["Driver"]
        true_winner = race_df.sort_values("FinishPosition").iloc[0]["Driver"]
        winner_hits.append(int(pred_winner == true_winner))

        relevance = (21 - race_df["FinishPosition"]).values.reshape(1, -1)
        pred_scores = race_df["PLScore"].values.reshape(1, -1)

        ndcg3_scores.append(ndcg_score(relevance, pred_scores, k=3))
        ndcg5_scores.append(ndcg_score(relevance, pred_scores, k=5))

        pl_predictions.append(race_df)

    metrics = {
        "Mean Spearman": np.nanmean(spearman_scores),
        "Mean KendallTau": np.nanmean(kendall_scores),
        "Top3 Accuracy": np.nanmean(top3_scores),
        "Top5 Accuracy": np.nanmean(top5_scores),
        "Winner Accuracy": np.nanmean(winner_hits),
        "NDCG@3": np.nanmean(ndcg3_scores),
        "NDCG@5": np.nanmean(ndcg5_scores)
    }

    predictions_df = pd.concat(pl_predictions, ignore_index=True)

    return metrics, predictions_df

# ==============================
# MODEL SELECTION: AIC / BIC
# ==============================

def fit_pl_model(races, predictors, lambda_l2=1.0):
    p = len(predictors)
    beta0 = np.zeros(p)

    res = minimize(
        pl_neg_log_likelihood_l2,
        beta0,
        args=(races, lambda_l2),
        method="L-BFGS-B",
        options={
            "maxiter": 100,
            "ftol": 1e-6
        }
    )

    beta_hat = res.x
    loglik = pl_loglik(beta_hat, races)

    return beta_hat, loglik, res


def compute_aic_bic(loglik, k, n):
    aic = -2 * loglik + 2 * k
    bic = -2 * loglik + k * np.log(n)
    return aic, bic


def backward_selection_pl(model_df, predictors, criterion="BIC", lambda_l2=1.0):
    current_predictors = predictors.copy()
    selection_history = []

    # Use all races for model selection
    races = build_races_from_df(model_df, current_predictors)
    n = len(model_df)

    beta_hat, loglik, res = fit_pl_model(races, current_predictors, lambda_l2)
    best_aic, best_bic = compute_aic_bic(loglik, len(current_predictors), n)

    best_score = best_bic if criterion == "BIC" else best_aic

    print(f"\nStarting {criterion} backward selection")
    print(f"Initial {criterion}: {best_score:.3f}")
    print(f"Initial predictors: {current_predictors}")

    improved = True

    while improved and len(current_predictors) > 1:
        improved = False
        candidate_results = []

        for feature in current_predictors:
            print (f"Testing removal of {feature}")
            trial_predictors = [f for f in current_predictors if f != feature]

            trial_races = build_races_from_df(model_df, trial_predictors)

            beta_trial, loglik_trial, res_trial = fit_pl_model(
                trial_races,
                trial_predictors,
                lambda_l2
            )

            aic_trial, bic_trial = compute_aic_bic(
                loglik_trial,
                len(trial_predictors),
                n
            )

            score = bic_trial if criterion == "BIC" else aic_trial

            candidate_results.append({
                "Removed Feature": feature,
                "Num Predictors": len(trial_predictors),
                "LogLikelihood": loglik_trial,
                "AIC": aic_trial,
                "BIC": bic_trial,
                criterion: score
            })

        candidate_df = pd.DataFrame(candidate_results)
        candidate_df = candidate_df.sort_values(criterion)

        best_candidate = candidate_df.iloc[0]

        if best_candidate[criterion] < best_score:
            removed_feature = best_candidate["Removed Feature"]

            print(
                f"Removing {removed_feature} improved {criterion}: "
                f"{best_score:.3f} -> {best_candidate[criterion]:.3f}"
            )

            current_predictors.remove(removed_feature)
            best_score = best_candidate[criterion]

            selection_history.append(best_candidate.to_dict())
            improved = True
        else:
            print(f"No further {criterion} improvement.")
            break

    final_races = build_races_from_df(model_df, current_predictors)
    final_beta, final_loglik, final_res = fit_pl_model(
        final_races,
        current_predictors,
        lambda_l2
    )

    final_aic, final_bic = compute_aic_bic(
        final_loglik,
        len(current_predictors),
        n
    )

    final_summary = {
        "Criterion": criterion,
        "Final Predictors": current_predictors,
        "Num Predictors": len(current_predictors),
        "Final LogLikelihood": final_loglik,
        "Final AIC": final_aic,
        "Final BIC": final_bic
    }

    history_df = pd.DataFrame(selection_history)

    return current_predictors, final_summary, history_df

# AIC stepwise selection 
def stepwise_selection_pl(model_df, all_predictors, criterion="AIC", lambda_l2=1.0):
    current_predictors = []
    remaining_predictors = all_predictors.copy()
    selection_history = []

    n = len(model_df)

    # Start with null model
    null_races = build_races_from_df(model_df, [])
    best_score = np.inf

    improved = True

    while improved:
        improved = False
        candidate_results = []

        # Forward step: try adding each remaining feature
        for feature in remaining_predictors:
            trial_predictors = current_predictors + [feature]
            trial_races = build_races_from_df(model_df, trial_predictors)

            beta_trial, loglik_trial, _ = fit_pl_model(
                trial_races,
                trial_predictors,
                lambda_l2
            )

            aic_trial, bic_trial = compute_aic_bic(
                loglik_trial,
                len(trial_predictors),
                n
            )

            score = aic_trial if criterion == "AIC" else bic_trial

            candidate_results.append({
                "Step": "Add",
                "Feature": feature,
                "Num Predictors": len(trial_predictors),
                "LogLikelihood": loglik_trial,
                "AIC": aic_trial,
                "BIC": bic_trial,
                "Score": score
            })

        # Backward step: try removing each current feature
        for feature in current_predictors:
            trial_predictors = [f for f in current_predictors if f != feature]

            if len(trial_predictors) == 0:
                continue

            trial_races = build_races_from_df(model_df, trial_predictors)

            beta_trial, loglik_trial, _ = fit_pl_model(
                trial_races,
                trial_predictors,
                lambda_l2
            )

            aic_trial, bic_trial = compute_aic_bic(
                loglik_trial,
                len(trial_predictors),
                n
            )

            score = aic_trial if criterion == "AIC" else bic_trial

            candidate_results.append({
                "Step": "Remove",
                "Feature": feature,
                "Num Predictors": len(trial_predictors),
                "LogLikelihood": loglik_trial,
                "AIC": aic_trial,
                "BIC": bic_trial,
                "Score": score
            })

        candidate_df = pd.DataFrame(candidate_results)

        if candidate_df.empty:
            break

        best_candidate = candidate_df.sort_values("Score").iloc[0]

        if best_candidate["Score"] < best_score:
            best_score = best_candidate["Score"]
            improved = True

            if best_candidate["Step"] == "Add":
                selected_feature = best_candidate["Feature"]
                current_predictors.append(selected_feature)
                remaining_predictors.remove(selected_feature)
                print(f"Adding {selected_feature}, {criterion} = {best_score:.3f}")

            else:
                removed_feature = best_candidate["Feature"]
                current_predictors.remove(removed_feature)
                remaining_predictors.append(removed_feature)
                print(f"Removing {removed_feature}, {criterion} = {best_score:.3f}")

            selection_history.append(best_candidate.to_dict())
        else:
            print(f"No further {criterion} improvement.")
            break

    history_df = pd.DataFrame(selection_history)

    print("\nFinal AIC-stepwise predictors:")
    print(current_predictors)

    return current_predictors, history_df


# ==============================
# 7. TRAIN/EVALUATE PL WITH 5-FOLD CV
# ==============================

groups = model_df["RaceID"].values
gkf = GroupKFold(n_splits=5)

scale_features = [f for f in predictors if f != "QualiPercentile"]

all_metrics = []
all_predictions = []
all_coef_tables = []
lambda_l2 = 1.0

# ==============================
# 7A. PL MODEL SELECTION
# ==============================

bic_predictors = [
    "QualiPercentile",
    "TeamStrength",
    "Quali_x_Overtake",
    "Driver_vs_Team"
]

predictors = bic_predictors


scale_features = [f for f in predictors if f != "QualiPercentile"]

model_df = df[
    ["Year", "Round", "RaceID", "Driver", "Team", "FinishPosition"] + predictors
].copy()

groups = model_df["RaceID"].values


print("\n================ PL MODEL SELECTION: BIC BACKWARD ================")

bic_predictors, bic_summary, bic_history = backward_selection_pl(
    model_df=model_df,
    predictors=FEATURES.copy(),
    criterion="BIC",
    lambda_l2=lambda_l2
)

print("\nFinal BIC-selected predictors:")
print(bic_predictors)
print("\nBIC Selection Summary:")
print(bic_summary)

bic_history.to_csv("DEBUGF7_pl_BIC_backward_selection_history.csv", index=False)

predictors = bic_predictors
scale_features = [f for f in predictors if f != "QualiPercentile"]

model_df = df[
    ["Year", "Round", "RaceID", "Driver", "Team", "FinishPosition"] + predictors
].copy()

groups = model_df["RaceID"].values


print("\n================ PL MODEL SELECTION: AIC BACKWARD ================")

aic_predictors, aic_summary, aic_history = backward_selection_pl(
    model_df=model_df,
    predictors=FEATURES.copy(),
    criterion="AIC",
    lambda_l2=lambda_l2
)

print("\nFinal AIC-selected predictors:")
print(aic_predictors)
print("\nAIC Selection Summary:")
print(aic_summary)

aic_history.to_csv("DEBUGF7_pl_AIC_backward_selection_history.csv", index=False)
'''

'''
# VIF CHECK + BACKWARD SELECTION
def vif_backward_selection_pl(df, predictors, threshold=5.0):
    print("\n================ PL VIF BACKWARD SELECTION ================")

    X = df[predictors].copy()
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=predictors
    )

    current_features = predictors.copy()

    while True:
        vif_values = []

        for i, feature in enumerate(current_features):
            vif = variance_inflation_factor(
                X_scaled[current_features].values, i
            )
            vif_values.append((feature, vif))

        vif_df = pd.DataFrame(vif_values, columns=["Feature", "VIF"])
        vif_df = vif_df.sort_values("VIF", ascending=False)

        print("\nCurrent VIF:")
        print(vif_df)

        max_vif = vif_df.iloc[0]["VIF"]
        max_feature = vif_df.iloc[0]["Feature"]

        if max_vif < threshold:
            break

        print(f"Removing {max_feature} (VIF={max_vif:.2f})")
        current_features.remove(max_feature)

    print("\nFinal PL VIF-selected features:")
    print(current_features)

    return current_features

# VIF-based backward selection
vif_predictors = vif_backward_selection_pl(model_df, FEATURES.copy())

predictors = vif_predictors
scale_features = [f for f in predictors if f != "QualiPercentile"]

model_df = df[
    ["Year", "Round", "RaceID", "Driver", "Team", "FinishPosition"] + predictors
].copy()

groups = model_df["RaceID"].values


print("\n================ PL MODEL SELECTION: AIC STEPWISE ================")

aic_stepwise_predictors, aic_stepwise_history = stepwise_selection_pl(
    model_df=model_df,
    all_predictors=FEATURES.copy(),
    criterion="AIC",
    lambda_l2=lambda_l2
)

predictors = aic_stepwise_predictors
scale_features = [f for f in predictors if f != "QualiPercentile"]

model_df = df[
    ["Year", "Round", "RaceID", "Driver", "Team", "FinishPosition"] + predictors
].copy()

groups = model_df["RaceID"].values

# 5 fold group CV
for fold, (train_idx, test_idx) in enumerate(gkf.split(model_df, groups=groups), start=1):
    print(f"\n================ PL Fold {fold} ================")

    train = model_df.iloc[train_idx].copy()
    test = model_df.iloc[test_idx].copy()

    scaler = StandardScaler()

    train[scale_features] = scaler.fit_transform(train[scale_features])
    test[scale_features] = scaler.transform(test[scale_features])

    train_races = build_races_from_df(train, predictors)
    test_races = build_races_from_df(test, predictors)

    p = len(predictors)
    beta0 = np.zeros(p)

    res = minimize(
        pl_neg_log_likelihood_l2,
        beta0,
        args=(train_races, lambda_l2),
        method="BFGS"
    )

    beta_hat = res.x

    # PL coefficient importance + approximate p-values
    try:
        hess_inv = np.asarray(res.hess_inv)
        se = np.sqrt(np.diag(hess_inv))

        z_scores = beta_hat / se
        p_values = 2 * (1 - norm.cdf(np.abs(z_scores)))

        coef_df = pd.DataFrame({
            "Fold": fold,
            "Feature": predictors,
            "Coefficient": beta_hat,
            "Std_Error": se,
            "Z": z_scores,
            "P_Value": p_values,
            "Abs_Coefficient": np.abs(beta_hat)
        })

        coef_df = coef_df.sort_values("Abs_Coefficient", ascending=False)

        print("\nPL Coefficients / Importance:")
        print(coef_df)

        all_coef_tables.append(coef_df)

    except Exception as e:
        print("Could not compute PL standard errors:", e)

    metrics, preds = evaluate_pl_predictions(test_races, beta_hat)
    metrics["Test LogLikelihood"] = pl_loglik(beta_hat, test_races)
    metrics["Fold"] = fold

    print(metrics)

    preds["Fold"] = fold

    all_metrics.append(metrics)
    all_predictions.append(preds)

pl_metrics_df = pd.DataFrame(all_metrics)
pl_predictions_df = pd.concat(all_predictions, ignore_index=True)

print("\n================ FINAL PL CROSS-VALIDATED METRICS ================")
print(pl_metrics_df.mean(numeric_only=True))

pl_metrics_df.to_csv("DEBUGF7_pl_BIC_comparable_cv_metrics.csv", index=False)
pl_predictions_df.to_csv("DEBUGF7_pl_BIC_comparable_cv_predictions.csv", index=False)

print("\nSaved files:")
print("1. DEBUGF7_pl_BIC_comparable_cv_metrics.csv")
print("2. DEBUGF7_pl_BIC_comparable_cv_predictions.csv")

pl_coef_df = pd.concat(all_coef_tables, ignore_index=True)

pl_coef_df.to_csv("DEBUGF7_enriched_pl_BIC_coefficients_pvalues.csv", index=False)


pl_coef_summary = (
    pl_coef_df
    .groupby("Feature", as_index=False)
    .agg({
        "Coefficient": "mean",
        "Std_Error": "mean",
        "Z": "mean",
        "P_Value": "mean",
        "Abs_Coefficient": "mean"
    })
    .sort_values("Abs_Coefficient", ascending=False)
)

print("\n================ FINAL ENRICHED PL COEFFICIENT SUMMARY ================")
print(pl_coef_summary)

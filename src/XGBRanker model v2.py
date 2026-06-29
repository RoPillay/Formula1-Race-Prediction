import warnings
warnings.filterwarnings("ignore")

import fastf1
import numpy as np
import pandas as pd

from scipy.stats import spearmanr, kendalltau
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss, ndcg_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRanker
from statsmodels.stats.outliers_influence import variance_inflation_factor

fastf1.Cache.enable_cache(
    r"C:\Users\Owner\OneDrive\Download1\Research STA199\cache"
)

# ==============================================================
# CONFIG
# ==============================================================

YEARS = [2022, 2023, 2024, 2025]

# Set to a CSV path to skip FastF1 and load an existing enriched dataset.
# The DEBUGF7 CSV from the original pipeline works here.
CSV_FALLBACK = (
    "C:/Users/Owner/OneDrive/Download1/Research STA199/"
    "DEBUGF7_f1_enriched_prediction_dataset_2022_2025.csv"
)

FEATURES = [
    "QualiPercentile", "QualiGapToPole", "TeammateQualiGap",
    "FP_LongRunPace", "FP_LongRunVar", "DriverForm", "TeamStrength",
    "DriverDNFRate", "TeamDNFRate", "AirTemp", "TrackTemp", "Rainfall",
    "OvertakeIndex", "Quali_x_Overtake", "Reliability_x_Rain",
    "OvertakeOpportunity", "Driver_vs_Team", "GridSpread",
]

OVERTAKE_INDEX = {
    "Monaco": 0.20, "Hungary": 0.35, "Singapore": 0.35,
    "Netherlands": 0.40, "Emilia Romagna": 0.40, "Spain": 0.55,
    "Japan": 0.55, "Qatar": 0.55, "Australia": 0.60, "Miami": 0.60,
    "France": 0.60, "Mexico": 0.65, "Abu Dhabi": 0.65, "China": 0.65,
    "Bahrain": 0.70, "Canada": 0.70, "Great Britain": 0.70,
    "United States": 0.70, "Saudi Arabia": 0.75, "Austria": 0.75,
    "Azerbaijan": 0.75, "Brazil": 0.75, "Italy": 0.80, "Las Vegas": 0.80,
    "Belgium": 0.85,
}


# ==============================================================
# HELPERS
# ==============================================================

def normalize_event_location(name):
    if pd.isna(name):
        return None
    s = str(name).lower()
    if "bahrain" in s or "sakhir" in s:         return "Bahrain"
    if "saudi" in s or "jeddah" in s:           return "Saudi Arabia"
    if "australia" in s or "melbourne" in s:    return "Australia"
    if "japan" in s or "suzuka" in s:           return "Japan"
    if "china" in s or "shanghai" in s:         return "China"
    if "miami" in s:                            return "Miami"
    if "emilia" in s or "imola" in s:           return "Emilia Romagna"
    if "monaco" in s:                           return "Monaco"
    if "canada" in s or "montreal" in s or "gilles" in s: return "Canada"
    if "spain" in s or "barcelona" in s:        return "Spain"
    if "austria" in s or "spielberg" in s:      return "Austria"
    if "great britain" in s or "silverstone" in s or "british" in s: return "Great Britain"
    if "hungary" in s or "budapest" in s:       return "Hungary"
    if "belgium" in s or "spa" in s:            return "Belgium"
    if "netherlands" in s or "zandvoort" in s:  return "Netherlands"
    if "italy" in s or "monza" in s:            return "Italy"
    if "azerbaijan" in s or "baku" in s:        return "Azerbaijan"
    if "singapore" in s or "marina bay" in s:   return "Singapore"
    if "united states" in s or "austin" in s or "cota" in s: return "United States"
    if "mexico" in s:                           return "Mexico"
    if "france" in s or "castellet" in s or "paul ricard" in s: return "France"
    if "brazil" in s or "interlagos" in s or "sao paulo" in s: return "Brazil"
    if "las vegas" in s:                        return "Las Vegas"
    if "qatar" in s or "lusail" in s:           return "Qatar"
    if "abu dhabi" in s or "yas" in s:          return "Abu Dhabi"
    return str(name)


def safe_seconds(x):
    try:
        return np.nan if pd.isna(x) else x.total_seconds()
    except Exception:
        return np.nan


def normalize_team(name):
    if pd.isna(name):
        return np.nan
    name = str(name).lower()
    if "mclaren" in name:                                          return "McLaren"
    if "red bull" in name:                                         return "Red Bull"
    if "ferrari" in name and "haas" not in name:                   return "Ferrari"
    if "mercedes" in name:                                         return "Mercedes"
    if "aston martin" in name:                                     return "Aston Martin"
    if "alpine" in name or "renault" in name:                      return "Alpine"
    if "williams" in name:                                         return "Williams"
    if "alphatauri" in name or "rb" in name or "racing bulls" in name \
            or "visa" in name:                                     return "RB"
    if "sauber" in name or "alfa romeo" in name or "kick" in name: return "Sauber"
    if "haas" in name:                                             return "Haas"
    return str(name)


def is_dnf_status(status):
    bad = [
        "accident", "collision", "engine", "gearbox", "hydraulics",
        "retired", "dnf", "mechanical", "brakes", "electrical",
        "power unit", "oil", "water", "fuel", "disqualified", "suspension",
    ]
    return int(any(w in str(status).lower() for w in bad))


def get_rounds_for_year(year):
    rounds = {2021: 22, 2022: 22, 2023: 22, 2024: 24, 2025: 24}
    return list(range(1, rounds.get(year, 24) + 1))


# ==============================================================
# FEATURE EXTRACTION — race + weather combined (FIX: single load)
# ==============================================================

def extract_race_and_weather(year, rnd):
    """Load race session once with weather enabled to avoid double-loading."""
    try:
        session = fastf1.get_session(year, rnd, "R")
        session.load(laps=False, telemetry=False, weather=True, messages=False)
    except Exception as e:
        print(f"  Race load failed {year} R{rnd}: {e}")
        return pd.DataFrame(), {"AirTemp": np.nan, "TrackTemp": np.nan, "Rainfall": 0}

    results = session.results
    event_name = session.event.get("EventName", f"{year}_Round_{rnd}")
    event_location = session.event.get("Location", event_name)

    rows = []
    for _, row in results.iterrows():
        drv = row.get("Abbreviation")
        if pd.isna(drv):
            continue
        rows.append({
            "Driver": drv,
            "Team": normalize_team(row.get("TeamName")),
            "FinishPosition": row.get("Position", np.nan),
            "DNF": is_dnf_status(row.get("Status", "")),
            "EventName": event_name,
            "EventLocation": event_location,
        })
    race_df = pd.DataFrame(rows)

    w = session.weather_data
    if w is not None and not w.empty:
        weather = {
            "AirTemp":   w["AirTemp"].mean()   if "AirTemp"   in w.columns else np.nan,
            "TrackTemp": w["TrackTemp"].mean()  if "TrackTemp" in w.columns else np.nan,
            "Rainfall":  float(w["Rainfall"].mean() > 0) if "Rainfall" in w.columns else 0,
        }
    else:
        weather = {"AirTemp": np.nan, "TrackTemp": np.nan, "Rainfall": 0}

    return race_df, weather


def extract_quali_features(year, rnd):
    try:
        session = fastf1.get_session(year, rnd, "Q")
        session.load()
        laps = session.laps
        results = session.results
    except Exception:
        return pd.DataFrame()

    fastest = (
        laps.groupby("Driver")["LapTime"].min()
        .dropna()
        .apply(safe_seconds)
    )
    if fastest.empty:
        return pd.DataFrame()
    pole = fastest.min()

    rows = []
    for _, row in results.iterrows():
        drv = row.get("Abbreviation")
        if pd.isna(drv):
            continue
        qt = fastest.get(drv, np.nan)
        rows.append({
            "Driver": drv,
            "Team": normalize_team(row.get("TeamName")),
            "QualiPosition": row.get("Position", np.nan),
            "QualiTime": qt,
            "QualiGapToPole": qt - pole if not pd.isna(qt) else np.nan,
        })

    qdf = pd.DataFrame(rows)

    gaps = []
    for _, g in qdf.groupby("Team"):
        if len(g) != 2:
            gaps += [(r["Driver"], np.nan) for _, r in g.iterrows()]
            continue
        g = g.sort_values("QualiTime")
        d1, d2 = g.iloc[0], g.iloc[1]
        gaps.append((d1["Driver"], d1["QualiTime"] - d2["QualiTime"]))
        gaps.append((d2["Driver"], d2["QualiTime"] - d1["QualiTime"]))

    gap_df = pd.DataFrame(gaps, columns=["Driver", "TeammateQualiGap"])
    return qdf.merge(gap_df, on="Driver", how="left")


def extract_practice_features(year, rnd):
    rows = []
    for session_name in ["FP2", "FP3"]:
        try:
            session = fastf1.get_session(year, rnd, session_name)
            session.load()
            laps = session.laps.pick_quicklaps().pick_wo_box()
        except Exception:
            continue
        if laps is None or laps.empty:
            continue
        for drv in laps["Driver"].dropna().unique():
            d = laps[laps["Driver"] == drv].copy()
            d["LapSeconds"] = d["LapTime"].apply(safe_seconds)
            d = d.dropna(subset=["LapSeconds"])
            if len(d) < 5:
                continue
            long_run = d[
                (d["LapSeconds"] > d["LapSeconds"].quantile(0.20)) &
                (d["LapSeconds"] < d["LapSeconds"].quantile(0.85))
            ]
            if len(long_run) < 4:
                long_run = d
            rows.append({
                "Driver": drv,
                "FP_LongRunPace": long_run["LapSeconds"].mean(),
                "FP_LongRunVar": long_run["LapSeconds"].std(),
            })
    if not rows:
        return pd.DataFrame(columns=["Driver", "FP_LongRunPace", "FP_LongRunVar"])
    return pd.DataFrame(rows).groupby("Driver", as_index=False).mean(numeric_only=True)


# ==============================================================
# RAW DATASET BUILD
# ==============================================================

def build_raw_dataset():
    all_rows = []
    load_log = []

    for year in YEARS:
        for rnd in get_rounds_for_year(year):
            print(f"Loading {year} Round {rnd}...")

            quali = extract_quali_features(year, rnd)
            race, weather = extract_race_and_weather(year, rnd)   # FIX: single load
            fp = extract_practice_features(year, rnd)

            if quali.empty or race.empty:
                print(f"  SKIPPED {year} R{rnd}: quali={len(quali)} race={len(race)}")
                load_log.append({"Year": year, "Round": rnd, "Loaded": False,
                                 "QualiRows": len(quali), "RaceRows": len(race)})
                continue

            df = race.merge(quali, on=["Driver", "Team"], how="left")
            df = df.merge(fp, on="Driver", how="left")

            df["Year"] = year
            df["Round"] = rnd
            df["RaceID"] = f"{year}_{rnd}"
            df["AirTemp"] = weather["AirTemp"]
            df["TrackTemp"] = weather["TrackTemp"]
            df["Rainfall"] = weather["Rainfall"]

            df["EventName"] = df["EventName"].fillna(f"{year}_Round_{rnd}")
            df["EventLocation"] = df["EventLocation"].fillna(df["EventName"])

            track_key = normalize_event_location(df["EventLocation"].iloc[0])
            df["TrackKey"] = track_key
            df["OvertakeIndex"] = OVERTAKE_INDEX.get(track_key, 0.60)

            if track_key not in OVERTAKE_INDEX:
                print(f"  WARNING: OvertakeIndex default for {year} R{rnd}: {track_key}")

            all_rows.append(df)
            load_log.append({"Year": year, "Round": rnd, "Loaded": True,
                             "QualiRows": len(quali), "RaceRows": len(race)})

    pd.DataFrame(load_log).to_csv("v2_f1_load_log.csv", index=False)

    if not all_rows:
        raise ValueError("No races loaded.")

    data = pd.concat(all_rows, ignore_index=True)
    data = data.dropna(subset=["FinishPosition"])
    data["FinishPosition"] = data["FinishPosition"].astype(float)
    return data


# ==============================================================
# VECTORIZED ROLLING FEATURES  (FIX: replaces O(n²) row loop)
# ==============================================================

def add_rolling_features(data):
    data = data.sort_values(["Year", "Round"]).copy()

    POINTS_MAP = {1:25, 2:18, 3:15, 4:12, 5:10, 6:8, 7:6, 8:4, 9:2, 10:1}
    data["Points"] = data["FinishPosition"].map(POINTS_MAP).fillna(0)

    # Driver rolling: each driver appears once per race, so shift(1) safely
    # excludes the current race and rolling(3) looks back 3 races
    data = data.sort_values(["Year", "Round", "Driver"]).copy()
    data["DriverForm"] = (
        data.groupby("Driver")["Points"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )
    data["DriverDNFRate"] = (
        data.groupby("Driver")["DNF"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )

    # Team rolling: aggregate to one row per (Team, Race) first to avoid
    # within-race leakage that would occur if we shifted at the driver level
    team_race = (
        data.groupby(["Team", "Year", "Round", "RaceID"])[["Points", "DNF"]]
        .mean()
        .reset_index()
        .sort_values(["Team", "Year", "Round"])
    )
    team_race["TeamStrength"] = (
        team_race.groupby("Team")["Points"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )
    team_race["TeamDNFRate"] = (
        team_race.groupby("Team")["DNF"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )

    data = data.drop(columns=["TeamStrength", "TeamDNFRate"], errors="ignore")
    data = data.merge(
        team_race[["Team", "RaceID", "TeamStrength", "TeamDNFRate"]],
        on=["Team", "RaceID"],
        how="left",
    )
    return data


# ==============================================================
# FEATURE FINALIZATION
# ==============================================================

def finalize_features(data):
    data = data.copy()
    data = add_rolling_features(data)

    # FIX: QualiPercentile uses actual grid size, not a fixed 19
    n_starters = data.groupby("RaceID")["Driver"].transform("count")
    data["QualiPercentile"] = np.where(
        data["QualiPosition"].notna() & (n_starters > 1),
        (data["QualiPosition"] - 1) / (n_starters - 1),
        np.nan,
    )

    data["Quali_x_Overtake"] = data["QualiPercentile"] * data["OvertakeIndex"]
    data["Reliability_x_Rain"] = data["TeamDNFRate"] * data["Rainfall"]
    data["OvertakeOpportunity"] = (1 - data["QualiPercentile"]) * data["OvertakeIndex"]
    data["Driver_vs_Team"] = data["DriverForm"] - data["TeamStrength"]
    data["GridSpread"] = data.groupby("RaceID")["QualiGapToPole"].transform("std")

    for col in FEATURES:
        if col not in data.columns:
            data[col] = np.nan
        data[col] = data[col].replace([np.inf, -np.inf], np.nan)
        if data[col].isna().all():
            data[col] = 0
        else:
            data[col] = data[col].fillna(data[col].median())

    return data.dropna(subset=["FinishPosition"])


# ==============================================================
# WALK-FORWARD SPLITS
# ==============================================================

def walkforward_splits(data, years):
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
            train_idx = data.index[data["Year"].isin(tr_yrs)].tolist()
            test_idx  = data.index[data["Year"] == te_year].tolist()
            if train_idx and test_idx:
                splits.append((train_idx, test_idx, tr_yrs, te_year))
                seen.add(key)

    # Expanding cumulative: train on all years up to y, test on y+1
    for i in range(2, len(years)):
        tr_yrs  = years[:i]
        te_year = years[i]
        key = (tuple(tr_yrs), te_year)
        if key not in seen:
            train_idx = data.index[data["Year"].isin(tr_yrs)].tolist()
            test_idx  = data.index[data["Year"] == te_year].tolist()
            if train_idx and test_idx:
                splits.append((train_idx, test_idx, tr_yrs, te_year))
                seen.add(key)

    return splits


# ==============================================================
# VIF SELECTION  (FIX: result now feeds directly into model)
# ==============================================================

def vif_backward_selection(data, features, threshold=5.0):
    """Returns (selected_features, steps_df, final_vif_df)."""
    print("\n===== VIF Backward Selection =====")
    X = data[features].copy().replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

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
        print(vif_df.to_string(index=False))

        worst = vif_df.iloc[0]
        if worst["VIF"] < threshold:
            break
        print(f"  Removing {worst['Feature']} (VIF={worst['VIF']:.2f})")
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

    print(f"\nFinal VIF-selected features ({len(current)}): {current}")
    steps_df = pd.DataFrame(steps) if steps else pd.DataFrame(
        columns=["Step", "Feature_Removed", "VIF_at_Removal", "Features_Remaining"])
    return current, steps_df, final_vif_df


# ==============================================================
# EVALUATION
# ==============================================================

def evaluate_predictions(test_df, pred_score_col):
    spearmans, kendalls = [], []
    top3s, top5s, winners = [], [], []
    ndcg3s, ndcg5s = [], []

    for _, g in test_df.groupby("RaceID"):
        if len(g) < 10:
            continue
        g = g.copy()
        pred_rank = g[pred_score_col].rank(ascending=False, method="first")
        actual_rank = g["FinishPosition"].rank(ascending=True, method="first")
        rho, _ = spearmanr(pred_rank, actual_rank)
        tau, _ = kendalltau(pred_rank, actual_rank)
        if np.isfinite(rho):
            spearmans.append(rho)
        if np.isfinite(tau):
            kendalls.append(tau)

        pred_top3 = set(g.nlargest(3, pred_score_col)["Driver"])
        true_top3 = set(g.nsmallest(3, "FinishPosition")["Driver"])
        pred_top5 = set(g.nlargest(5, pred_score_col)["Driver"])
        true_top5 = set(g.nsmallest(5, "FinishPosition")["Driver"])
        top3s.append(len(pred_top3 & true_top3) / 3)
        top5s.append(len(pred_top5 & true_top5) / 5)

        pred_winner = g.nlargest(1, pred_score_col)["Driver"].iloc[0]
        true_winner = g.nsmallest(1, "FinishPosition")["Driver"].iloc[0]
        winners.append(int(pred_winner == true_winner))

        relevance = (21 - g["FinishPosition"]).values.reshape(1, -1)
        scores_arr = g[pred_score_col].values.astype(float)
        if not np.all(np.isfinite(scores_arr)):
            continue
        scores_2d = scores_arr.reshape(1, -1)
        ndcg3s.append(ndcg_score(relevance, scores_2d, k=3))
        ndcg5s.append(ndcg_score(relevance, scores_2d, k=5))

    return {
        "Spearman":   np.mean(spearmans),
        "KendallTau": np.mean(kendalls),
        "Top3 Acc":   np.mean(top3s),
        "Top5 Acc":   np.mean(top5s),
        "Winner Acc": np.mean(winners),
        "NDCG@3":     np.mean(ndcg3s),
        "NDCG@5":     np.mean(ndcg5s),
        "N_Races":    len(spearmans),
    }


# ==============================================================
# TWO-STAGE MODEL  (features passed as parameter)
# ==============================================================

def train_and_evaluate_fold(train, test, features):
    scale_cols = [f for f in features if f != "QualiPercentile"]
    scaler = StandardScaler()

    train_s = train.copy()
    test_s = test.copy()
    if scale_cols:
        train_s[scale_cols] = scaler.fit_transform(train[scale_cols])
        test_s[scale_cols] = scaler.transform(test[scale_cols])

    X_train = train_s[features]
    X_test = test_s[features]

    # Stage 1: DNF probability
    dnf_model = LogisticRegression(max_iter=2000, class_weight="balanced")
    dnf_model.fit(X_train, train["DNF"])
    train_dnf = dnf_model.predict_proba(X_train)[:, 1]
    test_dnf = dnf_model.predict_proba(X_test)[:, 1]

    # Stage 2: Ranking (DNFProbFeature as additional input)
    rank_features = features + ["DNFProbFeature"]

    train_r = train_s.copy()
    test_r = test_s.copy()
    train_r["DNFProbFeature"] = train_dnf
    test_r["DNFProbFeature"] = test_dnf

    # Sort by RaceID so group_sizes align with the data order for XGBRanker
    train_r = train_r.sort_values("RaceID").copy()
    test_r = test_r.sort_values("RaceID").copy()

    y_train = 21 - train_r["FinishPosition"]
    group_sizes = train_r.groupby("RaceID").size().values

    ranker = XGBRanker(
        objective="rank:ndcg", eval_metric="ndcg@3",
        n_estimators=500, learning_rate=0.03, max_depth=3,
        subsample=0.85, colsample_bytree=0.85,
        reg_lambda=5.0, reg_alpha=0.5, random_state=42,
    )
    ranker.fit(train_r[rank_features], y_train, group=group_sizes)

    # FIX: no magic 2.0 multiplier — ranker already learns the DNF penalty
    # via DNFProbFeature; manual adjustment double-penalised DNF-prone drivers
    test_r["PredScore"] = ranker.predict(test_r[rank_features])
    # Fill NaN QualiPosition (e.g. late DNS) with worst rank so baseline is valid
    worst_grid = test_r["QualiPosition"].max() + 1
    test_r["BaselineScore"] = -test_r["QualiPosition"].fillna(worst_grid)
    test_r["PredDNFProb"] = test_dnf

    model_metrics = evaluate_predictions(test_r, "PredScore")
    baseline_metrics = evaluate_predictions(test_r, "BaselineScore")

    dnf_metrics = {
        "DNF_LogLoss": log_loss(
            test_r["DNF"], np.clip(test_r["PredDNFProb"], 1e-5, 1 - 1e-5)
        ),
        "DNF_Brier": brier_score_loss(test_r["DNF"], test_r["PredDNFProb"]),
    }

    return model_metrics, baseline_metrics, dnf_metrics, test_r


# ==============================================================
# FINAL MODEL (for feature importance, trained on all eval data)
# ==============================================================

def train_final_model(data, features):
    # Sort first so group_sizes align with data order throughout
    data = data.sort_values("RaceID").copy()

    scale_cols = [f for f in features if f != "QualiPercentile"]
    scaler = StandardScaler()

    X_scaled = data[features].copy()
    if scale_cols:
        X_scaled[scale_cols] = scaler.fit_transform(data[scale_cols])

    dnf_model = LogisticRegression(max_iter=2000, class_weight="balanced")
    dnf_model.fit(X_scaled[features], data["DNF"])
    dnf_prob = dnf_model.predict_proba(X_scaled[features])[:, 1]

    rank_features = features + ["DNFProbFeature"]
    final_data = data.copy()
    final_data["DNFProbFeature"] = dnf_prob   # index-aligned after sort

    y_rank = 21 - final_data["FinishPosition"]
    group_sizes = final_data.groupby("RaceID").size().values

    ranker = XGBRanker(
        objective="rank:ndcg", eval_metric="ndcg@3",
        n_estimators=500, learning_rate=0.03, max_depth=3,
        subsample=0.85, colsample_bytree=0.85,
        reg_lambda=5.0, reg_alpha=0.5, random_state=42,
    )
    ranker.fit(final_data[rank_features], y_rank, group=group_sizes)

    importance = pd.DataFrame({
        "Feature": rank_features,
        "Importance": ranker.feature_importances_,
    }).sort_values("Importance", ascending=False)

    return dnf_model, ranker, scaler, importance


# ==============================================================
# LEAVE-ONE-OUT ANALYSIS  (diagnostic — slow, set RUN_LOO=True to enable)
# ==============================================================

RUN_LOO = True

def leave_one_out_analysis(data, features, splits):
    print("\n===== Leave-One-Out Feature Analysis =====")

    base_metrics_per_fold = []
    for train_idx, test_idx, train_years, test_year in splits:
        mm, _, _, _ = train_and_evaluate_fold(
            data.loc[train_idx], data.loc[test_idx], features
        )
        base_metrics_per_fold.append(mm)
    base = pd.DataFrame(base_metrics_per_fold).mean()

    results = []
    for feat in features:
        reduced = [f for f in features if f != feat]
        fold_metrics = []
        for train_idx, test_idx, train_years, test_year in splits:
            mm, _, _, _ = train_and_evaluate_fold(
                data.loc[train_idx], data.loc[test_idx], reduced
            )
            fold_metrics.append(mm)
        new = pd.DataFrame(fold_metrics).mean()
        row = {"Feature_Removed": feat}
        for m in ["Spearman", "KendallTau", "Top3 Acc", "Top5 Acc",
                  "Winner Acc", "NDCG@3", "NDCG@5"]:
            row[f"Delta_{m}"] = new[m] - base[m]
        results.append(row)

    loo_df = pd.DataFrame(results).sort_values("Delta_Spearman", ascending=False)
    print(loo_df.to_string(index=False))
    return loo_df


# ==============================================================
# MAIN
# ==============================================================

def load_or_build_dataset():
    """Load existing enriched CSV if available, otherwise pull from FastF1."""
    import os
    if CSV_FALLBACK and os.path.exists(CSV_FALLBACK):
        print(f"Loading existing dataset from CSV (skipping FastF1):\n  {CSV_FALLBACK}")
        df = pd.read_csv(CSV_FALLBACK)
        df = df.dropna(subset=["FinishPosition"])
        df["FinishPosition"] = df["FinishPosition"].astype(float)
        # Re-apply QualiPercentile fix using actual grid size
        if "QualiPosition" in df.columns:
            n_starters = df.groupby("RaceID")["Driver"].transform("count")
            df["QualiPercentile"] = np.where(
                df["QualiPosition"].notna() & (n_starters > 1),
                (df["QualiPosition"] - 1) / (n_starters - 1),
                np.nan,
            )
            df["Quali_x_Overtake"] = df["QualiPercentile"] * df["OvertakeIndex"]
            df["OvertakeOpportunity"] = (1 - df["QualiPercentile"]) * df["OvertakeIndex"]
        # Impute any NaNs in FEATURES
        for col in FEATURES:
            if col in df.columns:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                if not df[col].isna().all():
                    df[col] = df[col].fillna(df[col].median())
        return df
    else:
        print("Building dataset from FastF1 (this may take a while)...")
        raw = build_raw_dataset()
        return finalize_features(raw)


def main():
    data = load_or_build_dataset()

    print(f"\nDataset: {len(data)} rows, {data['RaceID'].nunique()} races, "
          f"years: {sorted(data['Year'].unique())}")

    splits = walkforward_splits(data, YEARS)
    print(f"\n{len(splits)} folds: " +
          ", ".join(f"{'+'.join(str(y) for y in ty)}->{t}" for _, _, ty, t in splits))

    all_fold_rows = []
    all_predictions = []

    for train_idx, test_idx, train_years, test_year in splits:
        train_label = "+".join(str(y) for y in train_years)
        print(f"\n{'='*65}")
        print(f"Fold: train=[{train_label}]  ->  test={test_year}")
        train = data.loc[train_idx]
        test  = data.loc[test_idx]
        print(f"  Train: {train['RaceID'].nunique()} races  "
              f"| Test: {test['RaceID'].nunique()} races")

        mm, bm, dm, preds = train_and_evaluate_fold(train, test, FEATURES)

        print(f"\n  {'Metric':<14} {'Model':>10} {'Baseline':>10} {'Delta':>10}")
        for key in ["Spearman", "KendallTau", "Top3 Acc", "Top5 Acc", "Winner Acc", "NDCG@3", "NDCG@5"]:
            delta = mm[key] - bm[key]
            sign = "+" if delta >= 0 else ""
            print(f"  {key:<14} {mm[key]:>10.4f} {bm[key]:>10.4f} {sign}{delta:>9.4f}")
        print(f"  DNF LogLoss: {dm['DNF_LogLoss']:.4f}  Brier: {dm['DNF_Brier']:.4f}")

        row = {"TrainYears": train_label, "TestYear": test_year,
               "N_Races": mm["N_Races"]}
        for k, v in mm.items():
            if k != "N_Races":
                row[f"Model_{k}"] = v
        for k, v in bm.items():
            if k != "N_Races":
                row[f"BL_{k}"] = v
        row.update(dm)
        all_fold_rows.append(row)
        preds["TrainYears"] = train_label
        preds["TestYear"]   = test_year
        all_predictions.append(preds)

    summary_df      = pd.DataFrame(all_fold_rows)
    predictions_df  = pd.concat(all_predictions, ignore_index=True)

    print(f"\n{'='*65}")
    print("MEAN ACROSS ALL FOLDS")
    mean_cols = [c for c in summary_df.columns
                 if c.startswith("Model_") or c.startswith("BL_")]
    print(summary_df[mean_cols].mean().round(4).to_string())

    summary_df.to_csv("v2_walkforward_summary.csv", index=False)
    predictions_df.to_csv("v2_predictions.csv", index=False)

    # ── VIF selection ───────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("VIF BACKWARD SELECTION (reference: 2022+2023+2024 training data)")
    ref_data = data[data["Year"].isin([2022, 2023, 2024])]
    vif_selected, vif_steps_df, final_vif_df = vif_backward_selection(
        ref_data, FEATURES, threshold=5.0
    )
    vif_steps_df.to_csv("v2_vif_steps.csv", index=False)
    final_vif_df.to_csv("v2_vif_final.csv", index=False)

    # ── Walk-forward with VIF-selected features ─────────────────────────────
    print(f"\n{'='*65}")
    print(f"WALK-FORWARD WITH VIF-SELECTED FEATURES ({len(vif_selected)})")
    vif_fold_rows = []
    for train_idx, test_idx, train_years, test_year in splits:
        train_label = "+".join(str(y) for y in train_years)
        print(f"\nFold: train=[{train_label}]  ->  test={test_year}")
        train = data.loc[train_idx]
        test  = data.loc[test_idx]
        mm, bm, dm, _ = train_and_evaluate_fold(train, test, vif_selected)
        row = {"TrainYears": train_label, "TestYear": test_year,
               "N_Features": len(vif_selected), "N_Races": mm["N_Races"]}
        for k, v in mm.items():
            if k != "N_Races": row[f"Model_{k}"] = v
        for k, v in bm.items():
            if k != "N_Races": row[f"BL_{k}"] = v
        row.update(dm)
        vif_fold_rows.append(row)
        print(f"  Spearman: {mm['Spearman']:.4f}  (BL: {bm['Spearman']:.4f})")

    vif_summary_df = pd.DataFrame(vif_fold_rows)
    vif_summary_df.to_csv("v2_vif_summary.csv", index=False)

    print(f"\n{'='*65}")
    print("VIF FOLD MEAN")
    vif_mean_cols = [c for c in vif_summary_df.columns
                     if c.startswith("Model_") or c.startswith("BL_")]
    print(vif_summary_df[vif_mean_cols].mean().round(4).to_string())

    # ── LOO analysis ────────────────────────────────────────────────────────
    if RUN_LOO:
        loo_df = leave_one_out_analysis(data, FEATURES, splits)
        loo_df.to_csv("v2_loo_results.csv", index=False)
        print("\nSaved: v2_loo_results.csv")

    # ── Final model + feature importance ────────────────────────────────────
    print("\nTraining final model on all data...")
    _, _, _, importance = train_final_model(data, FEATURES)
    print("\n===== Feature Importance =====")
    print(importance.to_string(index=False))
    importance.to_csv("v2_feature_importance.csv", index=False)

    print("\nSaved: v2_walkforward_summary.csv, v2_predictions.csv, "
          "v2_feature_importance.csv, v2_vif_steps.csv, v2_vif_final.csv, "
          "v2_vif_summary.csv" +
          (", v2_loo_results.csv" if RUN_LOO else ""))


if __name__ == "__main__":
    main()

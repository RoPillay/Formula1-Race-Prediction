import warnings
warnings.filterwarnings("ignore")

import fastf1
import numpy as np
import pandas as pd

from scipy.stats import spearmanr, kendalltau
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss, ndcg_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRanker
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ==============================================================
# FASTF1 CACHE
# ==============================================================

fastf1.Cache.enable_cache(
    r"C:\Users\Owner\OneDrive\Download1\Research STA199\cache"
)


# ==============================================================
# CONFIG
# ==============================================================

YEARS = [2021, 2022, 2023, 2024, 2025]
EVAL_YEARS = [2022, 2023, 2024, 2025]

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


# Higher = easier to overtake
OVERTAKE_INDEX = {
    "Monaco": 0.20,
    "Hungary": 0.35,
    "Singapore": 0.35,
    "Netherlands": 0.40,
    "Emilia Romagna": 0.40,
    "Spain": 0.55,
    "Japan": 0.55,
    "Qatar": 0.55,
    "Australia": 0.60,
    "Miami": 0.60,
    "France": 0.60,
    "Mexico": 0.65,
    "Abu Dhabi": 0.65,
    "China": 0.65,
    "Bahrain": 0.70,
    "Canada": 0.70,
    "Great Britain": 0.70,
    "United States": 0.70,
    "Saudi Arabia": 0.75,
    "Austria": 0.75,
    "Azerbaijan": 0.75,
    "Brazil": 0.75,
    "Italy": 0.80,
    "Las Vegas": 0.80,
    "Belgium": 0.85,
}



# ==============================================================
# HELPERS
# ==============================================================
def normalize_event_location(name):
    if pd.isna(name):
        return None

    s = str(name).lower()

    if "bahrain" in s or "sakhir" in s:
        return "Bahrain"
    if "saudi" in s or "jeddah" in s:
        return "Saudi Arabia"
    if "australia" in s or "melbourne" in s:
        return "Australia"
    if "japan" in s or "suzuka" in s:
        return "Japan"
    if "china" in s or "shanghai" in s:
        return "China"
    if "miami" in s:
        return "Miami"
    if "emilia" in s or "imola" in s:
        return "Emilia Romagna"
    if "monaco" in s:
        return "Monaco"
    if "canada" in s or "montreal" in s or "montréal" in s or "gilles" in s:
        return "Canada"
    if "spain" in s or "barcelona" in s:
        return "Spain"
    if "austria" in s or "spielberg" in s:
        return "Austria"
    if "great britain" in s or "silverstone" in s or "british" in s:
        return "Great Britain"
    if "hungary" in s or "budapest" in s:
        return "Hungary"
    if "belgium" in s or "spa" in s:
        return "Belgium"
    if "netherlands" in s or "zandvoort" in s or "dutch" in s:
        return "Netherlands"
    if "italy" in s or "monza" in s:
        return "Italy"
    if "azerbaijan" in s or "baku" in s:
        return "Azerbaijan"
    if "singapore" in s or "marina bay" in s:
        return "Singapore"
    if "united states" in s or "austin" in s or "cota" in s:
        return "United States"
    if "mexico" in s or "mexico city" in s:
        return "Mexico"
    if "france" in s or "french" in s or "castellet" in s or "paul ricard" in s:
        return "France"
    if "brazil" in s or "são paulo" in s or "sao paulo" in s or "interlagos" in s:
        return "Brazil"
    if "las vegas" in s:
        return "Las Vegas"
    if "qatar" in s or "lusail" in s:
        return "Qatar"
    if "abu dhabi" in s or "yas" in s:
        return "Abu Dhabi"

    return str(name)

def safe_seconds(x):
    try:
        if pd.isna(x):
            return np.nan
        return x.total_seconds()
    except Exception:
        return np.nan


def normalize_team(name):
    if pd.isna(name):
        return np.nan

    name = str(name).lower()

    if "mclaren" in name:
        return "McLaren"
    if "red bull" in name:
        return "Red Bull"
    if "ferrari" in name and "haas" not in name:
        return "Ferrari"
    if "mercedes" in name:
        return "Mercedes"
    if "aston martin" in name:
        return "Aston Martin"
    if "alpine" in name or "renault" in name:
        return "Alpine"
    if "williams" in name:
        return "Williams"
    if "alphatauri" in name or "rb" in name or "racing bulls" in name or "visa" in name:
        return "RB"
    if "sauber" in name or "alfa romeo" in name or "kick" in name:
        return "Sauber"
    if "haas" in name:
        return "Haas"

    return str(name)


def is_dnf_status(status):
    status = str(status).lower()

    bad_words = [
        "accident",
        "collision",
        "engine",
        "gearbox",
        "hydraulics",
        "retired",
        "dnf",
        "mechanical",
        "brakes",
        "electrical",
        "power unit",
        "oil",
        "water",
        "fuel",
        "disqualified",
        "suspension"
    ]

    return int(any(w in status for w in bad_words))


def get_event_name(year, rnd):
    return f"{year}_Round_{rnd}"


def get_rounds_for_year(year):
    if year == 2021:
        return list(range(1,23))
    if year == 2022:
        return list(range(1, 23))   # 22 races
    if year == 2023:
        return list(range(1, 23))   # 22 races
    if year == 2024:
        return list(range(1, 25))   # 24 races
    if year == 2025:
        return list(range(1, 25))   # adjust if some 2025 rounds are unavailable
    return list(range(1, 25))


# ==============================================================
# PRACTICE LONG-RUN FEATURES
# ==============================================================

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
                "FP_LongRunVar": long_run["LapSeconds"].std()
            })

    if not rows:
        return pd.DataFrame(columns=["Driver", "FP_LongRunPace", "FP_LongRunVar"])

    out = pd.DataFrame(rows)
    out = out.groupby("Driver", as_index=False).mean(numeric_only=True)
    return out


# ==============================================================
# QUALIFYING FEATURES
# ==============================================================

def extract_quali_features(year, rnd):
    try:
        quali = fastf1.get_session(year, rnd, "Q")
        quali.load()
        laps = quali.laps
        results = quali.results
    except Exception:
        return pd.DataFrame()

    rows = []

    fastest = (
        laps.groupby("Driver")["LapTime"]
        .min()
        .dropna()
        .apply(safe_seconds)
    )

    if fastest.empty:
        return pd.DataFrame()

    pole = fastest.min()

    for _, row in results.iterrows():
        drv = row.get("Abbreviation")
        team = normalize_team(row.get("TeamName"))

        if pd.isna(drv):
            continue

        quali_pos = row.get("Position", np.nan)
        quali_time = fastest.get(drv, np.nan)

        rows.append({
            "Driver": drv,
            "Team": team,
            "QualiPosition": quali_pos,
            "QualiTime": quali_time,
            "QualiGapToPole": quali_time - pole if not pd.isna(quali_time) else np.nan
        })

    qdf = pd.DataFrame(rows)

    teammate_gaps = []

    for team, g in qdf.groupby("Team"):
        if len(g) != 2:
            for _, r in g.iterrows():
                teammate_gaps.append((r["Driver"], np.nan))
            continue

        g = g.sort_values("QualiTime")
        d1, d2 = g.iloc[0], g.iloc[1]

        teammate_gaps.append((d1["Driver"], d1["QualiTime"] - d2["QualiTime"]))
        teammate_gaps.append((d2["Driver"], d2["QualiTime"] - d1["QualiTime"]))

    gap_df = pd.DataFrame(teammate_gaps, columns=["Driver", "TeammateQualiGap"])
    qdf = qdf.merge(gap_df, on="Driver", how="left")

    return qdf


# ==============================================================
# RACE FEATURES / TARGETS
# ==============================================================

def extract_race_features(year, rnd):
    try:
        race = fastf1.get_session(year, rnd, "R")
        race.load(laps=False, telemetry=False, weather=False, messages=False)

        results = race.results
        event_name = race.event.get("EventName", f"{year}_Round_{rnd}")
        event_location = race.event.get("Location", event_name)

    except Exception as e:
        print(f"Race load failed {year} Round {rnd}: {e}")
        return pd.DataFrame()

    rows = []

    for _, row in results.iterrows():
        drv = row.get("Abbreviation")
        team = normalize_team(row.get("TeamName"))

        if pd.isna(drv):
            continue

        status = row.get("Status", "")
        dnf = is_dnf_status(status)
        finish_pos = row.get("Position", np.nan)

        rows.append({
            "Driver": drv,
            "Team": team,
            "FinishPosition": finish_pos,
            "DNF": dnf,
            "EventName": event_name,
            "EventLocation": event_location
        })

    return pd.DataFrame(rows)

# ==============================================================
# WEATHER FEATURES
# ==============================================================

def extract_weather_features(year, rnd):
    try:
        race = fastf1.get_session(year, rnd, "R")
        race.load()
        w = race.weather_data
    except Exception:
        return {
            "AirTemp": np.nan,
            "TrackTemp": np.nan,
            "Rainfall": 0
        }

    if w is None or w.empty:
        return {
            "AirTemp": np.nan,
            "TrackTemp": np.nan,
            "Rainfall": 0
        }

    return {
        "AirTemp": w["AirTemp"].mean() if "AirTemp" in w.columns else np.nan,
        "TrackTemp": w["TrackTemp"].mean() if "TrackTemp" in w.columns else np.nan,
        "Rainfall": float(w["Rainfall"].mean() > 0) if "Rainfall" in w.columns else 0
    }

    
# ==============================================================
# BUILD RAW DRIVER-RACE DATASET
# ==============================================================

def build_raw_dataset():
    all_rows = []
    load_log = []

    for year in YEARS:
        rounds = get_rounds_for_year(year)

        for rnd in rounds:
            print(f"Loading {year} Round {rnd}...")

            event_name = get_event_name(year, rnd)

            quali = extract_quali_features(year, rnd)
            race = extract_race_features(year, rnd)
            fp = extract_practice_features(year, rnd)
            # tire = extract_tire_degradation_proxy(year, rnd)
            weather = extract_weather_features(year, rnd)

            print(
                f"{year} Round {rnd}: "
                f"quali_rows={len(quali)}, race_rows={len(race)}, "
            #    f"fp_rows={len(fp)}, tire_rows={len(tire)}"
            )

            if quali.empty or race.empty:
                print(
                    f"SKIPPED {year} Round {rnd}: "
                    f"quali_empty={quali.empty}, race_empty={race.empty}"
                )

                load_log.append({
                    "Year": year,
                    "Round": rnd,
                    "Loaded": False,
                    "QualiRows": len(quali),
                    "RaceRows": len(race),
                    "FPRows": len(fp)
                   # "TireRows": len(tire)
                })
                continue

            df = race.merge(quali, on=["Driver", "Team"], how="left")
            df = df.merge(fp, on="Driver", how="left")
           # df = df.merge(tire, on="Driver", how="left")

            df["Year"] = year
            df["Round"] = rnd
            df["RaceID"] = f"{year}_{rnd}"
            

            df["AirTemp"] = weather["AirTemp"]
            df["TrackTemp"] = weather["TrackTemp"]
            df["Rainfall"] = weather["Rainfall"]

            df["EventName"] = df["EventName"].fillna(f"{year}_Round_{rnd}")
            df["EventLocation"] = df["EventLocation"].fillna(df["EventName"])

            raw_location = df["EventLocation"].iloc[0]
            track_key = normalize_event_location(raw_location)

            df["TrackKey"] = track_key
            df["OvertakeIndex"] = OVERTAKE_INDEX.get(track_key, 0.60)

            if track_key not in OVERTAKE_INDEX:
                print(f"WARNING: OvertakeIndex default used for {year} Round {rnd}: {raw_location} → {track_key}")

            all_rows.append(df)

            load_log.append({
                "Year": year,
                "Round": rnd,
                "Loaded": True,
                "QualiRows": len(quali),
                "RaceRows": len(race),
                "FPRows": len(fp)
              #  "TireRows": len(tire)
            })

    pd.DataFrame(load_log).to_csv("f1_load_log.csv", index=False)

    if not all_rows:
        raise ValueError("No races loaded. Check FastF1 session loading.")

    data = pd.concat(all_rows, ignore_index=True)

    data = data.dropna(subset=["FinishPosition"])
    data["FinishPosition"] = data["FinishPosition"].astype(float)

    print("\nRAW DATASET CHECK")
    print("Rows:", len(data))
    print("Unique races:", data["RaceID"].nunique())
    print("Races per year:")
    print(data.groupby("Year")["RaceID"].nunique())
    print("Rows per year:")
    print(data.groupby("Year").size())

    return data


# ==============================================================
# HISTORICAL ROLLING FEATURES
# ==============================================================

def add_rolling_features(data):
    data = data.sort_values(["Year", "Round", "RaceID"]).copy()

    POINTS_MAP = {
        1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
        6: 8, 7: 6, 8: 4, 9: 2, 10: 1
    }

    data["Points"] = data["FinishPosition"].map(POINTS_MAP).fillna(0)

    data["DriverForm"] = np.nan
    data["TeamStrength"] = np.nan
    data["DriverDNFRate"] = np.nan
    data["TeamDNFRate"] = np.nan

    for idx, row in data.iterrows():
        year = row["Year"]
        rnd = row["Round"]
        drv = row["Driver"]
        team = row["Team"]

        # --------------------------
        # DRIVER FORM
        # Last 3 races overall.
        # Allows previous season races as fallback.
        # --------------------------
        driver_past = data[
            (
                (data["Year"] < year) |
                ((data["Year"] == year) & (data["Round"] < rnd))
            ) &
            (data["Driver"] == drv)
        ].tail(3)

        if len(driver_past) > 0:
            data.loc[idx, "DriverForm"] = driver_past["Points"].mean()
            data.loc[idx, "DriverDNFRate"] = driver_past["DNF"].mean()

        # --------------------------
        # TEAM STRENGTH
        # Use current-season rolling team form once enough data exists.
        # Otherwise fallback to previous-season team performance.
        # --------------------------
        current_team_past = data[
            (data["Year"] == year) &
            (data["Round"] < rnd) &
            (data["Team"] == team)
        ]

        previous_team_past = data[
            (data["Year"] < year) &
            (data["Team"] == team)
        ]

        # 6 rows = roughly 3 races for a 2-driver team
        if len(current_team_past) >= 6:
            team_source = current_team_past
        else:
            team_source = previous_team_past.tail(6)

        if len(team_source) > 0:
            data.loc[idx, "TeamStrength"] = team_source["Points"].mean()
            data.loc[idx, "TeamDNFRate"] = team_source["DNF"].mean()

    return data

# ==============================================================
# FINAL FEATURE CLEANING
# ==============================================================

def finalize_features(data):
    data = data.copy()

    data = add_rolling_features(data)
    data["QualiPercentile"] = (data["QualiPosition"] - 1) / 19

    # Interaction terms
    data["Quali_x_Overtake"] = data["QualiPercentile"] * data["OvertakeIndex"]
    # data["Pace_x_TireDeg"] = data["FP_LongRunPace"] * data["TireDegProxy"]
    data["Reliability_x_Rain"] = data["TeamDNFRate"] * data["Rainfall"]

    # More pre-race interaction features
    data["OvertakeOpportunity"] = (1 - data["QualiPercentile"]) * data["OvertakeIndex"]

    data["Driver_vs_Team"] = data["DriverForm"] - data["TeamStrength"]

    data["GridSpread"] = data.groupby("RaceID")["QualiGapToPole"].transform("std")

    # Impute missing values
    for col in FEATURES:
        if col not in data.columns:
            data[col] = np.nan

        data[col] = data[col].replace([np.inf, -np.inf], np.nan)

        if data[col].isna().all():
            data[col] = 0
        else:
            data[col] = data[col].fillna(data[col].median())

    data = data.dropna(subset=["FinishPosition"])

    return data


# ==============================================================
# TWO-STAGE MODEL
# ==============================================================

def evaluate_predictions(test_df, pred_score, dnf_prob):
    out = test_df.copy()
    out["PredScore"] = pred_score
    out["PredDNFProb"] = dnf_prob

    spearmans = []
    kendalls = []
    top3_accs = []
    top5_accs = []
    winner_hits = []
    ndcg3s = []
    ndcg5s = []

    for race_id, g in out.groupby("RaceID"):
        if len(g) < 10:
            continue

        g = g.copy()

        # Higher PredScore = better
        g["PredRank"] = g["PredScore"].rank(ascending=False, method="first")
        g["ActualRank"] = g["FinishPosition"].rank(ascending=True, method="first")

        rho, _ = spearmanr(g["PredRank"], g["ActualRank"])
        tau, _ = kendalltau(g["PredRank"], g["ActualRank"])

        if np.isfinite(rho):
            spearmans.append(rho)
        if np.isfinite(tau):
            kendalls.append(tau)

        pred_top3 = set(g.sort_values("PredScore", ascending=False).head(3)["Driver"])
        true_top3 = set(g.sort_values("FinishPosition").head(3)["Driver"])

        pred_top5 = set(g.sort_values("PredScore", ascending=False).head(5)["Driver"])
        true_top5 = set(g.sort_values("FinishPosition").head(5)["Driver"])

        top3_accs.append(len(pred_top3 & true_top3) / 3)
        top5_accs.append(len(pred_top5 & true_top5) / 5)

        pred_winner = g.sort_values("PredScore", ascending=False).iloc[0]["Driver"]
        true_winner = g.sort_values("FinishPosition").iloc[0]["Driver"]
        winner_hits.append(int(pred_winner == true_winner))

        relevance = (21 - g["FinishPosition"]).values.reshape(1, -1)
        scores = g["PredScore"].values.reshape(1, -1)

        ndcg3s.append(ndcg_score(relevance, scores, k=3))
        ndcg5s.append(ndcg_score(relevance, scores, k=5))

    metrics = {
        "Mean Spearman": np.mean(spearmans),
        "Mean KendallTau": np.mean(kendalls),
        "Top3 Accuracy": np.mean(top3_accs),
        "Top5 Accuracy": np.mean(top5_accs),
        "Winner Accuracy": np.mean(winner_hits),
        "NDCG@3": np.mean(ndcg3s),
        "NDCG@5": np.mean(ndcg5s),
        "DNF LogLoss": log_loss(out["DNF"], np.clip(dnf_prob, 1e-5, 1 - 1e-5)),
        "DNF Brier": brier_score_loss(out["DNF"], dnf_prob)
    }

    return metrics, out


def train_two_stage_model(data):
    data = data.copy()

    groups = data["RaceID"].values
    unique_races = data["RaceID"].unique()

    gkf = GroupKFold(n_splits=5)

    all_metrics = []
    all_predictions = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(data, groups=groups), start=1):
        print(f"\n================ Fold {fold} ================")

        train = data.iloc[train_idx].copy()
        test = data.iloc[test_idx].copy()

        scale_features = [f for f in FEATURES if f != "QualiPercentile"]

        scaler = StandardScaler()

        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(train[scale_features]),
            columns=scale_features,
            index=train.index
        )

        X_test_scaled = pd.DataFrame(
            scaler.transform(test[scale_features]),
            columns=scale_features,
            index=test.index
        )

        X_train_scaled["QualiPercentile"] = train["QualiPercentile"]
        X_test_scaled["QualiPercentile"] = test["QualiPercentile"]

        X_train = X_train_scaled[FEATURES]
        X_test = X_test_scaled[FEATURES]

        # --------------------------
        # Stage 1: DNF Model
        # --------------------------
        dnf_model = LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        )

        dnf_model.fit(X_train, train["DNF"])
        train_dnf_prob = dnf_model.predict_proba(X_train)[:, 1]
        test_dnf_prob = dnf_model.predict_proba(X_test)[:, 1]

        # --------------------------
        # Stage 2: Ranking Model
        # --------------------------
        train_rank = train.copy()
        test_rank = test.copy()

        train_rank["DNFProbFeature"] = train_dnf_prob
        test_rank["DNFProbFeature"] = test_dnf_prob

        rank_features = FEATURES + ["DNFProbFeature"]

        X_train_rank = train_rank[rank_features]
        X_test_rank = test_rank[rank_features]

        # Higher label = better finish
        y_train_rank = 21 - train_rank["FinishPosition"]

        train_group_sizes = train_rank.groupby("RaceID").size().values

        ranker = XGBRanker(
            objective="rank:ndcg",
            eval_metric="ndcg@3",
            n_estimators=500,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=5.0,
            reg_alpha=0.5,
            random_state=42
        )

        ranker.fit(
            X_train_rank,
            y_train_rank,
            group=train_group_sizes
        )

        raw_score = ranker.predict(X_test_rank)

        # --------------------------
        # Two-stage final adjustment
        # --------------------------
        # Higher score = better predicted result
        final_score = raw_score - (2.0 * test_dnf_prob)

        metrics, preds = evaluate_predictions(
            test_rank,
            final_score,
            test_dnf_prob
        )

        print(metrics)

        all_metrics.append(metrics)
        all_predictions.append(preds)

    metrics_df = pd.DataFrame(all_metrics)
    predictions_df = pd.concat(all_predictions, ignore_index=True)

    return metrics_df, predictions_df


# ==============================================================
# FEATURE IMPORTANCE FINAL MODEL
# ==============================================================

def train_final_model(data):
    scale_features = [f for f in FEATURES if f != "QualiPercentile"]

    scaler = StandardScaler()

    X_scaled = pd.DataFrame(
        scaler.fit_transform(data[scale_features]),
        columns=scale_features,
        index=data.index
    )

    X_scaled["QualiPercentile"] = data["QualiPercentile"]
    X = X_scaled[FEATURES]

    dnf_model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    )
    dnf_model.fit(X, data["DNF"])

    dnf_prob = dnf_model.predict_proba(X)[:, 1]

    final_data = data.copy()
    final_data["DNFProbFeature"] = dnf_prob

    rank_features = FEATURES + ["DNFProbFeature"]

    y_rank = 21 - final_data["FinishPosition"]
    group_sizes = final_data.groupby("RaceID").size().values

    ranker = XGBRanker(
        objective="rank:ndcg",
        eval_metric="ndcg@3",
        n_estimators=500,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=5.0,
        reg_alpha=0.5,
        random_state=42
    )

    ranker.fit(
        final_data[rank_features],
        y_rank,
        group=group_sizes
    )

    importance = pd.DataFrame({
        "Feature": rank_features,
        "Importance": ranker.feature_importances_
    }).sort_values("Importance", ascending=False)

    return dnf_model, ranker, scaler, importance

# ==============================================================
# LOO method
# ==============================================================
def leave_one_out_analysis(data):
    global FEATURES

    print("\n================ LEAVE-ONE-OUT FEATURE ANALYSIS ================")

    # ---- Baseline ----
    print("\nRunning baseline model...")
    base_metrics_df, _ = train_two_stage_model(data)
    base_metrics = base_metrics_df.mean(numeric_only=True)

    results = []

    original_features = FEATURES.copy()

    for feature in original_features:
        print(f"\nRemoving feature: {feature}")

        # Remove one feature
        FEATURES = [f for f in original_features if f != feature]

        # Retrain model
        metrics_df, _ = train_two_stage_model(data)
        new_metrics = metrics_df.mean(numeric_only=True)

        # Compute difference
        diff = new_metrics - base_metrics

        row = {"Feature Removed": feature}

        for metric in base_metrics.index:
            row[f"Baseline {metric}"] = base_metrics[metric]
            row[f"Removed {metric}"] = new_metrics[metric]
            row[f"Delta {metric}"] = new_metrics[metric] - base_metrics[metric]

        results.append(row)

    # Restore FEATURES
    FEATURES = original_features

    results_df = pd.DataFrame(results).sort_values("Delta NDCG@3")
    print("\nLOO Results:")
    print(results_df)

    return results_df

# ==============================================================
# VIF ANALYSIS
# ==============================================================
def vif_backward_selection(data, threshold=5.0):
    print("\n================ VIF BACKWARD SELECTION ================")

    X = data[FEATURES].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=FEATURES
    )

    current_features = FEATURES.copy()

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

        print(f"\nRemoving {max_feature} (VIF={max_vif:.2f})")
        current_features.remove(max_feature)

    print("\nFinal VIF-selected features:")
    print(current_features)

    return current_features

# ==============================================================
# MAIN
# ==============================================================

def main():
    global FEATURES
    print("\nBuilding raw FastF1 dataset...")
    raw = build_raw_dataset()

    print("\nFinalizing enriched features...")
    data = finalize_features(raw)

    print("\n================ FEATURE CORRELATION ANALYSIS ================")

    corr = data[FEATURES].corr()

    # Print top correlations 
    for col in FEATURES:
        high_corr = corr[col][(abs(corr[col]) > 0.7) & (corr[col] != 1)]
        if not high_corr.empty:
            print(f"\nHigh correlations with {col}:")
            print(high_corr.sort_values(ascending=False))

    # check qualifying relationships
    print("\nCorrelation with QualiPercentile:")
    print(corr["QualiPercentile"].sort_values(ascending=False))

    data = data[data["Year"].isin(EVAL_YEARS)].copy()

    print("\nOvertakeIndex mapping check:")
    print(
        data[["Year", "Round", "EventName", "EventLocation", "OvertakeIndex"]]
        .drop_duplicates()
        .sort_values(["Year", "Round"])
        .to_string(index=False)
    )

    
    print("\nFINAL DATASET CHECK")
    print("Total rows:", len(data))
    print("Unique races:", data["RaceID"].nunique())
    print("Races per year:")
    print(data.groupby("Year")["RaceID"].nunique())
    print("Rows per year:")
    print(data.groupby("Year").size())
    print("Drivers per race summary:")
    print(data.groupby("RaceID").size().describe())

    
    front_cols = [
        "Year", "Round", "RaceID", "EventName", "EventLocation", "TrackKey",
        "Driver", "Team",
        "QualiPosition", "QualiTime", "QualiGapToPole", "TeammateQualiGap",
        "FinishPosition", "DNF", "Points",
        "DriverForm", "TeamStrength", "DriverDNFRate", "TeamDNFRate",
        "FP_LongRunPace", "FP_LongRunVar",
        "AirTemp", "TrackTemp", "Rainfall",
        "OvertakeIndex", "QualiPercentile", "Quali_x_Overtake", "Reliability_x_Rain", "OvertakeOpportunity", "Driver_vs_Team", "GridSpread"
    ]

    # Keep any remaining columns after the main readable columns
    remaining_cols = [c for c in data.columns if c not in front_cols]

    data = data[front_cols + remaining_cols]

    # Sort by season, race, and finishing order
    data = data.sort_values(
        ["Year", "Round", "FinishPosition"]
    ).reset_index(drop=True)

    print("\nSaving enriched dataset...")
    data.to_csv("DEBUGF10_f1_enriched_prediction_dataset_2022_2025.csv", index=False)

    print("\nTraining two-stage enriched prediction model...")
    metrics_df, predictions_df = train_two_stage_model(data)

    print("\n================ FINAL CROSS-VALIDATED METRICS ================")
    print(metrics_df.mean(numeric_only=True))

    metrics_df.to_csv("DEBUGF10_f1_two_stage_cv_metrics.csv", index=False)
    predictions_df.to_csv("DEBUGF10_f1_two_stage_predictions.csv", index=False)

    print("\nTraining final model on all data...")
    dnf_model, ranker, scaler, importance = train_final_model(data)

    print("\n================ FEATURE IMPORTANCE ================")
    print(importance)

    importance.to_csv("DEBUGF10_f1_two_stage_feature_importance.csv", index=False)

    print("\nSaved files:")
    print("1. DEBUGF10_f1_enriched_prediction_dataset_2022_2025.csv")
    print("2. DEBUGF10_f1_two_stage_cv_metrics.csv")
    print("3. DEBUGF10_f1_two_stage_predictions.csv")
    print("4. DEBUGF10_f1_two_stage_feature_importance.csv")

    print("\nRunning Leave-One-Out Feature Analysis...")
    loo_results = leave_one_out_analysis(data)

    loo_results.to_csv("DEBUGF10_f1_LOO_results.csv", index=False)

    print("\nRunning VIF Analysis...")
    vif_features = vif_backward_selection(data)

    print("\nTraining two-stage model with VIF-selected features...")

    original_features = FEATURES.copy()

    FEATURES = vif_features

    vif_metrics_df, vif_predictions_df = train_two_stage_model(data)

    print("\n================ VIF MODEL METRICS ================")
    print(vif_metrics_df.mean(numeric_only=True))

    FEATURES = original_features

    data["RaceID"].nunique()
    len(data)

if __name__ == "__main__":
    main()
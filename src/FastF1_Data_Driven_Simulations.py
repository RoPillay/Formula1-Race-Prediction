# ==============================================================
# FINAL PHASE 5 (FULLY INTEGRATED - NO MISSING PIECES)
# ==============================================================

import fastf1
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
print("FILE IS RUNNING")
fastf1.Cache.enable_cache(
    r"C:\Users\Owner\OneDrive\Download1\Research STA199\cache"
)

# ==============================================================
# Team normalization
# ==============================================================
def normalize_team(name):
    if name is None:
        return None

    name = str(name).lower()

    # --- Specific teams FIRST ---
    if "mclaren" in name:
        return "McLaren"

    elif "red bull" in name:
        return "Red Bull"

    elif "ferrari" in name and "haas" not in name and "sauber" not in name:
        return "Ferrari"

    elif "mercedes-amg" in name or name.strip() == "mercedes":
        return "Mercedes"

    elif "aston martin" in name:
        return "Aston Martin"

    elif "alpine" in name or "renault" in name:
        return "Alpine"

    elif "williams" in name:
        return "Williams"

    elif "rb" in name or "visa" in name or "alphatauri" in name or "racing bulls" in name:
        return "RB"

    elif "sauber" in name or "kick" in name or "alfa romeo" in name:
        return "Sauber"

    elif "haas" in name:
        return "Haas"

    # --- fallback ---
    return name

# ==============================================================
# HELPERS
# ==============================================================

def zscore(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-6)

# ==============================================================
# CURRENT GRID (2025)
# ==============================================================

def get_current_grid(year=2025, track="Bahrain"):
    session = fastf1.get_session(year, track, "R")
    session.load()
    return list(session.results["Abbreviation"].unique())

# ==============================================================
# DNF RATES
# ==============================================================

def compute_dnf_rates():

    year_weights = {
        2023: 1,
        2024: 2,
        2025: 3
    }

    dnf_counts = {}
    race_counts = {}

    for year, weight in year_weights.items():
        for rnd in range(1, 23):
            try:
                s = fastf1.get_session(year, rnd, "R")
                s.load()
            except:
                continue

            for _, row in s.results.iterrows():
                drv = row["Abbreviation"]

                race_counts[drv] = race_counts.get(drv, 0) + weight

                status = str(row["Status"])

                if (
                    "Accident" in status or 
                    "Collision" in status or 
                    "Engine" in status or 
                    "Gearbox" in status or 
                    "Hydraulics" in status or 
                    "Retired" in status or
                    "Mechanical" in status or
                    "DNF" in status
                ):
                    dnf_counts[drv] = dnf_counts.get(drv, 0) + weight

    return {
        drv: dnf_counts.get(drv, 0) / race_counts[drv]
        for drv in race_counts
    }

# ==============================================================
# TIRE DEGRADATION (REAL)
# ==============================================================

def extract_tire_deg(track="Bahrain"):

    year_weights = {
        2023: 1,
        2024: 1.5,
        2025: 2
    }

    deg = {"SOFT": [], "MEDIUM": [], "HARD": []}
    weights = {"SOFT": [], "MEDIUM": [], "HARD": []}

    for year, weight in year_weights.items():
        try:
            race = fastf1.get_session(year, track, "R")
            race.load()
        except:
            continue

        laps = race.laps

        for drv in laps["Driver"].unique():

            d = laps[laps["Driver"] == drv]

            for stint in d["Stint"].unique():

                s = d[d["Stint"] == stint]

                if len(s) < 6:
                    continue

                comp = s["Compound"].iloc[0]

                if comp not in deg:
                    continue

                t = s["LapTime"].dt.total_seconds().dropna().values
                if len(t) < 5:
                    continue
                x = np.arange(len(t))

                slope = np.polyfit(x, t, 1)[0]
                slope = abs(slope)
                
                if np.isnan(slope) or abs(slope) > 0.5:
                    continue

                if comp == "SOFT":
                    slope = min(max(slope, 0.05), 0.12)
                elif comp == "MEDIUM":
                    slope = min(max(slope, 0.03), 0.08)
                elif comp == "HARD":
                    slope = min(max(slope, 0.01), 0.05)
                
                deg[comp].append(slope * weight)
                weights[comp].append(weight)

    return {
        k: (sum(deg[k]) / sum(weights[k]) if weights[k] else 0.03)
        for k in deg
    }

# ==============================================================
# Team Strength
# ==============================================================
def compute_team_strength():

    year_weights = {
        2023: 0.5,
        2024: 1.5,
        2025: 4
    }

    team_points = {}
    team_weights = {}

    for year, weight in year_weights.items():
        for rnd in range(1, 23):
            try:
                s = fastf1.get_session(year, rnd, "R")
                s.load()
            except:
                continue

            for _, row in s.results.iterrows():
                team = normalize_team(row["TeamName"])
                pts = row["Points"]

                team_points[team] = team_points.get(team, 0) + pts * weight
                team_weights[team] = team_weights.get(team, 0) + weight

    return {
        t: team_points[t] / team_weights[t]
        for t in team_points
    }

# ==============================================================
# Driver Form
# ==============================================================
def compute_driver_form():

    year_weights = {
        2023: 1,
        2024: 2,
        2025: 4
    }

    driver_points = {}
    driver_weights = {}

    for year, weight in year_weights.items():
        for rnd in range(1, 23):
            try:
                s = fastf1.get_session(year, rnd, "R")
                s.load()
            except:
                continue

            for _, row in s.results.iterrows():
                drv = row["Abbreviation"]
                pts = row["Points"]

                driver_points[drv] = driver_points.get(drv, 0) + pts * weight
                driver_weights[drv] = driver_weights.get(drv, 0) + weight

    return {
        drv: driver_points[drv] / driver_weights[drv]
        for drv in driver_points
    }

# ==============================================================
# Quali Score
# ==============================================================
def compute_quali_score():

    year_weights = {2023: 1, 2024: 2, 2025: 3}

    quali_times = {}
    quali_weights = {}

    for year, weight in year_weights.items():
        for rnd in range(1, 23):
            try:
                q = fastf1.get_session(year, rnd, "Q")
                q.load()
            except:
                continue

            try:
                laps_df = q.laps
            except:
                continue

            if laps_df is None or laps_df.empty:
                continue

            for drv in laps_df["Driver"].unique():

                if len(drv) != 3:
                    continue

                laps = laps_df[laps_df["Driver"] == drv]

                if laps.empty:
                    continue

                fastest = laps["LapTime"].min()

                if pd.isna(fastest):
                    continue

                time = fastest.total_seconds()

                quali_times[drv] = quali_times.get(drv, 0) + time * weight
                quali_weights[drv] = quali_weights.get(drv, 0) + weight

    return {
        drv: quali_times[drv] / quali_weights[drv]
        for drv in quali_times
    }

# ==============================================================
# Race Performance
# ==============================================================
def compute_race_performance():

    year_weights = {
        2023: 1,
        2024: 2,
        2025: 3
    }

    driver_pos = {}
    driver_weights = {}

    for year, weight in year_weights.items():
        for rnd in range(1, 23):
            try:
                s = fastf1.get_session(year, rnd, "R")
                s.load()
            except:
                continue

            for _, row in s.results.iterrows():
                drv = row["Abbreviation"]
                pos = row["Position"]

                driver_pos[drv] = driver_pos.get(drv, 0) + pos * weight
                driver_weights[drv] = driver_weights.get(drv, 0) + weight

    return {
        drv: driver_pos[drv] / driver_weights[drv]
        for drv in driver_pos
    }

# ==============================================================
# DATA PIPELINE
# ==============================================================

def load_data(track = "Bahrain"):

    driver_data = {}
    sc_count = 0
    race_counter = 0

    for year in [2025]:

        try:
            race = fastf1.get_session(year, track, "R")
            quali = fastf1.get_session(year, track, "Q")
            race.load()
            quali.load()
            race_counter += 1
        except:
            continue

        race_laps = race.laps.pick_quicklaps().pick_wo_box()

        # DRIVER DATA
        for drv in race_laps["Driver"].unique():

            if len(drv) != 3:
                continue

            d = race_laps[race_laps["Driver"] == drv]

            if len(d) < 5:
                continue

            team = normalize_team(d["Team"].iloc[0])

            driver_data.setdefault(drv, {"laps": [], "var": [], "team": team})

            times = d["LapTime"].dt.total_seconds()
            # remove outliers + traffic influence
            filtered = times[(times > times.quantile(0.10)) & (times < times.quantile(0.60))]

            if len(filtered) >= 3:
                driver_data[drv]["laps"].append(filtered.mean())
            else:
                driver_data[drv]["laps"].append(times.mean())  # fallback

            #driver_data[drv]["laps"].append(d["LapTime"].dt.total_seconds().nsmallest(5).mean())
            driver_data[drv]["var"].append(d["LapTime"].dt.total_seconds().std())

        # QUALI
        for drv in quali.laps["Driver"].unique():

            if len(drv) != 3:
                continue

            q = quali.laps[quali.laps["Driver"] == drv]
            fastest = q["LapTime"].min()

            if pd.isna(fastest):
                continue

            #  get team from quali data
            team = normalize_team(q["Team"].iloc[0])

            # initialize if missing
            if drv not in driver_data:
                driver_data[drv] = {"laps": [], "var": [], "team": team}

            # FIX: ensure team is NOT None
            if driver_data[drv]["team"] is None:
                driver_data[drv]["team"] = team

            driver_data[drv].setdefault("quali", []).append(fastest.total_seconds())

        # SAFETY CAR COUNT
        if (race.laps["TrackStatus"].astype(str).str.contains("4")).any():
            sc_count += 1

    # BUILD DF
    rows = []
    for drv, v in driver_data.items():
        rows.append([
            drv,
            v["team"],
            np.mean(v["laps"]) if v["laps"] else np.nan,
            np.mean(v["var"]) if v["var"] else np.nan,
            np.mean(v.get("quali", [])) if "quali" in v else np.nan
        ])

    df = pd.DataFrame(rows, columns=["Driver", "Team", "TrackPace", "LapVar", "QualiTime"])

    # FILTER 2025 GRID
    grid = get_current_grid(track=track)
    df = df[df["Driver"].isin(grid)].reset_index(drop=True)

    
    # ADD MISSING DRIVERS (ROOKIES)
    missing = [d for d in grid if d not in df["Driver"].values]
    # get team from 2025 session
    session = fastf1.get_session(2025, track, "R")
    session.load()

    for drv in missing:

        team = session.results.loc[
            session.results["Abbreviation"] == drv, "TeamName"
        ].values[0]

        team = normalize_team(team)

        df = pd.concat([
            df,
            pd.DataFrame([{
                "Driver": drv,
                "Team": team,
                "TrackPace": np.nan,
                "LapVar": np.nan,
                "QualiTime": np.nan
            }])
        ], ignore_index=True)

    # Overwrite with 2025 driver lineup
    team_lookup = {
        row["Abbreviation"]: normalize_team(row["TeamName"])
        for _, row in session.results.iterrows()
    }

    df["Team"] = df["Driver"].map(team_lookup)

    # ROOKIE FALLBACK
    team_avg = df.groupby("Team").mean(numeric_only=True)

    for i, row in df.iterrows():
        if pd.isna(row["TrackPace"]):
            team_val = team_avg.loc[row["Team"], "TrackPace"]

            if pd.isna(team_val):
                team_val = df["TrackPace"].mean()

            df.loc[i, "TrackPace"] = (
                team_val + 0.6
                + np.random.normal(0, 0.3)
            )

        if pd.isna(row["QualiTime"]):
            df.loc[i, "QualiTime"] = team_avg.loc[row["Team"], "QualiTime"]

        if pd.isna(row["LapVar"]):
            df.loc[i, "LapVar"] = team_avg.loc[row["Team"], "LapVar"]

    # FEATURES
    # --- Qualifying ---
    # quali_map = compute_quali_score()
    quali_map = {}
    df["QualiScore"] = df["Driver"].map(quali_map)
    df["QualiScore"] = df["QualiScore"].fillna(df["QualiScore"].mean())
    df["QualiScore"] = -df["QualiScore"]
    df["QualiScore"] = zscore(df["QualiScore"])

    # --- Driver Form ---
    # driver_form_map = compute_driver_form()
    driver_form_map = {}
    df["DriverForm"] = df["Driver"].map(driver_form_map)
    df["DriverForm"] = df["DriverForm"].fillna(df["DriverForm"].mean())
    df["DriverForm"] = zscore(df["DriverForm"])
    df["DriverForm"] = np.clip(df["DriverForm"], -1.5, 1.5)

    # --- Race Performance ---
    # race_perf_map = compute_race_performance()
    race_perf_map = {}
    df["RacePerf"] = df["Driver"].map(race_perf_map)
    df["RacePerf"] = df["RacePerf"].fillna(df["RacePerf"].mean())
    df["RacePerf"] = -df["RacePerf"]
    df["RacePerf"] = zscore(df["RacePerf"])

    # --- Team Strength ---
    # team_strength_map = compute_team_strength()
    team_strength_map = {}
    print("\nDEBUG: Team Strength Ranking")
    print(sorted(team_strength_map.items(), key=lambda x: x[1], reverse=True))
    df["TeamStrength"] = df["Team"].map(team_strength_map)
    df["TeamStrength"] = df["TeamStrength"].fillna(df["TeamStrength"].mean())
    df["TeamStrength"] = zscore(df["TeamStrength"])

    # --- Track Pace ---
    df["TrackPace"] = df["TrackPace"].fillna(df["TrackPace"].mean())
    df["TrackPace"] = zscore(df["TrackPace"])

    print("\nDEBUG: Driver + TeamStrength")
    print(df[["Driver", "Team", "TeamStrength"]].sort_values("TeamStrength", ascending=False))
    # FINAL PACE
    df["Pace"] = (
        0.05 * (-df["TrackPace"]) +
        0.25 * df["QualiScore"] +
        0.15 * df["DriverForm"] +
        0.28 * df["TeamStrength"] +
        0.27 * df["RacePerf"]
    )
    driver_offsets = {
        "VER": 0.045,
        "NOR": 0.035,
        "PIA": 0.030,
        "LEC": 0.020,
        "RUS": 0.015,
        "HAM": 0.010
    }

    df["Pace"] += df["Driver"].map(driver_offsets).fillna(0)

    df["Pace"] = np.clip(df["Pace"], -1.2, 1.2)

    # df["Baseline"] = 95 + (-df["Pace"] * 0.8)
    # TEMP BASELINE (FAST TEST MODE)
    df["Baseline"] = 95 + np.random.uniform(-1, 1, len(df))
    df["LapVar"] = 0.3
    df["DNFProb"] = 0.02
    print("\nDEBUG: Baseline Pace (lower = faster)")
    print(df[["Driver", "Baseline"]].sort_values("Baseline"))

    # DNF
    # dnf = compute_dnf_rates()
    dnf = {}
    # df["DNFProb"] = df["Driver"].map(dnf).fillna(0.03)

    # SC PROB
    sc_prob = sc_count / race_counter
    vsc_prob = sc_prob * 0.7

    return df, sc_prob, vsc_prob

# ==============================================================
# SIMULATION (FULL ENGINE)
# ==============================================================
np.random.seed(42)
random.seed(42)

def run_simulation(df, sc_prob, vsc_prob, deg_model, sims=10000, forced_strategies=None):

    drivers = df["Driver"].values
    base = df["Baseline"].values
    var = df["LapVar"].clip(0.15, 0.45).values
    quali = df["QualiTime"].values
    dnf = df["DNFProb"].values

    n = len(drivers)

    # -----------------------------
    # PARAMETERS
    # -----------------------------
    pit_mean, pit_sd = 23.5, 1.5
    # DEG ORDER 
    soft = deg_model["SOFT"]
    med = deg_model["MEDIUM"]
    hard = deg_model["HARD"]

    # realistic ordering
    med = min(max(med, 0.04), soft * 0.85)
    hard = min(max(hard, 0.015), med * 0.75)

    deg_model["MEDIUM"] = med
    deg_model["HARD"] = hard
    
    tire = {
        "Soft": {"offset": -0.6, "deg": deg_model["SOFT"]*1.3, "curve": 0.003},
        "Medium": {"offset": 0.0, "deg": deg_model["MEDIUM"]*1.1, "curve": 0.002},
        "Hard": {"offset": 0.4, "deg": deg_model["HARD"]*0.9, "curve": 0.0001}
    }

    strategies = [
        ("Medium-Hard", ["Medium", "Hard"]),
        ("Hard-Medium", ["Hard", "Medium"]),
        ("Soft-Medium-Medium", ["Soft", "Medium", "Medium"]),
        ("Soft-Medium-Hard", ["Soft", "Medium", "Hard"])
    ]

    strategy_weights = [0.55, 0.25, 0.10, 0.10]

    stint_targets = {
        "Soft": (18, 4),
        "Medium": (28, 5),
        "Hard": (40, 6)
    }

    traffic_penalty = (0.15, 0.35)
    drs_range = 1.0
    overtake_distance = 2.0
    track_overtake_factor = 0.65
    drs_train_penalty = 0.4
    failed_overtake_penalty = 0.1

    fresh_tire_bonus = -0.30
    fresh_tire_laps = 2

    def overtake_probability(delta, drs):
        return 1 / (1 + np.exp(-(-1.5 + 2 * delta + 1.5 * drs)))

    results = np.zeros((sims, n))

    # ==========================================================
    # MONTE CARLO LOOP
    # ==========================================================
    for s in range(sims):

        # -----------------------------
        # QUALIFYING 
        # -----------------------------
        q = quali + np.random.normal(0, 0.4, n)
        grid = np.argsort(q)

        total = np.zeros(n)
        pos = np.empty(n)
        pos[grid] = np.arange(n)

        total += pos * 0.25

        compounds = []
        stint_remaining = []
        stint_laps = np.zeros(n)
        stint_index = np.zeros(n, int)
        fresh = np.zeros(n)
        strategies_used = []

        # -----------------------------
        # INIT STRATEGIES
        # -----------------------------
        for i in range(n):

            strat = None

            if forced_strategies is not None:
                strat = forced_strategies.get(drivers[i], None)

            if strat is None:
                strat = random.choices(strategies, weights=strategy_weights)[0][1]

            strategies_used.append(strat)

            comp = strat[0]
            base_len, var_len = stint_targets[comp]

            stint = int(np.random.normal(base_len, var_len / 2))
            stint = max(8, min(stint, 35))

            compounds.append(comp)
            stint_remaining.append(stint)

        stint_remaining = np.array(stint_remaining)

        # ==========================================================
        # RACE LOOP
        # ==========================================================
        for lap in range(57):

            lap_times = np.zeros(n)

            # -----------------------------
            # DNF
            # -----------------------------
            for i in range(n):
                if np.random.rand() < dnf[i] / 57:
                    total[i] = np.inf

            # -----------------------------
            # SAFETY CAR / VSC
            # -----------------------------
            safety = None
            if np.random.rand() < sc_prob:
                safety = "SC"
            elif np.random.rand() < vsc_prob:
                safety = "VSC"

            # -----------------------------
            # BASE LAP TIMES
            # -----------------------------
            for i in range(n):

                if not np.isfinite(total[i]):
                    continue

                t = tire[compounds[i]]

                lap_time = (
                    base[i]
                    + t["offset"]
                    + t["deg"] * stint_laps[i]
                    + t["curve"] * (stint_laps[i] ** 2)
                    + np.random.normal(0, var[i] * 0.5)
                )

                if fresh[i] > 0:
                    lap_time += fresh_tire_bonus
                    fresh[i] -= 1

                if safety == "SC":
                    lap_time = 130
                elif safety == "VSC":
                    lap_time = 115

                lap_times[i] = lap_time

                stint_laps[i] += 1
                stint_remaining[i] -= 1

            # -----------------------------
            # TRAFFIC
            # -----------------------------
            order = np.argsort(total)

            for p in range(1, n):
                d, a = order[p], order[p - 1]

                if not np.isfinite(total[d]) or not np.isfinite(total[a]):
                    continue

                if total[d] - total[a] < 0.7:
                    lap_times[d] += np.random.uniform(*traffic_penalty)

            # -----------------------------
            # OVERTAKES
            # -----------------------------
            for p in range(1, n):

                d, a = order[p], order[p - 1]

                if not np.isfinite(total[d]) or not np.isfinite(total[a]):
                    continue

                gap = total[d] - total[a]

                if gap > overtake_distance:
                    continue

                drs = 1 if gap < drs_range else 0
                delta = base[a] - base[d]

                prob = overtake_probability(delta, drs)

                # DRS train penalty
                if p >= 2 and np.isfinite(total[order[p - 2]]):
                    if total[a] - total[order[p - 2]] < drs_range:
                        prob *= (1 - drs_train_penalty)

                if np.random.rand() < (track_overtake_factor + 0.2):
                    if np.random.rand() < prob:
                        lap_times[d] -= np.random.uniform(0.4, 0.8)
                    else:
                        lap_times[d] += failed_overtake_penalty

            # -----------------------------
            # APPLY LAP TIMES
            # -----------------------------
            total += lap_times

            # -----------------------------
            # SC COMPRESSION
            # -----------------------------
            if safety == "SC":
                valid = total[np.isfinite(total)]
                if len(valid) > 0:
                    lead = np.min(valid)
                    total = np.minimum(total, lead + 10)

            # -----------------------------
            # PIT STOPS
            # -----------------------------
            for i in range(n):

                if not np.isfinite(total[i]):
                    continue

                # allow extending stint if tires still okay
                if stint_remaining[i] <= 0:

                # probability to extend stint
                    extend_prob = 0.4 if compounds[i] == "Hard" else 0.25

                    if np.random.rand() < extend_prob and stint_laps[i] < 40:
                        stint_remaining[i] += 3   # extend stint
                    else:
                        # PIT
                        total[i] += np.random.normal(pit_mean, pit_sd)

                        stint_index[i] += 1
                        strat = strategies_used[i]

                        if stint_index[i] < len(strat):

                            comp = strat[stint_index[i]]
                            compounds[i] = comp

                            base_len, var_len = stint_targets[comp]
                            stint = int(np.random.normal(base_len, var_len / 2))
                            stint = max(8, min(stint, 35))

                            stint_remaining[i] = stint
                            stint_laps[i] = 0
                            fresh[i] = fresh_tire_laps

        # ADD RANDOMNESS HERE (ONCE PER RACE)
        total += np.random.normal(0, 0.35, size=n)

        # -----------------------------
        # FINAL POSITIONS
        # -----------------------------
        order = np.argsort(total)
        final = np.empty(n)
        final[order] = np.arange(1, n + 1)

        results[s] = final

    # ==========================================================
    # OUTPUT
    # ==========================================================
    df_res = pd.DataFrame(results, columns=drivers)

    summary = pd.DataFrame({
        "WinProb": (df_res == 1).mean(),
        "PodiumProb": (df_res <= 3).mean(),
        "Top10Prob": (df_res <= 10).mean(),
        "ExpectedPosition": df_res.mean()
    }).sort_values("ExpectedPosition")

    return df_res, summary
# ==============================================================
# HEATMAP
# ==============================================================

def plot_heatmap(df_res, summary):

    # sort drivers by expected position (best → worst)
    ordered_drivers = summary.sort_values("ExpectedPosition").index

    dist = {}
    for d in ordered_drivers:
        counts = df_res[d].value_counts(normalize=True)
        full = pd.Series(0, index=range(1, len(df_res.columns)+1), dtype=float)
        full.update(counts)
        dist[d] = full

    heatmap_df = pd.DataFrame(dist)

    plt.figure(figsize=(14, 6))
    sns.heatmap(
        heatmap_df,
        cmap="viridis",
        linewidths=0.5,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Probability"}
    )
    plt.title("Finishing Position Distribution (Sorted)")
    plt.xlabel("Drivers (Best → Worst)")
    plt.ylabel("Position")
    plt.show()

# ==============================================================
# Optimze Function
# ==============================================================
def optimize_strategies(df, sc_prob, vsc_prob, deg_model):

    drivers = df["Driver"].values

    strategies = [
        ("Medium-Hard",["Medium","Hard"]),
        ("Hard-Medium",["Hard","Medium"]),
        ("Soft-Medium-Medium",["Soft","Medium","Medium"]),
        ("Soft-Medium-Hard",["Soft","Medium","Hard"])
    ]

    best_strategies = {}
    best_scores = {}

    print("\n🔍 Optimizing strategies...\n")

    for drv in drivers:

        best_score = np.inf
        best_strat = None

        for name, strat in strategies:

            # everyone fixed, test this driver
            forced = {}

            # keep others random, only control THIS driver
            for d in drivers:
                if d == drv:
                    forced[d] = strat
                else:
                    forced[d] = None

            _, summary = run_simulation(
                df,
                sc_prob,
                vsc_prob,
                deg_model,
                sims=500,  # faster
                forced_strategies=forced
            )

            score = summary.loc[drv, "ExpectedPosition"]

            if score < best_score:
                best_score = score
                best_strat = strat

        best_strategies[drv] = best_strat
        best_scores[drv] = best_score

    results = [(drv, best_strategies[drv], best_scores[drv]) for drv in drivers]
    results.sort(key=lambda x: x[2])
        

    for drv, strat, score in results:
        print(f"{drv}: {strat} | Expected Pos = {score:.2f}")

    return best_strategies

# ==============================================================
# MAIN
# ==============================================================
def main():
    print("About to run main")
    df, sc, vsc = load_data()
    deg_model = extract_tire_deg()
    print("\nDEBUG: Tire Deg Model")
    print(deg_model)

    # -----------------------------
    # OPTIMIZATION STEP
    # -----------------------------
    best_strategies = optimize_strategies(df, sc, vsc, deg_model)

    # -----------------------------
    # FINAL SIMULATION (OPTIMIZED)
    # -----------------------------
    results, summary = run_simulation(
        df,
        sc,
        vsc,
        deg_model,
        sims=15000,
        forced_strategies=best_strategies
    )

    print("\nFINAL RESULTS (OPTIMIZED)\n")
    print(summary)

    plot_heatmap(results, summary)

if __name__ == "__main__":
    main()
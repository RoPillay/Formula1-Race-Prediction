import fastf1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

# ==============================================================
# 1. CONFIGURATION
# ==============================================================

CACHE_DIR = "cache"
N_SIMULATIONS = 1000

fastf1.Cache.enable_cache(CACHE_DIR)

# ==============================================================
# 2. DATA LOADER
# ==============================================================

def load_session(year, race_name, session_type="R"):
    print(f"\nLoading {year} {race_name} {session_type} session...")
    session = fastf1.get_session(year, race_name, session_type)
    session.load()
    return session

# ==============================================================
# 3. CLEAN LAP EXTRACTION
# ==============================================================

def get_clean_laps(session, driver):
    laps = session.laps.pick_driver(driver).copy()

    laps = laps[laps["LapTime"].notna()]

    # Remove in/out laps
    laps = laps[laps["PitInTime"].isna()]
    laps = laps[laps["PitOutTime"].isna()]

    # Remove extreme slow laps (SC/VSC)
    laps["LapSeconds"] = laps["LapTime"].dt.total_seconds()
    laps = laps[laps["LapSeconds"] < laps["LapSeconds"].quantile(0.95)]

    return laps

# ==============================================================
# 4. DEGRADATION MODEL
# ==============================================================

def estimate_degradation(laps):
    X = laps["LapNumber"].values.reshape(-1, 1)
    y = laps["LapSeconds"].values

    model = LinearRegression()
    model.fit(X, y)

    slope = model.coef_[0]
    intercept = model.intercept_

    return intercept, slope

# ==============================================================
# 5. PIT STOP ESTIMATION
# ==============================================================

def estimate_pit_loss(session):
    laps = session.laps.copy()

    pit_laps = laps[laps["PitOutTime"].notna()].copy()
    if len(pit_laps) == 0:
        return 20  # fallback average

    # Rough estimate
    pit_loss = 20  # seconds typical
    return pit_loss

# ==============================================================
# 6. MONTE CARLO SIMULATION (SINGLE DRIVER)
# ==============================================================

def simulate_race(base_pace, deg_slope, total_laps, pit_lap, pit_loss, sigma):

    total_time = 0

    for lap in range(1, total_laps + 1):

        noise = np.random.normal(0, sigma)

        lap_time = base_pace + deg_slope * lap + noise

        if lap == pit_lap:
            lap_time += pit_loss

        total_time += lap_time

    return total_time

# ==============================================================
# 7. MAIN SIMULATION ENGINE
# ==============================================================

def run_simulation(year, race_name, driver):

    session = load_session(year, race_name, "R")

    laps = get_clean_laps(session, driver)

    if len(laps) < 10:
        print("Not enough data for simulation.")
        return

    base_pace, deg_slope = estimate_degradation(laps)

    sigma = laps["LapSeconds"].std()

    total_laps = int(laps["LapNumber"].max())

    pit_lap = total_laps // 2  # placeholder strategy

    pit_loss = estimate_pit_loss(session)

    simulated_times = []

    for _ in range(N_SIMULATIONS):
        sim_time = simulate_race(
            base_pace,
            deg_slope,
            total_laps,
            pit_lap,
            pit_loss,
            sigma
        )
        simulated_times.append(sim_time)

    simulated_times = np.array(simulated_times)

    print("\n===== Simulation Results =====")
    print(f"Expected Race Time: {simulated_times.mean():.2f} sec")
    print(f"Std Dev of Race Time: {simulated_times.std():.2f} sec")

    # Plot distribution
    plt.hist(simulated_times, bins=40)
    plt.title(f"{driver} Race Time Distribution")
    plt.xlabel("Total Race Time (sec)")
    plt.ylabel("Frequency")
    plt.show()

# ==============================================================
# 8. RUN
# ==============================================================

if __name__ == "__main__":
    run_simulation(2024, "Bahrain Grand Prix", "VER")
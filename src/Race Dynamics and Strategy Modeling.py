import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random

# -----------------------------
# RACE CONFIGURATION
# -----------------------------

race_name = "Bahrain Grand Prix"
laps = 57
pit_loss = 22

# -----------------------------
# SIMULATION PARAMETERS
# -----------------------------

n_simulations = 50000
lap_time_noise_sd = 0.6

# -----------------------------
# DRIVER BASELINE PACE
# -----------------------------

data = {
    "Driver": [
        "VER","PER","NOR","PIA","LEC","SAI",
        "HAM","RUS","ALO","STR",
        "GAS","OCO","TSU","RIC",
        "ALB","SAR","MAG","HUL",
        "BOT","ZHO"
    ],

    "BaselineLapTime": [
        95.20,95.21,95.22,95.24,
        95.27,95.29,
        95.33,95.35,
        95.40,95.43,
        95.48,95.50,
        95.53,95.55,
        95.58,95.62,
        95.65,95.68,
        95.72,95.75
    ]
}

df = pd.DataFrame(data)

drivers = df["Driver"].values
baseline = df["BaselineLapTime"].values
n_drivers = len(drivers)

# -----------------------------
# TIRE MODEL
# -----------------------------

tire_compounds = {

    "Soft": {
        "pace_offset": -0.4,
        "deg_rate": 0.05,
        "deg_curve": 0.002
    },

    "Medium": {
        "pace_offset": 0.0,
        "deg_rate": 0.035,
        "deg_curve": 0.0015
    },

    "Hard": {
        "pace_offset": 0.3,
        "deg_rate": 0.02,
        "deg_curve": 0.0008
    }
}

# -----------------------------
# STRATEGY SET
# -----------------------------

strategies = [
    ("Medium-Hard", ["Medium","Hard"]),
    ("Hard-Medium", ["Hard","Medium"]),
    ("Soft-Medium-Medium", ["Soft","Medium","Medium"]),
    ("Soft-Medium-Hard", ["Soft","Medium","Hard"])
]

stint_targets = {

    "Soft": (15,4),
    "Medium": (22,5),
    "Hard": (30,6)

}

strategy_weights = [0.35,0.30,0.20,0.15]

# Track strategy performance per driver
strategy_results = {
    s[0]: {driver: [] for driver in drivers}
    for s in strategies
}

# -----------------------------
# MONTE CARLO SIMULATION
# -----------------------------

results = np.zeros((n_simulations, n_drivers))

for sim in range(n_simulations):

    total_times = np.zeros(n_drivers)
    driver_strategy = []

    for i in range(n_drivers):

        strategy_name, strategy = random.choices(
            strategies,
            weights=strategy_weights
        )[0]

        driver_strategy.append(strategy_name)

        race_time = 0
        lap_counter = 0

        stint_index = 0
        compound = strategy[stint_index]
        tire = tire_compounds[compound]

        base, var = stint_targets[compound]

        stint_laps = int(np.random.normal(base, var/2))
        stint_laps = max(8, min(stint_laps, 35))

        stint_lap_counter = 0

        # simulate race lap-by-lap
        while lap_counter < laps:

            noise = np.random.normal(0, lap_time_noise_sd)

            lap_time = (
                baseline[i]
                + tire["pace_offset"]
                + tire["deg_rate"] * stint_lap_counter
                + tire["deg_curve"] * (stint_lap_counter ** 2)
                + noise
            )

            race_time += lap_time

            lap_counter += 1
            stint_lap_counter += 1

            # pit stop condition
            if (
                stint_lap_counter >= stint_laps
                and stint_index < len(strategy) - 1
            ):

                race_time += pit_loss

                stint_index += 1
                compound = strategy[stint_index]
                tire = tire_compounds[compound]

                base, var = stint_targets[compound]
                stint_laps = int(np.random.normal(base, var/2))
                stint_laps = max(8, min(stint_laps, 35))

                stint_lap_counter = 0

        total_times[i] = race_time

    # determine finishing positions
    finishing_order = np.argsort(total_times)

    positions = np.empty(n_drivers)
    positions[finishing_order] = np.arange(1, n_drivers+1)

    results[sim] = positions

    # store strategy results per driver
    for i in range(n_drivers):

        driver = drivers[i]
        strategy = driver_strategy[i]

        strategy_results[strategy][driver].append(positions[i])

# -----------------------------
# RESULTS DATAFRAME
# -----------------------------

results_df = pd.DataFrame(results, columns=drivers)

# -----------------------------
# PROBABILITY METRICS
# -----------------------------

win_prob = (results_df == 1).mean()
podium_prob = (results_df <= 3).mean()
top10_prob = (results_df <= 10).mean()
expected_position = results_df.mean()

summary = pd.DataFrame({

    "WinProb": win_prob,
    "PodiumProb": podium_prob,
    "Top10Prob": top10_prob,
    "ExpectedPosition": expected_position

}).sort_values("ExpectedPosition")

print("\nDriver Outcome Summary")
print(summary)

# -----------------------------
# FINISHING POSITION DISTRIBUTION
# -----------------------------

position_distribution = {}

for driver in drivers:

    counts = (
        results_df[driver]
        .value_counts(normalize=True)
        .reindex(range(1, n_drivers+1), fill_value=0)
    )

    position_distribution[driver] = counts

position_distribution_df = pd.DataFrame(position_distribution)

# -----------------------------
# HEATMAP
# -----------------------------

position_distribution_df = position_distribution_df[summary.index]

plt.figure(figsize=(12,6))

sns.heatmap(
    position_distribution_df,
    cmap="viridis",
    linewidths=0.5,
    annot=True,
    fmt=".2f",
    cbar_kws={"label":"Probability"}
)

plt.title(f"{race_name} Strategy Simulation Heatmap")
plt.xlabel("Driver")
plt.ylabel("Finishing Position")

plt.show()

# -----------------------------
# STRATEGY COMPARISON
# -----------------------------

print("\nStrategy Performance by Driver")

for driver in drivers:

    print(f"\n{driver}")

    driver_summary = {}

    for strategy in strategy_results:

        positions = np.array(strategy_results[strategy][driver])

        driver_summary[strategy] = {

            "AvgFinish": positions.mean(),
            "WinProb": np.mean(positions == 1),
            "PodiumProb": np.mean(positions <= 3)

        }

    driver_df = pd.DataFrame(driver_summary).T
    print(driver_df)
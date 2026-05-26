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

track_overtake_factor = 0.8
drs_train_penalty = 0.4
failed_overtake_penalty = 0.1

fresh_tire_bonus = -0.45
fresh_tire_laps = 3

# Safety car probabilities
vsc_probability = 0.02
sc_probability = 0.01

vsc_lap_time = 115
sc_lap_time = 130

# -----------------------------
# SIMULATION PARAMETERS
# -----------------------------

n_simulations = 50000
lap_time_noise_sd = 0.6

traffic_penalty = (0.15, 0.35)
drs_range = 1.0
overtake_distance = 2.0

# -----------------------------
# DRIVER BASELINE PACE
# -----------------------------

data = {
    "Driver":[
        "VER","PER","NOR","PIA","LEC","SAI",
        "HAM","RUS","ALO","STR",
        "GAS","OCO","TSU","RIC",
        "ALB","SAR","MAG","HUL",
        "BOT","ZHO"
    ],

    "BaselineLapTime":[
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

    "Soft": {"pace_offset":-0.4,"deg_rate":0.05,"deg_curve":0.002},

    "Medium":{"pace_offset":0.0,"deg_rate":0.035,"deg_curve":0.0015},

    "Hard":{"pace_offset":0.3,"deg_rate":0.02,"deg_curve":0.0008}

}

# -----------------------------
# STRATEGY SET
# -----------------------------

strategies = [
    ("Medium-Hard",["Medium","Hard"]),
    ("Hard-Medium",["Hard","Medium"]),
    ("Soft-Medium-Medium",["Soft","Medium","Medium"]),
    ("Soft-Medium-Hard",["Soft","Medium","Hard"])
]

stint_targets = {

    "Soft":(15,4),
    "Medium":(22,5),
    "Hard":(30,6)

}

strategy_weights = [0.35,0.30,0.20,0.15]

# -----------------------------
# OVERTAKE MODEL
# -----------------------------

def overtake_probability(delta_pace, drs):

    alpha = -1.5
    beta = 3.0
    gamma = 1.5

    logit = alpha + beta*delta_pace + gamma*drs

    return 1/(1+np.exp(-logit))


# -----------------------------
# MONTE CARLO SIMULATION
# -----------------------------

results = np.zeros((n_simulations,n_drivers))
race_time_storage = []

for sim in range(n_simulations):

    total_times = np.zeros(n_drivers)

    fresh_tire_counter = np.zeros(n_drivers)

    driver_strategy = []

    stint_index = np.zeros(n_drivers,dtype=int)
    stint_lap_counter = np.zeros(n_drivers)

    compounds = []
    stint_laps_remaining = []

    # -----------------------------
    # ASSIGN STRATEGIES
    # -----------------------------

    for i in range(n_drivers):

        strategy_name,strategy = random.choices(
            strategies,
            weights=strategy_weights
        )[0]

        driver_strategy.append(strategy_name)

        compound = strategy[0]

        base,var = stint_targets[compound]

        stint_len = int(np.random.normal(base,var/2))
        stint_len = max(8,min(stint_len,35))

        compounds.append(compound)
        stint_laps_remaining.append(stint_len)

    stint_laps_remaining = np.array(stint_laps_remaining)

    # -----------------------------
    # LAP SIMULATION
    # -----------------------------

    for lap in range(laps):

        lap_times = np.zeros(n_drivers)

        safety_mode = None

        if np.random.rand() < sc_probability:
            safety_mode = "SC"

        elif np.random.rand() < vsc_probability:
            safety_mode = "VSC"

        # -----------------------------
        # LAP TIME CALCULATION
        # -----------------------------

        for i in range(n_drivers):

            tire = tire_compounds[compounds[i]]

            noise = np.random.normal(0,lap_time_noise_sd)

            lap_time = (
                baseline[i]
                + tire["pace_offset"]
                + tire["deg_rate"] * stint_lap_counter[i]
                + tire["deg_curve"] * (stint_lap_counter[i]**2)
                + noise
            )

            if fresh_tire_counter[i] > 0:
                lap_time += fresh_tire_bonus
                fresh_tire_counter[i] -= 1

            if safety_mode == "SC":
                lap_time = sc_lap_time

            elif safety_mode == "VSC":
                lap_time = vsc_lap_time

            lap_times[i] = lap_time

            stint_lap_counter[i] += 1
            stint_laps_remaining[i] -= 1


        # -----------------------------
        # TRAFFIC EFFECT
        # -----------------------------

        order = np.argsort(total_times)

        for pos in range(1,n_drivers):

            driver = order[pos]
            ahead = order[pos-1]

            gap = total_times[driver] - total_times[ahead]

            if gap < 1.5:

                penalty = np.random.uniform(*traffic_penalty)

                lap_times[driver] += penalty


        # -----------------------------
        # OVERTAKES
        # -----------------------------

        order = np.argsort(total_times)

        for pos in range(1,n_drivers):

            driver = order[pos]
            ahead = order[pos-1]

            gap = total_times[driver] - total_times[ahead]

            if gap > overtake_distance:
                continue

            drs = 1 if gap < drs_range else 0

            delta_pace = baseline[ahead] - baseline[driver]

            p = overtake_probability(delta_pace,drs)

            if pos >= 2:

                ahead_gap = total_times[ahead] - total_times[order[pos-2]]

                if gap < drs_range and ahead_gap < drs_range:

                    p *= (1 - drs_train_penalty)

            if np.random.rand() < track_overtake_factor:

                if np.random.rand() < p:

                    lap_times[driver] -= np.random.uniform(0.2,0.5)

                else:

                    lap_times[driver] += failed_overtake_penalty


        total_times += lap_times


        # -----------------------------
        # SAFETY CAR COMPRESSION
        # -----------------------------

        if safety_mode == "SC":

            leader_time = np.min(total_times)

            for i in range(n_drivers):

                gap = total_times[i] - leader_time

                if gap > 10:

                    total_times[i] = leader_time + 10


        # -----------------------------
        # PIT STOPS
        # -----------------------------

        for i in range(n_drivers):

            if stint_laps_remaining[i] <= 0:

                strategy_name = driver_strategy[i]

                strategy = next(
                    s[1] for s in strategies
                    if s[0] == strategy_name
                )

                if stint_index[i] < len(strategy)-1:

                    total_times[i] += pit_loss

                    fresh_tire_counter[i] = fresh_tire_laps

                    stint_index[i] += 1

                    compound = strategy[stint_index[i]]

                    compounds[i] = compound

                    base,var = stint_targets[compound]

                    stint_len = int(np.random.normal(base,var/2))
                    stint_len = max(8,min(stint_len,35))

                    stint_laps_remaining[i] = stint_len
                    stint_lap_counter[i] = 0


    # -----------------------------
    # FINAL POSITIONS
    # -----------------------------

    finishing_order = np.argsort(total_times)

    positions = np.empty(n_drivers)

    positions[finishing_order] = np.arange(1,n_drivers+1)

    results[sim] = positions

    race_time_storage.append(np.mean(total_times))


# -----------------------------
# RESULTS
# -----------------------------

results_df = pd.DataFrame(results,columns=drivers)

summary = pd.DataFrame({

    "WinProb":(results_df==1).mean(),
    "PodiumProb":(results_df<=3).mean(),
    "Top10Prob":(results_df<=10).mean(),
    "ExpectedPosition":results_df.mean()

}).sort_values("ExpectedPosition")

print("\nDriver Outcome Summary")

print(summary)

print("\nAverage Race Time")

print(np.mean(race_time_storage))

# -----------------------------
# FINISH POSITION DISTRIBUTION
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

# reorder drivers by expected position
position_distribution_df = position_distribution_df[summary.index]

# -----------------------------
# HEATMAP
# -----------------------------

plt.figure(figsize=(14,6))

sns.heatmap(
    position_distribution_df,
    cmap="viridis",
    linewidths=0.5,
    annot=True,          # show probabilities
    fmt=".2f",           # format to 2 decimals
    cbar_kws={"label":"Probability"}
)

plt.title(f"{race_name} Monte Carlo Finishing Position Distribution")
plt.xlabel("Driver")
plt.ylabel("Finishing Position")

plt.show()
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# PARAMETERS
# -----------------------------
n_simulations = 50000
n_laps = 57
lap_time_noise_sd = 0.6

# -----------------------------
# CREATE BASELINE DRIVER DATA
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
        95.20, 95.21, 95.22, 95.24, 95.27, 95.29, 
        95.33, 95.35, 95.40, 95.43,
        95.48, 95.50, 95.53, 95.55,
        95.58, 95.62, 95.65, 95.68,
        95.72, 95.75
    ]
}

df = pd.DataFrame(data)

# -----------------------------
# INPUT DATA
# -----------------------------
drivers = df["Driver"].values
baseline = df["BaselineLapTime"].values

n_drivers = len(drivers)

# -----------------------------
# VECTORISED MONTE CARLO
# -----------------------------

# generate lap noise
lap_noise = np.random.normal(
    0,
    lap_time_noise_sd,
    size=(n_simulations, n_laps, n_drivers)
)

# broadcast baseline pace
baseline_matrix = baseline.reshape(1,1,n_drivers)

lap_times = baseline_matrix + lap_noise

# total race time
race_times = lap_times.sum(axis=1)

# determine finishing order
positions = np.argsort(np.argsort(race_times, axis=1), axis=1) + 1

results_df = pd.DataFrame(positions, columns=drivers)

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
})

summary = summary.sort_values("ExpectedPosition")

print(summary)

# -----------------------------
# FINISHING POSITION DISTRIBUTION
# -----------------------------

position_distribution = {}

for driver in drivers:
    counts = (
        results_df[driver]
        .value_counts(normalize=True)
        .reindex(range(1, n_drivers + 1), fill_value=0)
    )
    position_distribution[driver] = counts

position_distribution_df = pd.DataFrame(position_distribution).fillna(0)

print("\nFinishing Position Distribution")
print(position_distribution_df)

# -----------------------------
# HEATMAP VISUALIZATION
# -----------------------------
position_distribution_df = position_distribution_df[summary.index]
plt.figure(figsize=(14,7))

sns.heatmap(
    position_distribution_df,
    cmap="viridis",
    linewidths=0.5,
    annot=True,
    fmt=".2f",
    cbar_kws={"label": "Probability"}
)

plt.title("Monte Carlo Finishing Position Probabilities")
plt.xlabel("Driver")
plt.ylabel("Finishing Position")

plt.show()
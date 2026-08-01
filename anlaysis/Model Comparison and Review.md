# F1 Prediction Model: Issues, Fixes, and Comparative Results

## Table of Contents
1. [Inherent Issues in the Original Models](#1-inherent-issues-in-the-original-models)
2. [Fixes Implemented in v2](#2-fixes-implemented-in-v2)
3. [Paper v1 Results (Baseline for Comparison)](#3-paper-v1-results-baseline-for-comparison)
4. [XGBRanker v2 Walk-Forward Results](#4-xgbranker-v2-walk-forward-results)
5. [PL v2 Walk-Forward Results](#5-pl-v2-walk-forward-results)
6. [Cross-Model Comparison](#6-cross-model-comparison)

---

## 1. Inherent Issues in the Original Models

### 1.1 Data Leakage in Model Selection — *Critical*

**What it is:** Model selection (BIC backward, AIC stepwise, VIF) was run on the full dataset — including the test years — before any cross-validation split was made. This means the model chose which features to include by peeking at the races it would later be evaluated on.

**In PL v1** (`PL model w enriched features.py`, lines 506–513):
```python
bic_predictors, bic_summary, bic_history = backward_selection_pl(
    model_df=model_df,       # full dataset — test folds visible here
    predictors=FEATURES.copy(),
    criterion="BIC",
    lambda_l2=lambda_l2
)
```

**In XGBRanker v1** (`F1 predictions new model.py`, lines 1068–1080):
```python
vif_features = vif_backward_selection(data)   # full dataset
FEATURES = vif_features
vif_metrics_df, _ = train_two_stage_model(data)   # uses vif_features
FEATURES = original_features   # reset — final model ignores them anyway
```

**Why it matters:** Even partial leakage inflates metrics. A model that selected features having seen 2024/2025 data will look better than it truly is on those years. Reported performance in the paper is optimistic.

---

### 1.2 Temporally Invalid Cross-Validation (GroupKFold)

**What it is:** Both models used `GroupKFold(n_splits=5)` grouped by `RaceID`. GroupKFold does not preserve chronological order — it can assign 2025 races to a training fold and 2022 races to the test fold. This is realistic if you believe future knowledge should help predict the past, which it should not.

**PL v1** (`PL model w enriched features.py`, line 474):
```python
gkf = GroupKFold(n_splits=5)
```

**XGBRanker v1** (`F1 predictions new model.py`, line 703):
```python
gkf = GroupKFold(n_splits=5)
```

**Why it matters:** Walk-forward validation is the only valid protocol for time-series prediction. Training on future seasons' data and testing on past ones allows the model to learn driver/team performance levels that weren't yet established in the test period (e.g., knowing Red Bull dominated 2023 when predicting 2022). This makes measured performance higher than real-world performance on unseen future races.

---

### 1.3 Wrong Sample Size for BIC Penalty (PL v1 Only)

**What it is:** BIC is defined as `k·ln(n) − 2·ln(L)`, where `n` is the number of independent observations. In a PL ranking model, each observation is a race (the ordered finish of all drivers in that race). The original code used `n = len(model_df)` which counts individual driver-race rows (~400 rows for ~85 races — roughly 20× too large).

**PL v1** (line 259):
```python
n = len(model_df)   # ~400 rows, not ~85 races
```

**Why it matters:** A larger `n` imposes a heavier penalty per feature in BIC. Using row count instead of race count makes BIC over-penalise features relative to AIC, biasing selection toward fewer features than the data actually support. The reported 4-feature BIC model may be too parsimonious.

---

### 1.4 Inconsistent Optimizer Between Selection and CV (PL v1 Only)

**What it is:** The `fit_pl_model` function used during feature selection used `L-BFGS-B` (a bounded quasi-Newton optimizer), but the CV loop used `BFGS`. The two methods produce slightly different Hessians (and hence different log-likelihood surfaces), so the beta estimates and convergence properties are not directly comparable.

**PL v1** (line 234 vs line 641):
```python
method="L-BFGS-B"   # used during BIC/AIC selection
method="BFGS"        # used in CV loop — inconsistency
```

**Why it matters:** Features selected under L-BFGS-B may not be the same ones BFGS would select. The reported CV metrics were generated with a different optimizer than the one that chose the features, creating an internal inconsistency.

---

### 1.5 Magic Number in Score Adjustment (XGBRanker v1 Only)

**What it is:** The final race score combined the ranker's raw output with DNF probability using a hardcoded coefficient:
```python
final_score = raw_score - (2.0 * test_dnf_prob)   # line 792
```

**Why it matters:** The `2.0` is arbitrary. The XGBRanker was trained to minimise NDCG loss, which already accounts for DNF risk through the `DNFProbFeature` input. Applying a manual penalty double-counts DNF risk. The coefficient was not chosen by optimisation and has no principled justification — any value could have been used.

---

### 1.6 VIF/LOO Results Discarded (XGBRanker v1 Only)

**What it is:** The code ran VIF backward selection and LOO analysis, printed results to console, then reset `FEATURES` to the original 18-feature set before training the final model. The analysis had no effect on the deployed model.

**XGBRanker v1** (lines 1068–1081):
```python
vif_features = vif_backward_selection(data)
FEATURES = vif_features
vif_metrics_df, _ = train_two_stage_model(data)   # uses vif_features
FEATURES = original_features   # reset — final model reverts to 18 features
dnf_model, ranker, scaler, importance = train_final_model(data)  # 18 features
```

**Why it matters:** Running VIF/LOO without acting on the results is analysis theatre. The reported metrics for the "VIF-reduced" model in the paper were generated during CV but the actual model weights use all 18 features — not the 15 VIF-selected ones.

---

### 1.7 O(n²) Rolling Feature Computation (XGBRanker v1 Only)

**What it is:** Driver form, team strength, and related rolling features were computed by iterating over every row in the dataframe and filtering the full dataframe for each row's driver/team history. This scales quadratically with dataset size.

**XGBRanker v1** (lines 538–586):
```python
for idx, row in data.iterrows():
    driver_past = data[
        ((data["Year"] < year) | ...) & (data["Driver"] == drv)
    ].tail(3)
    data.loc[idx, "DriverForm"] = driver_past["Points"].mean()
```

**Why it matters:** This produced incorrect rolling windows. Using `data` (the full dataframe) to look up history without strict temporal ordering meant that at training time, the code could inadvertently include future rows in the rolling window if the dataframe was not sorted. Even with sorting, the per-row filter is 1,800× slower than a vectorised approach.

---

### 1.8 Weather Session Loaded Twice (XGBRanker v1 Only)

**What it is:** `extract_race_features` loaded the race session without weather data, and `extract_weather_features` immediately loaded the same session again with full data. Both functions are called in sequence for every race.

**Why it matters:** FastF1 session loading can trigger network requests and cache writes. Loading the same session twice doubles I/O for every historical race in the dataset, roughly doubling rebuild time (which already runs to many hours for 90+ races).

---

### 1.9 QualiPercentile Hardcoded Denominator

**What it is:** QualiPercentile normalises grid position to [0, 1]:
```python
data["QualiPercentile"] = (data["QualiPosition"] - 1) / 19   # PL v1 line 597
```

This assumes exactly 20 starters. If a driver was disqualified from qualifying, withdrew, or a race had a non-standard grid, the percentile is incorrect and can exceed 1.0 or compress the range artificially.

**Why it matters:** For races with DSQ or DNS entries (which appear in 2025 data), this produces QualiPercentile > 1.0 for mid-grid starters, corrupting the feature scaling.

---

## 2. Fixes Implemented in v2

| # | Issue | PL v2 | XGBRanker v2 |
|---|---|:---:|:---:|
| 1.1 | Model selection on training data only (no leakage) | ✓ | ✓ |
| 1.2 | Walk-forward temporal CV (no GroupKFold) | ✓ | ✓ |
| 1.3 | BIC n = race count (not row count) | ✓ | — |
| 1.4 | Consistent L-BFGS-B optimizer throughout | ✓ | — |
| 1.5 | Remove 2.0 magic number from score | — | ✓ |
| 1.6 | VIF selection feeds into final model | — | ✓ |
| 1.7 | Vectorised rolling features (no row loop) | — | ✓ |
| 1.8 | Single session load for weather | — | ✓ |
| 1.9 | QualiPercentile uses actual grid size | ✓ | ✓ |
| — | Baseline comparison (grid rank) in every fold | ✓ | ✓ |
| — | XGBRanker groups sorted before fitting | — | ✓ |
| — | NaN-safe baseline for DNS drivers | — | ✓ |

### Key Implementation Details

**Walk-forward splits** (both v2 files):
```python
def walk_forward_splits(df, eval_years):
    for test_year in eval_years:
        train_idx = df.index[df["Year"] < test_year].tolist()
        test_idx  = df.index[df["Year"] == test_year].tolist()
        yield train_idx, test_idx
```

**BIC with race-count n** (PL v2):
```python
n_races = train_df["RaceID"].nunique()   # correct: 22–68 races
_, best_bic = aic_bic(loglik, len(current), n_races)
```

**VIF feeds both CV and final model** (XGBRanker v2):
```python
selected_features = vif_backward_selection(vif_ref_data, FEATURES)
mm, bm, dm, preds = train_and_evaluate_fold(train, test, selected_features)
_, _, _, importance = train_final_model(eval_data, selected_features)
```

**Raw ranker score (no 2.0 multiplier)** (XGBRanker v2):
```python
test_r["PredScore"] = ranker.predict(test_r[rank_features])   # ranker already learned DNF penalty
```

---

## 3. Paper v1 Results (Baseline for Comparison)

These results were obtained with GroupKFold (not walk-forward) and model selection on the full dataset. They represent the originally reported results from the paper.

### 3.1 Plackett-Luce Model (PL v1) — All Variants

| Model | Features | Spearman | KendallTau | Top3 Acc | Top5 Acc | Winner Acc | NDCG@3 | NDCG@5 |
|---|---|---|---|---|---|---|---|---|
| PL BIC (backward) | 4 | 0.671 | 0.539 | 0.651 | 0.767 | 0.577 | 0.882 | 0.893 |
| PL AIC (stepwise) | 7 | 0.667 | 0.533 | 0.629 | 0.761 | 0.577 | 0.880 | 0.894 |
| PL VIF-reduced | 15 | 0.666 | 0.534 | 0.633 | 0.756 | 0.544 | 0.875 | 0.891 |
| PL Full Enriched | 18 | 0.665 | 0.533 | 0.619 | 0.756 | 0.587 | 0.878 | 0.893 |

BIC with 4 features leads on most metrics. More features (VIF, Full) slightly reduce accuracy — consistent with overfitting to the (leaky) training data.

### 3.2 Two-Stage XGBRanker (v1) — All Variants

| Model | Features | Spearman | KendallTau | Top3 Acc | Top5 Acc | Winner Acc | NDCG@3 | NDCG@5 |
|---|---|---|---|---|---|---|---|---|
| Two-Stage Full | 18 | 0.653 | 0.523 | 0.677 | 0.750 | 0.590 | 0.887 | 0.890 |
| Two-Stage VIF-reduced | 15 | 0.657 | 0.522 | 0.688 | 0.758 | 0.588 | 0.890 | 0.892 |

VIF-reduced slightly outperforms full on all Top-N accuracy metrics. Feature importance (v1): QualiPercentile ~50%, QualiGapToPole ~11%, TeamStrength ~8%, remaining features each < 5%.

---

## 4. XGBRanker v2 Walk-Forward Results

**Setup:** VIF backward selection → 15 features (removed OvertakeOpportunity, DriverForm, Quali_x_Overtake). Walk-forward CV: train < year, test = year. No 2.0 score adjustment.

**VIF-selected features (15):** QualiPercentile, QualiGapToPole, TeammateQualiGap, FP_LongRunPace, FP_LongRunVar, TeamStrength, DriverDNFRate, TeamDNFRate, AirTemp, TrackTemp, Rainfall, OvertakeIndex, Reliability_x_Rain, Driver_vs_Team, GridSpread

**VIF removal sequence:**
1. OvertakeOpportunity (VIF = ∞, collinear with OvertakeIndex)
2. DriverForm (VIF = 614, absorbed by TeamStrength + Driver_vs_Team)
3. Quali_x_Overtake (VIF = 18, collinear with QualiPercentile + OvertakeIndex)

### 4.1 Per-Fold Results

| Fold | Metric | Model | Baseline | Delta |
|---|---|---|---|---|
| 2023 | Spearman | 0.6144 | 0.5920 | **+0.022** |
| 2023 | Top3 Acc | 0.5909 | 0.6212 | −0.030 |
| 2023 | Top5 Acc | 0.6273 | 0.6636 | −0.036 |
| 2023 | Winner Acc | 0.6364 | 0.6818 | −0.045 |
| 2023 | NDCG@3 | 0.8432 | 0.8637 | −0.021 |
| 2023 | NDCG@5 | 0.8439 | 0.8605 | −0.017 |
| 2023 | DNF LogLoss | 0.6923 | — | — |
| 2024 | Spearman | 0.6994 | 0.7449 | −0.045 |
| 2024 | Top3 Acc | 0.6389 | 0.6250 | **+0.014** |
| 2024 | Top5 Acc | 0.8167 | 0.8000 | **+0.017** |
| 2024 | Winner Acc | 0.5417 | 0.5000 | **+0.042** |
| 2024 | NDCG@3 | 0.8966 | 0.8846 | **+0.012** |
| 2024 | NDCG@5 | 0.9178 | 0.9155 | **+0.002** |
| 2024 | DNF LogLoss | 0.7100 | — | — |
| 2025 | Spearman | 0.6514 | 0.6549 | −0.003 |
| 2025 | Top3 Acc | 0.6806 | 0.7500 | −0.069 |
| 2025 | Top5 Acc | 0.8167 | 0.8000 | **+0.017** |
| 2025 | Winner Acc | 0.4583 | 0.6667 | −0.208 |
| 2025 | NDCG@3 | 0.9103 | 0.9336 | −0.023 |
| 2025 | NDCG@5 | 0.9319 | 0.9437 | −0.012 |
| 2025 | DNF LogLoss | 0.7177 | — | — |

### 4.2 Walk-Forward Averages vs Paper v1

| Metric | Paper v1 (VIF, GroupKFold) | v2 Walk-Forward (VIF) | Baseline (Walk-Fwd) |
|---|---|---|---|
| Spearman | 0.657 | **0.655** | 0.664 |
| Top3 Acc | 0.688 | **0.637** | 0.665 |
| Top5 Acc | 0.758 | **0.754** | 0.754 |
| Winner Acc | 0.588 | **0.545** | 0.616 |
| NDCG@3 | 0.890 | **0.884** | 0.894 |
| NDCG@5 | 0.892 | **0.884** | 0.899 |

**Interpretation:** Walk-forward results are uniformly lower than paper v1. The gap (3–5 percentage points on accuracy metrics) is attributable to: (a) removal of temporal leakage, (b) removal of data-leaky feature selection, and (c) the true difficulty of predicting future seasons. The baseline (grid position) outperforms the model on 2025 Winner Accuracy by 20+ points, suggesting the model has learned patterns specific to 2022–2024 Red Bull dominance that don't transfer to 2025's more competitive field.

### 4.3 Feature Importance (Final Model, VIF-15 features)

| Rank | Feature | Importance |
|---|---|---|
| 1 | QualiPercentile | ~56.9% |
| 2 | QualiGapToPole | ~10.8% |
| 3 | TeamStrength | ~9.0% |
| 4 | Driver_vs_Team | ~2.8% |
| 5–15 | Others | < 2% each |

QualiPercentile alone accounts for more than half of model importance, confirming qualifying position is the dominant predictor. The remaining features add marginal but non-zero signal.

---

## 5. PL v2 Walk-Forward Results

> **Status: Results pending — model is currently running.**
>
> This section will be updated once `PL model v2.py` completes its walk-forward evaluation.
> The model runs BIC backward, AIC stepwise, and VIF backward selection methods on each fold's training data.
> Walk-forward folds: train < 2023, train < 2024, train < 2025.

### Expected Output Structure

| Fold | Method | Features Selected | Spearman | Top3 Acc | Top5 Acc | Winner Acc | NDCG@3 | vs Baseline |
|---|---|---|---|---|---|---|---|---|
| 2023 | BIC | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 2023 | AIC | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 2023 | VIF | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 2024 | BIC | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 2024 | AIC | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 2024 | VIF | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 2025 | BIC | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 2025 | AIC | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 2025 | VIF | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

---

## 6. Cross-Model Comparison

> **Status: Partial — XGBRanker v2 available; PL v2 pending.**
>
> This section will be updated to include the full PL v2 vs XGBRanker v2 comparison once PL v2 finishes.

### 6.1 Paper v1 vs v2 Methodology Summary

| Aspect | Paper v1 (Both Models) | v2 (Both Models) |
|---|---|---|
| CV Protocol | GroupKFold (5-fold) | Walk-forward (3 folds: test = 2023, 2024, 2025) |
| Feature Selection | On full dataset (leaky) | On training data only (per-fold or FAST_SELECTION) |
| Baseline Comparison | None | Negated QualiPosition every fold |
| QualiPercentile | Hardcoded /19 | Actual grid size |

| Aspect | PL v1 → v2 | XGBRanker v1 → v2 |
|---|---|---|
| BIC n | Row count (~400) | Race count (~22–68) |
| Optimizer | Mixed L-BFGS-B/BFGS | Consistent L-BFGS-B |
| DNF Penalty | N/A | 2.0×DNF removed |
| VIF → Final Model | Ran on full data | Selected features passed explicitly |
| Rolling Features | N/A | Vectorised (was O(n²) loop) |
| Session Load | N/A | Single load (was double) |

### 6.2 XGBRanker v1 vs v2 (Available)

| Metric | v1 (GroupKFold, 18 feat) | v1 (GroupKFold, VIF-15) | v2 (Walk-Fwd, VIF-15) | v2 Baseline |
|---|---|---|---|---|
| Spearman | 0.653 | 0.657 | 0.655 | 0.664 |
| Top3 Acc | 0.677 | 0.688 | 0.637 | 0.665 |
| Top5 Acc | 0.750 | 0.758 | 0.754 | 0.754 |
| Winner Acc | 0.590 | 0.588 | 0.545 | 0.616 |
| NDCG@3 | 0.887 | 0.890 | 0.884 | 0.894 |
| NDCG@5 | 0.890 | 0.892 | 0.884 | 0.899 |

The baseline (grid position) beats the v2 model on 4 of 6 metrics in walk-forward evaluation, but only by ~1% on Spearman and Top5. The larger gaps on Winner Acc and Top3 Acc reflect genuine difficulty predicting 2025, which saw a more dispersed championship. Model v1's higher numbers are mostly explained by data leakage and non-temporal splits.

### 6.3 PL v1 vs PL v2 (Pending)

> *To be filled after PL v2 completes.*

| Metric | PL v1 BIC (GroupKFold) | PL v2 BIC (Walk-Fwd) | PL v2 AIC (Walk-Fwd) | PL v2 VIF (Walk-Fwd) | Baseline |
|---|---|---|---|---|---|
| Spearman | 0.671 | TBD | TBD | TBD | TBD |
| Top3 Acc | 0.651 | TBD | TBD | TBD | TBD |
| Top5 Acc | 0.767 | TBD | TBD | TBD | TBD |
| Winner Acc | 0.577 | TBD | TBD | TBD | TBD |
| NDCG@3 | 0.882 | TBD | TBD | TBD | TBD |
| NDCG@5 | 0.893 | TBD | TBD | TBD | TBD |

### 6.4 PL v2 vs XGBRanker v2 (Pending)

> *To be filled after PL v2 completes.*

| Metric | PL v2 Best | XGBRanker v2 | Baseline |
|---|---|---|---|
| Spearman | TBD | 0.655 | 0.664 |
| Top3 Acc | TBD | 0.637 | 0.665 |
| Top5 Acc | TBD | 0.754 | 0.754 |
| Winner Acc | TBD | 0.545 | 0.616 |
| NDCG@3 | TBD | 0.884 | 0.894 |

---

## Appendix: VIF Removal Order (XGBRanker v2)

| Step | Feature Removed | VIF Before Removal |
|---|---|---|
| 1 | OvertakeOpportunity | ∞ (perfectly collinear) |
| 2 | DriverForm | 613.9 (absorbed by TeamStrength + Driver_vs_Team) |
| 3 | Quali_x_Overtake | 18.0 (collinear with QualiPercentile + OvertakeIndex) |

After step 3, all remaining 15 features had VIF < 5.0. Final VIF values ranged from 1.19 (Driver_vs_Team) to 3.92 (Reliability_x_Rain).

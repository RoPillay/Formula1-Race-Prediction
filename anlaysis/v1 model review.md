# F1 Prediction Model Review

**Models reviewed:**
- `PL model w enriched features.py` — Plackett-Luce ranking model (v1)
- `F1 predictions new model.py` — Two-stage XGBRanker model (v1)

**Fixes implemented in:**
- `PL model v2.py`
- `F1 predictions v2.py`

---

## Model 1: Plackett-Luce (`PL model w enriched features.py`)

### Strengths

**Theoretically grounded likelihood.**
PL is the correct statistical model for ordered outcomes. The log-likelihood correctly accounts for the sequential selection process of a race finish.

**Principled feature selection.**
AIC/BIC backward and stepwise selection is appropriate for this model class. The BIC-selected 4-feature model is lean and interpretable:
```python
# lines 487-492
bic_predictors = [
    "QualiPercentile",
    "TeamStrength",
    "Quali_x_Overtake",
    "Driver_vs_Team"
]
```

**Interpretable coefficients with uncertainty.**
The model extracts approximate standard errors and p-values from the quasi-Newton Hessian, giving coefficient-level insight that the XGBRanker cannot provide.

**Correct feature scaling within folds.**
Standard scaler is fit on training data only and applied to test data — no leakage from scaling.

---

### Issues

#### CRITICAL — Data leakage in model selection
Model selection (`backward_selection_pl`) runs on the full dataset before CV. The test folds are visible during feature selection, inflating reported metrics.

```python
# lines 506-513  <-- BUG
bic_predictors, bic_summary, bic_history = backward_selection_pl(
    model_df=model_df,       # <-- full dataset including test folds
    predictors=FEATURES.copy(),
    criterion="BIC",
    lambda_l2=lambda_l2
)
```

**Fix (v2):** `bic_backward_selection` is called inside each walk-forward fold on training data only:
```python
# PL model v2.py
selected, best_bic = bic_backward_selection(train, ALL_FEATURES.copy())
```

---

#### BIC `n` is the wrong sample size
`n = len(model_df)` counts driver-race rows (~400+), not races (~85). This overstates the BIC penalty, biasing selection toward fewer features than warranted.

```python
# line 259  <-- BUG
n = len(model_df)   # ~400 rows, not ~85 races
```

**Fix (v2):** Use race count:
```python
n_races = train_df["RaceID"].nunique()   # correct n for BIC
_, best_bic = aic_bic(loglik, len(current), n_races)
```

---

#### Inconsistent optimizer between model selection and CV
`fit_pl_model` (used in selection) uses `L-BFGS-B`, but the CV loop uses `BFGS`. This means the Hessian structure differs between the two contexts and the beta estimates are not directly comparable.

```python
# line 234 — selection uses L-BFGS-B
method="L-BFGS-B"

# line 641 — CV loop uses BFGS  <-- inconsistency
method="BFGS"
```

**Fix (v2):** `L-BFGS-B` used everywhere in `fit_pl`.

---

#### GroupKFold is not temporally ordered
GroupKFold can assign 2025 races to a training fold and 2022 races to a test fold. For time-series data this is unrealistic — it allows the model to train on future knowledge.

```python
# line 474
gkf = GroupKFold(n_splits=5)
```

**Fix (v2):** Walk-forward splits — train on all years before the test year:
```python
def walk_forward_splits(df, eval_years):
    for test_year in eval_years:
        train_idx = df.index[df["Year"] < test_year].tolist()
        test_idx  = df.index[df["Year"] == test_year].tolist()
```

---

#### No baseline comparison
There is no comparison against simply predicting grid position = finish position. Without a baseline it is impossible to tell whether the enriched features add value over qualifying alone.

**Fix (v2):** Every fold prints a side-by-side table of PL model vs. baseline (`-QualiPosition`).

---

#### Dead code and confusing redundancy
Large sections are triple-quoted out (lines 55-79, 547-696). The hardcoded `bic_predictors` list (lines 487-492) is immediately overwritten by a fresh selection run, making it unclear which features actually drive the CV.

**Fix (v2):** All dead code removed. Feature set is determined entirely by the in-fold BIC selection.

---

#### QualiPercentile assumes exactly 20 drivers
```python
# finalize_features in F1 predictions new model.py, line 597
data["QualiPercentile"] = (data["QualiPosition"] - 1) / 19
```

If a race has 18 or 19 starters (disqualification, withdrawal), the percentile is wrong.

**Fix (v2):**
```python
n_starters = df.groupby("RaceID")["Driver"].transform("count")
df["QualiPercentile"] = (df["QualiPosition"] - 1) / (n_starters - 1).clip(lower=1)
```

---

## Model 2: Two-Stage XGBRanker (`F1 predictions new model.py`)

### Strengths

**Two-stage design separates DNF risk from pace.**
Modelling DNF probability separately from race pace is conceptually correct. A driver can be fast but unreliable — conflating these in a single model loses signal.

**NDCG@3 objective.**
`XGBRanker` with `objective="rank:ndcg"` directly optimises for podium prediction accuracy, aligning the training objective with the evaluation metric.

**Live FastF1 ingestion pipeline.**
The feature engineering pipeline pulls from real session data and handles edge cases (missing sessions, weather, team name normalisation), making it usable for real race-week prediction.

**LOO feature analysis.**
Leave-one-out analysis correctly identifies which features drive predictive accuracy in the nonlinear, correlated feature space where VIF alone is insufficient.

**VIF backward selection.**
Variance inflation factor analysis identifies redundant features. In v1 this is diagnostic only; in v2 it feeds directly into the model.

---

### Issues

#### Magic number in final score adjustment
The `2.0` coefficient is arbitrary and double-penalises DNF-prone drivers: the ranker already learns to penalise them via `DNFProbFeature`, and then the manual adjustment does it again.

```python
# line 792  <-- BUG
final_score = raw_score - (2.0 * test_dnf_prob)
```

**Fix (v2):** Raw ranker score used directly — the ranker learns the appropriate penalty:
```python
test_r["PredScore"] = ranker.predict(test_r[rank_features])
```

---

#### VIF and LOO results do not feed back into the final model
The analysis runs, prints useful results, then `train_final_model` at line 1049 ignores them and uses all 18 FEATURES anyway.

```python
# lines 1068-1080  <-- VIF result discarded
vif_features = vif_backward_selection(data)

FEATURES = vif_features
vif_metrics_df, _ = train_two_stage_model(data)   # uses vif_features

FEATURES = original_features   # reset — final model ignores vif_features

# line 1049 — final model trained on all 18 features, not VIF-selected
dnf_model, ranker, scaler, importance = train_final_model(data)
```

**Fix (v2):** VIF selection result is passed explicitly to both CV and final model:
```python
selected_features = vif_backward_selection(vif_ref_data, FEATURES)
mm, bm, dm, preds = train_and_evaluate_fold(train, test, selected_features)
_, _, _, importance = train_final_model(eval_data, selected_features)
```

---

#### Rolling features computed with O(n²) row-by-row loop
The `add_rolling_features` function iterates over every row to compute `DriverForm`, `TeamStrength`, etc. This scales quadratically with dataset size.

```python
# lines 538-586  <-- O(n²) loop
for idx, row in data.iterrows():
    driver_past = data[
        ((data["Year"] < year) | ...) & (data["Driver"] == drv)
    ].tail(3)
    data.loc[idx, "DriverForm"] = driver_past["Points"].mean()
```

**Fix (v2):** Vectorised with `groupby + shift + rolling`:
```python
data["DriverForm"] = (
    data.groupby("Driver")["Points"]
    .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
)
# Team rolling uses race-level aggregation first to avoid within-race leakage
team_race["TeamStrength"] = (
    team_race.groupby("Team")["Points"]
    .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
)
```

---

#### Weather session loaded twice
`extract_race_features` loads the session without weather; `extract_weather_features` then loads the same session again with full data, re-downloading laps, telemetry, and messages unnecessarily.

```python
# extract_race_features, line 352
race.load(laps=False, telemetry=False, weather=False, messages=False)

# extract_weather_features, line 395 — same session loaded again  <-- BUG
race.load()
```

**Fix (v2):** Single combined function loads the session once with `weather=True`:
```python
def extract_race_and_weather(year, rnd):
    session = fastf1.get_session(year, rnd, "R")
    session.load(laps=False, telemetry=False, weather=True, messages=False)
    # extract results and weather from the same session object
```

---

#### GroupKFold is not temporally ordered
Same issue as the PL model — folds can train on future seasons.

```python
# line 703
gkf = GroupKFold(n_splits=5)
```

**Fix (v2):** Walk-forward splits (same approach as PL v2).

---

#### XGBRanker groups not sorted before fitting
`group_sizes` is computed from `groupby("RaceID").size()` but the data is not explicitly sorted by `RaceID` before this. If the dataframe is in a different order, the group sizes will not align with the actual data rows.

```python
# line 765  <-- potential misalignment
train_group_sizes = train_rank.groupby("RaceID").size().values
ranker.fit(X_train_rank, y_train_rank, group=train_group_sizes)
```

**Fix (v2):**
```python
train_r = train_r.sort_values("RaceID").copy()   # sort first
group_sizes = train_r.groupby("RaceID").size().values
ranker.fit(train_r[rank_features], y_train, group=group_sizes)
```

---

#### No baseline comparison
Same gap as PL v1 — no comparison against grid position.

**Fix (v2):** Every fold prints model vs. baseline table; `BaselineScore = -QualiPosition` is evaluated alongside `PredScore`.

---

## Shared Recommendations (applied in both v2 files)

| Recommendation | PL v2 | XGBRanker v2 |
|---|:---:|:---:|
| Walk-forward temporal CV | ✓ | ✓ |
| Baseline comparison (grid rank) | ✓ | ✓ |
| QualiPercentile uses actual grid size | ✓ | ✓ |
| Model selection on training data only | ✓ | — |
| BIC n = race count | ✓ | — |
| Consistent optimizer | ✓ | — |
| Remove 2.0 magic number | — | ✓ |
| VIF feeds into final model | — | ✓ |
| Vectorised rolling features | — | ✓ |
| Single session load (weather) | — | ✓ |
| XGBRanker groups sorted | — | ✓ |

---

## Outstanding Considerations (not yet implemented)

- **Standard errors in PL:** The Hessian from L-BFGS-B is a quasi-Newton approximation, not the true Fisher information. P-values should be treated as indicative. A bootstrap or profile likelihood approach would give valid inference.
- **`Reliability_x_Rain`:** This interaction is near-zero for most dry races. Consider whether it adds signal or just noise — check its BIC selection frequency across folds.
- **2025 calendar:** `get_rounds_for_year(2025)` is hardcoded to 24. Check against the actual published calendar.

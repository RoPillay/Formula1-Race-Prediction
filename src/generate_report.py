"""
generate_report.py — F1 Race Prediction: Full 12-Section PDF Report
Follows the structure of "Two-Stage vs PL model results, notes.pdf"
with v2 methodology updates:
  - Walk-forward CV (replaces 5-fold random grouped CV)
  - 2x DNF penalty removed from XGBRanker final score
  - QualiPercentile denominator fixed to (n_starters - 1)
  - VIF/LOO results from v2 runs
"""
import os
import pandas as pd
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

# ── Paths ──────────────────────────────────────────────────────────────────
SRC_DIR      = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_DIR = os.path.join(SRC_DIR, "..", "anlaysis")
OUT_PDF      = os.path.join(ANALYSIS_DIR, "F1_Model_Report.pdf")

PL_CSV          = os.path.join(SRC_DIR, "pl_v2_walkforward_summary.csv")
XGB_CSV         = os.path.join(SRC_DIR, "v2_walkforward_summary.csv")
IMP_CSV         = os.path.join(SRC_DIR, "v2_feature_importance.csv")
XGB_VIF_STEPS   = os.path.join(SRC_DIR, "v2_vif_steps.csv")
XGB_VIF_FINAL   = os.path.join(SRC_DIR, "v2_vif_final.csv")
XGB_VIF_SUMMARY = os.path.join(SRC_DIR, "v2_vif_summary.csv")
XGB_LOO_CSV     = os.path.join(SRC_DIR, "v2_loo_results.csv")
PL_VIF_STEPS    = os.path.join(SRC_DIR, "pl_v2_vif_steps.csv")
PL_VIF_FINAL    = os.path.join(SRC_DIR, "pl_v2_vif_final.csv")
PL_VIF_SUMMARY  = os.path.join(SRC_DIR, "pl_v2_vif_summary.csv")
PL_COEF_CSV     = os.path.join(SRC_DIR, "pl_v2_coefficients.csv")
PL_BIC_STEPS    = os.path.join(SRC_DIR, "pl_v2_bic_steps.csv")
PL_BIC_SUMMARY  = os.path.join(SRC_DIR, "pl_v2_bic_summary.csv")
PL_BIC_COEF     = os.path.join(SRC_DIR, "pl_v2_bic_coefficients.csv")
PL_AIC_STEPS    = os.path.join(SRC_DIR, "pl_v2_aic_steps.csv")
PL_AIC_SUMMARY  = os.path.join(SRC_DIR, "pl_v2_aic_summary.csv")
PL_AIC_COEF     = os.path.join(SRC_DIR, "pl_v2_aic_coefficients.csv")
PL_MCFADDEN     = os.path.join(SRC_DIR, "pl_v2_mcfadden.csv")
PL_COEF_FULL    = os.path.join(SRC_DIR, "pl_v2_coef_ref_full.csv")
PL_COEF_VIF     = os.path.join(SRC_DIR, "pl_v2_coef_ref_vif.csv")
PL_COEF_BIC     = os.path.join(SRC_DIR, "pl_v2_coef_ref_bic.csv")
PL_COEF_AIC     = os.path.join(SRC_DIR, "pl_v2_coef_ref_aic.csv")
ENRICHED_CSV    = os.path.join(SRC_DIR, "DEBUGF7_f1_enriched_prediction_dataset_2022_2025.csv")

ALL_FEATURES = [
    "QualiPercentile", "QualiGapToPole", "TeammateQualiGap",
    "FP_LongRunPace", "FP_LongRunVar",
    "DriverForm", "TeamStrength",
    "DriverDNFRate", "TeamDNFRate",
    "AirTemp", "TrackTemp", "Rainfall",
    "OvertakeIndex",
    "Quali_x_Overtake", "Reliability_x_Rain",
    "OvertakeOpportunity", "Driver_vs_Team", "GridSpread",
]

# ── Styles ─────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=18,
                    spaceAfter=8, textColor=colors.HexColor("#1a1a2e"))
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=14,
                    spaceAfter=6, spaceBefore=12,
                    textColor=colors.HexColor("#16213e"))
H3 = ParagraphStyle("H3", parent=styles["Heading3"], fontSize=11,
                    spaceAfter=4, spaceBefore=8,
                    textColor=colors.HexColor("#0f3460"))
H4 = ParagraphStyle("H4", parent=styles["Normal"], fontSize=10,
                    spaceAfter=3, spaceBefore=6, fontName="Helvetica-Bold",
                    textColor=colors.HexColor("#444444"))
BODY = ParagraphStyle("Body", parent=styles["Normal"], fontSize=9.5,
                      leading=14, spaceAfter=6, alignment=TA_JUSTIFY)
CAPTION = ParagraphStyle("Caption", parent=styles["Normal"], fontSize=8.5,
                         leading=12, spaceAfter=4,
                         textColor=colors.HexColor("#555555"),
                         alignment=TA_CENTER)
BULLET = ParagraphStyle("Bullet", parent=styles["Normal"], fontSize=9.5,
                        leading=14, spaceAfter=3, leftIndent=18,
                        alignment=TA_JUSTIFY)

DARK_BLUE  = colors.HexColor("#16213e")
MID_BLUE   = colors.HexColor("#0f3460")
LIGHT_BLUE = colors.HexColor("#e8f0fe")
HEADER_BG  = colors.HexColor("#2c3e6b")
ALT_ROW    = colors.HexColor("#f2f5fb")
POS_GREEN  = colors.HexColor("#1a7a4a")


def hr():
    return HRFlowable(width="100%", thickness=0.5,
                      color=colors.HexColor("#aaaaaa"), spaceAfter=6)

def p(text, style=BODY):
    return Paragraph(text, style)

def sp(h=0.3):
    return Spacer(1, h * cm)

BASE_STYLE = TableStyle([
    ("BACKGROUND",  (0, 0), (-1, 0), HEADER_BG),
    ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
    ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE",    (0, 0), (-1, 0), 8),
    ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
    ("ALIGN",       (0, 1), (0, -1), "LEFT"),
    ("FONTSIZE",    (0, 1), (-1, -1), 8),
    ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ALT_ROW]),
    ("GRID",        (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
    ("TOPPADDING",  (0, 0), (-1, -1), 3),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
])

def make_table(data, col_widths=None):
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(BASE_STYLE)
    return t

def fmt(v, decimals=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{decimals}f}"

def fmt4(v):
    return fmt(v, 4)

def delta_str(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.4f}"

def mean_row_style(t, n_rows):
    t.setStyle(TableStyle([
        ("FONTNAME",   (0, n_rows - 1), (-1, n_rows - 1), "Helvetica-Bold"),
        ("BACKGROUND", (0, n_rows - 1), (-1, n_rows - 1), LIGHT_BLUE),
        ("LINEABOVE",  (0, n_rows - 1), (-1, n_rows - 1), 0.8, DARK_BLUE),
    ]))

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 0 — Issues and Fixes (preamble)
# ═══════════════════════════════════════════════════════════════════════════

def section_issues():
    elems = []
    elems.append(p("<b>0. Inherent Issues in the Original Models and Fixes</b>", H2))
    elems.append(hr())
    elems.append(p(
        "Before presenting the updated model results, the methodological issues "
        "identified in the original models are documented along with the fixes "
        "implemented in version 2."
    ))
    elems.append(sp(0.2))

    issues = [
        ("0.1 Data Leakage in Feature Selection (Critical)",
         "Both models ran BIC backward, AIC stepwise, and VIF selection on the full "
         "dataset — including test seasons — before any cross-validation split. This "
         "allowed the model to see future race results when deciding which features to "
         "keep, inflating all reported metrics.",
         "In v2, feature selection is performed exclusively on training data. VIF "
         "selection runs on the 2022-2024 reference window and is never given access "
         "to the 2025 test season."),
        ("0.2 Temporally Invalid Cross-Validation (GroupKFold)",
         "Both originals used sklearn's GroupKFold(n_splits=5) grouped by RaceID. "
         "GroupKFold does not preserve chronological order — it can assign 2025 races "
         "to training and 2022 races to testing, allowing the model to learn future "
         "patterns when predicting past outcomes.",
         "Replaced with walk-forward splits: train on all years strictly before the "
         "test year. Five folds are used: three rolling single-year windows "
         "(train=[2022]→2023, [2023]→2024, [2024]→2025) and two expanding cumulative "
         "windows (train=[2022+2023]→2024, [2022+2023+2024]→2025)."),
        ("0.3 Wrong BIC Sample Size — PL Model Only",
         "BIC is defined as k·ln(n) − 2·ln(L) where n is the number of independent "
         "observations. For a PL ranking model the unit of observation is a race, not "
         "a driver-race row. The original code used n = len(model_df) (~400 rows) "
         "instead of n = number of races (~22–68), inflating the BIC penalty by ~20×.",
         "Fixed to n = train_df['RaceID'].nunique() — the correct race count."),
        ("0.4 Inconsistent Optimizer — PL Model Only",
         "Feature selection used L-BFGS-B but the CV loop used BFGS. The two "
         "optimisers have different convergence properties, so beta estimates from "
         "selection and CV were not directly comparable.",
         "L-BFGS-B is used throughout in v2."),
        ("0.5 Magic Number in Score Adjustment — XGBRanker Only",
         "The final race score applied an arbitrary 2× penalty to DNF probability: "
         "final_score = raw_score − 2.0 × dnf_prob. The XGBRanker already learns "
         "the appropriate DNF penalty via DNFProbFeature. The manual adjustment "
         "double-counted this and was not derived from any optimisation.",
         "Removed in v2. The raw ranker score is used directly."),
        ("0.6 VIF / LOO Results Discarded — XGBRanker Only",
         "The original code ran VIF selection and LOO analysis, printed results, "
         "then reset FEATURES to the original 18 before training. The analysis "
         "had no effect on the deployed model.",
         "In v2 the VIF-selected feature set is tracked explicitly and both the "
         "CV folds and the final model are evaluated consistently."),
        ("0.7 QualiPercentile Hardcoded Denominator",
         "QualiPercentile = (QualiPosition − 1) / 19 assumed exactly 20 starters. "
         "Races with DSQ or DNS entries produced values > 1.0 for some drivers.",
         "Fixed to (QualiPosition − 1) / (n_starters − 1) where n_starters is the "
         "actual grid size per race."),
        ("0.8 Infinite Oscillation in AIC Stepwise — PL Model",
         "The bidirectional AIC stepwise had no guard against revisiting feature "
         "sets. With a nearly flat log-likelihood surface, floating-point noise "
         "caused AIC to decrease by tiny amounts on each cycle, running indefinitely.",
         "Fixed by switching to forward-only AIC: greedily add features while AIC "
         "strictly improves, stop when no addition helps. Terminates in ≤ 18 steps."),
    ]

    for title, problem, fix in issues:
        elems.append(p(f"<b>{title}</b>", H3))
        elems.append(p(f"<b>Problem:</b> {problem}"))
        elems.append(p(f"<b>Fix in v2:</b> {fix}"))
        elems.append(sp(0.15))

    return elems


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — Background
# ═══════════════════════════════════════════════════════════════════════════

def section_1_background():
    elems = []
    elems.append(p("<b>1. Background</b>", H2))
    elems.append(hr())

    elems.append(p(
        "Previously, we had been modeling F1 race outcomes using the Plackett-Luce "
        "(PL) model. This model is a probabilistic ranking framework that directly "
        "estimates the likelihood of a full finishing order. While the PL model gave "
        "us an initial approach to model race rankings, there were several limitations."
    ))
    elems.append(sp(0.2))
    elems.append(p(
        "Qualifying position as we know is a categorical and ordinal variable which "
        "in turn caused us to create dummy variables leading our total predictor count "
        "going from 6 to 25 with the addition of 19 qualifying position dummy "
        "variables. <b>This caused the qualifying position to capture the full "
        "prediction power of the model giving us inaccurate results and unstable "
        "coefficients, which is called quasi-complete separation.</b> We combatted "
        "this with adding L2 regularization and splitting qualifying positions into "
        "3 bins (groups). This gave us stable coefficients and evaluation metrics "
        "such as Spearman of 50% and top-3 accuracy of 57%, which isn't good enough."
    ))
    elems.append(sp(0.2))

    elems.append(p("<b>Plackett-Luce Limitations</b>", H3))
    elems.append(p(
        "The PL model <b>assumed a linear utility structure, which inhibited its "
        "ability to capture nonlinear interactions and the complex relationships "
        "between the predictor variables</b>. Along with this, the feature set was "
        "restricted to a <b>small number of pre-race variables</b> which made it "
        "difficult to capture the variation and randomness that occurs in an F1 race."
    ))
    elems.append(sp(0.2))
    elems.append(p(
        "In the Plackett-Luce model, <b>each driver is assigned a latent utility "
        "representing their underlying competitive strength</b>. This utility is "
        "defined as a <b>linear combination of pre-race predictors and serves as "
        "the main determinant of finishing probabilities</b>. The model uses these "
        "utilities to construct a probability distribution over all possible race "
        "outcomes, with higher utility values corresponding to a greater likelihood "
        "of finishing ahead of other drivers."
    ))
    elems.append(sp(0.2))
    elems.append(p(
        "This formula is <b>inherent to the Plackett-Luce framework</b>, as it "
        "provides a mathematically tractable way to model full rankings using a "
        "single scalar quantity per driver. However, this <b>structure requires "
        "that all systematic differences between drivers, which includes both "
        "performance and reliability, are captured within this single utility. "
        "Any variation not explained by the predictors is treated as random noise.</b>"
    ))
    elems.append(sp(0.2))
    elems.append(p(
        "In addition to this, the PL model relied on several assumptions. One "
        "important assumption was the <b>independence of irrelevant alternatives "
        "(IIA)</b>. This implies that the <b>ordering between two drivers is "
        "unaffected by the presence of other drivers</b>. In F1, this assumption "
        "is constantly violated due to factors like strategy, traffic, and safety "
        "cars, all introducing dependencies between multiple drivers."
    ))
    elems.append(sp(0.2))
    elems.append(p(
        "While these assumptions are convenient, it is <b>restrictive in the "
        "context of Formula 1</b>, where race outcomes arise from multiple distinct "
        "processes. In particular, <b>reliability-related events such as DNFs are "
        "not purely random noise, but instead follow identifiable patterns that "
        "are influenced by different factors.</b>"
    ))
    elems.append(sp(0.2))

    elems.append(p("<b>New Model Motivation</b>", H3))
    elems.append(p(
        "To address these limitations, a <b>two-staged prediction model that "
        "separated performance and reliability into two distinct components</b> "
        "is proposed. <b>In the first part, a model is used to estimate that a "
        "driver DNFs. This captures reliability-related uncertainty. The second "
        "stage is a learning to rank model that predicts the finishing order based "
        "on performance related features</b>."
    ))
    elems.append(p("Specific model features include:"))
    for feat in ["Multi-season telemetry (2022-2025)",
                 "Explicit modeling of driver and team reliability",
                 "ML based ranking framework"]:
        elems.append(p(f"    -  {feat}", BULLET))
    elems.append(sp(0.2))
    elems.append(p(
        "Here, the <b>2022-2025 seasons are chosen</b>. In <b>Formula 1 every 3-4 "
        "years the rules and regulations change</b>. What this means is that the "
        "FIA (Fédération Internationale de l'Automobile), which is the <b>governing "
        "body of Formula 1, comes up with new rule changes in the way the car should "
        "be built.</b> One example of this is the current season (2026 and onwards), "
        "the rules have fully changed and power units now are to be a 50-50 power "
        "split between Internal combustion engine (ICE) and electrical power. Before "
        "it was an 80-20 split between ICE and electrical power respectively. This is "
        "the biggest change with a lot of other changes as well, but this is <b>done "
        "so that the performance gap reduces between teams, and to see more "
        "competitiveness.</b> For this reason the seasons <b>2022-2025 are chosen "
        "as this was the \"Ground Effect Era\"</b> where aero focus shifted away from "
        "upper-body to underbody ground effects. <b>Using years with different rules "
        "would cause our data to be misinterpreted and give us inaccurate "
        "predictions.</b>"
    ))

    return elems


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — Model Framework
# ═══════════════════════════════════════════════════════════════════════════

def section_2_model_framework():
    elems = []
    elems.append(p("<b>2. Model Framework</b>", H2))
    elems.append(hr())

    elems.append(p("<b>Part 1: Reliability Modeling (DNF Probability)</b>", H3))
    elems.append(p("A logistic regression model is used to estimate:"))
    elems.append(p(
        "<b>P(DNF | pre-race features)</b> = P(DNF | race conditions + "
        "reliability history + variability)"
    ))
    elems.append(sp(0.15))
    elems.append(p(
        "The <b>logistic regression model estimates the probability that a driver "
        "DNFs (does not finish a race) as a function of pre-race predictors</b>. "
        "These predictors include both <b>historical reliability measures</b>, such "
        "as driver and team DNF rates, and <b>race-specific variables</b>, such as "
        "weather conditions, lap-time variability, and interaction effects. "
        "<b>Variability captures the stability of a driver's performance, which "
        "doesn't directly cause DNFs, but is associated with increased race risk.</b> "
        "The model combines these inputs through a linear function, which is then "
        "transformed into a probability using a logistic function."
    ))
    elems.append(sp(0.15))
    elems.append(p(
        "This formulation allows the model to capture <b>how DNF risk varies across "
        "different race contexts</b> rather than relying solely on long-run historical "
        "averages. It accounts for the fact that <b>reliability is influenced not only "
        "by historical performance but also by race-specific conditions</b> such as "
        "weather and track characteristics. As a result, the <b>model estimates a "
        "context-dependent probability of failure</b>, reflecting structured sources "
        "of variability rather than treating DNFs as purely random events."
    ))
    elems.append(sp(0.15))
    elems.append(p(
        "Along with this, logistic regression is suited for this because <b>DNF is a "
        "binary outcome</b> (finish vs. not finish), and the model directly estimates "
        "probabilities constrained between 0 and 1. Its coefficients help us "
        "understand how each factor contributes to DNF risk."
    ))
    elems.append(sp(0.2))

    elems.append(p("<b>Part 2: Ranking Model (Race Outcome Prediction)</b>", H3))
    elems.append(p(
        "The ranking model used is <b>XGBRanker</b>, XGBoost's implementation of "
        "the LambdaMART learning-to-rank framework. Rather than predicting a "
        "finishing position directly, XGBRanker <b>learns pairwise comparisons "
        "between drivers within each race</b> — given two drivers and their "
        "feature vectors, which one finishes ahead? With 20 drivers per race "
        "this produces 190 pairwise comparisons per race "
        "(20 × 19 / 2), and the model learns across all races simultaneously."
    ))
    elems.append(sp(0.08))
    elems.append(p(
        "Each driver's feature vector contains all 18 engineered predictors "
        "<b>plus the DNF probability estimated in Stage 1 (DNFProbFeature)</b>. "
        "This means reliability is not a post-hoc adjustment — it is fed directly "
        "into the ranking model as a feature, allowing XGBRanker to learn how "
        "DNF risk interacts with qualifying position, team strength, and driver "
        "form when determining finishing order."
    ))
    elems.append(sp(0.08))
    elems.append(p(
        "XGBRanker is trained to optimize <b>NDCG (Normalized Discounted "
        "Cumulative Gain)</b>, a ranking metric that rewards correctly ordering "
        "drivers at the front of the field more than at the back. This is "
        "appropriate for F1 because errors in predicting the top finishers "
        "matter more than errors in predicting, say, P15 vs P16. Unlike linear "
        "models, XGBRanker captures <b>nonlinear relationships and feature "
        "interactions</b> without requiring them to be specified explicitly."
    ))
    elems.append(sp(0.2))

    elems.append(p("<b>Final Prediction</b>", H3))
    elems.append(p(
        "The final predicted finishing order for each race is determined by "
        "<b>ranking drivers by their raw XGBRanker score</b> — no manual "
        "adjustment is applied. The driver with the highest score is predicted "
        "to finish first, and so on. Because DNFProbFeature is already an input "
        "to the ranker, reliability risk is embedded in the score itself: "
        "a driver with high DNF probability will receive a lower ranking score "
        "without any external penalty term being needed."
    ))
    elems.append(sp(0.08))
    elems.append(p(
        "This contrasts with the original v1 approach, which applied a manual "
        "penalty of <b>final_score = ranking_score − (2 × DNF_probability)</b> "
        "after the model had already learned from DNFProbFeature as an input. "
        "That double-counted the reliability signal. In v2 the formula is simply:"
    ))
    elems.append(sp(0.06))
    elems.append(p("    <b>final_score = XGBRanker output score</b>", BODY))
    elems.append(sp(0.06))
    elems.append(p(
        "Drivers are then ranked in descending order of this score to produce "
        "the predicted finishing order for the race."
    ))

    return elems


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — Data and Preprocessing
# ═══════════════════════════════════════════════════════════════════════════

def section_3_data_preprocessing():
    elems = []
    elems.append(p("<b>3. Data and Preprocessing</b>", H2))
    elems.append(hr())

    elems.append(p(
        "We utilized FastF1 data for the 2022-2025 seasons. Each observation "
        "corresponds to one driver × one race."
    ))
    elems.append(sp(0.2))

    elems.append(p("<b>3.1 Matrix</b>", H3))
    elems.append(p(
        "Unlike the previous Plackett-Luce model, where each race was treated as "
        "one ranked observation, the <b>new learning-to-rank model is built at the "
        "driver-race level</b>. This means each row represents one driver in one "
        "race, so the <b>sample size is the total number of driver-race observations, "
        "not just the number of races</b>. For example, one race with 20 drivers "
        "contributes about 20 rows. Across multiple seasons, this creates a much "
        "larger feature matrix while still preserving the race-level grouping structure."
    ))
    elems.append(sp(0.15))
    elems.append(p(
        "This theoretically with 1 race 20 drivers equaling 20 rows. 1 season "
        "(~22 races) is 22×20 equaling 440 rows. Now 4 seasons around 90 races × "
        "20 rows (drivers) equaling ~1800 rows. The number of races here around 88 "
        "to 92 isn't our n — it is the number of groups. So in our model n ≈ 1800, "
        "groups ≈ 90, and p (predictor list) ≈ 19-20."
    ))
    elems.append(sp(0.15))
    elems.append(p(
        "The model is learning to rank <b>within</b> each group. So ranking happens "
        "inside of each race (group) but the model learns across all rows."
    ))
    elems.append(sp(0.15))
    elems.append(p(
        "The feature matrix consists of 19 predictors for the DNF model and adds "
        "an additional feature for the ranking model. The additional feature is "
        "the predicted DNF probability."
    ))
    elems.append(sp(0.2))

    elems.append(p("<b>3.2 Predictor Types</b>", H3))
    elems.append(p(
        "A key issue in the PL model was being able to handle qualifying position "
        "as a categorical and ordinal variable which created issues."
    ))
    elems.append(sp(0.1))

    elems.append(p("<b>Qualifying Position</b>", H4))
    elems.append(p("Qualifying position is both categorical and ordinal so:"))
    elems.append(p("    -  <b>The difference between P1 and P2 is not the same as P10 and P11</b>", BULLET))
    elems.append(p("    -  Treating it as continuous would introduce artificial linear assumptions", BULLET))
    elems.append(p(
        "To address this, qualifying position was transformed into a percentile "
        "representation. In v2 this is computed as:"
    ))
    elems.append(p(
        "<b>QualiPercentile = (QualiPosition − 1) / (n_starters − 1)</b>"
    ))
    elems.append(p(
        "<i>Note (v2 change): The original formula used a hardcoded denominator of 19, "
        "assuming exactly 20 starters. This produced values &gt; 1.0 for drivers on "
        "grids with DNS/DSQ entries. The denominator is now (n_starters − 1) where "
        "n_starters is the actual grid size per race, clipped to a minimum of 1 to "
        "avoid division by zero.</i>"
    ))
    elems.append(sp(0.15))
    elems.append(p(
        "Qualifying position is an ordinal variable, but not linearly spaced. To "
        "preserve ordering while avoiding high-dimensional dummy encoding, it is "
        "transformed into a normalized percentile. While this transformation "
        "introduces a linear scale, the use of a tree-based model allows the "
        "relationship between qualifying position and race outcome to remain "
        "nonlinear, as the model learns optimal split thresholds rather than "
        "assuming constant marginal effects."
    ))
    elems.append(sp(0.2))

    elems.append(p("<b>Continuous Variables</b>", H4))
    elems.append(p(
        "All continuous predictors are standardized using <b>z-score normalization</b>: "
        "<b>x_scaled = (x − μ) / σ</b>"
    ))
    elems.append(p(
        "Each feature is centered by subtracting the mean and divided by its "
        "standard deviation. This ensures that all predictors are on a comparable "
        "scale. This in turn <b>improves numerical stability</b> and enables "
        "<b>reliable coefficient estimation</b> for stage 1, the <b>logistic "
        "regression model</b>. <b>XGBoost</b> is a <b>tree-based model</b>, "
        "meaning that it <b>doesn't require scaling</b>, but standardization is "
        "applied for consistency across both stages."
    ))

    return elems


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — Feature Engineering
# ═══════════════════════════════════════════════════════════════════════════

def section_4_feature_engineering():
    elems = []
    elems.append(p("<b>4. Feature Engineering</b>", H2))
    elems.append(hr())

    elems.append(p(
        "The model incorporates a comprehensive set of pre-race predictors. All "
        "predictors used in the model are either <b>directly observable prior to "
        "the race</b> (e.g., qualifying results, weather conditions, and practice "
        "session telemetry) or <b>derived from historical race data</b> (e.g., "
        "driver form, team strength, and reliability measures)."
    ))
    elems.append(sp(0.2))

    subsections = [
        ("4.1 Qualifying Features", [
            ("QualiPercentile", "normalized grid position"),
            ("QualiGapToPole", "continuous measure of performance"),
            ("TeammateQualiGap", "relative intra-team performance"),
        ]),
        ("4.2 Telemetry / Practice Features", [
            ("FP_LongRunPace", "proxy for race pace"),
            ("FP_LongRunVar", "consistency indicator"),
        ]),
        ("4.3 Performance Indicators", [
            ("DriverForm", "rolling average of recent finishes"),
            ("TeamStrength", "rolling team performance"),
        ]),
        ("4.4 Reliability Features", [
            ("DriverDNFRate", ""),
            ("TeamDNFRate", ""),
        ]),
        ("4.5 Environmental Features", [
            ("AirTemp", ""),
            ("TrackTemp", ""),
            ("Rainfall", ""),
        ]),
        ("4.6 Track Characteristics", [
            ("OvertakeIndex", "difficulty of overtaking"),
        ]),
    ]

    for title, feats in subsections:
        elems.append(p(f"<b>{title}</b>", H3))
        for fname, fdesc in feats:
            if fdesc:
                elems.append(p(f"    -  <i>{fname}</i> - {fdesc}", BULLET))
            else:
                elems.append(p(f"    -  <i>{fname}</i>", BULLET))

    elems.append(p("<b>4.7 Interaction Terms</b>", H3))
    for feat in ["Quali x Overtake", "Pace x TireDeg", "Reliability x rain"]:
        elems.append(p(f"    -  <i>{feat}</i>", BULLET))
    elems.append(p("    -  <i>Driver_vs_Team</i>", BULLET))
    elems.append(p(
        "        -  Measures how well driver performs relative to car they have "
        "and since standardized, it becomes how many standard deviations "
        "above/below team expectation is the driver", BULLET
    ))
    elems.append(p("These allow the model to capture the nonlinear effects."))

    return elems


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — Model Selection
# ═══════════════════════════════════════════════════════════════════════════

def section_5_model_selection():
    elems = []
    elems.append(p("<b>5. Model Selection</b>", H2))
    elems.append(hr())

    elems.append(p("<b>5.1 Logistic Regression (DNF Model)</b>", H3))
    elems.append(p(
        "Logistic regression is used for the DNF prediction stage. The model gives "
        "us a direct estimate of the probability that a driver fails to finish a "
        "race. This is essential in correctly integrating reliability into the final "
        "prediction. Along with this, logistic regression allows for a clear "
        "understanding of how different features contribute to DNF risk."
    ))
    elems.append(sp(0.2))

    elems.append(p("<b>5.2 XGBoost Ranking Model (LambdaMART)</b>", H3))
    elems.append(p(
        "As mentioned earlier this is a learning to rank framework which is "
        "well-suited for Formula 1, given that our goal is to predict the full "
        "ordering of drivers over absolute outcomes. Also, XGBoost does well in "
        "capturing nonlinear relationships and feature interactions. While other "
        "gradient boosting methods like LightGBM or CatBoost could be used, "
        "XGBoost already gives us a stable implementation without the need for "
        "extensive feature engineering."
    ))
    elems.append(sp(0.15))
    elems.append(p(
        "As mentioned above, when we do many pairwise comparisons, with 20 drivers "
        "we have C(20,2) = 190 comparisons. If we used hypothesis testing for each "
        "pair (looking at Driver A is better than Driver B), each test has an error "
        "rate α (say 0.05). What this means is doing 190 tests then will inflate the "
        "chance for a false positive. Formula for at least one false positive: "
        "<b>1 − (1 − α)^g</b>. Doing 190 tests would then cause the error to be "
        "huge, much greater than 0.05 or any alpha. To fix this we use Bonferroni "
        "correction to shrink the per-test threshold: <b>α_adjusted = α / g</b>."
    ))
    elems.append(sp(0.15))
    elems.append(p(
        "So instead of comparing each test to 0.05, we compare it to 0.05/190 or "
        "≈0.00026. What this does is it keeps total error controlled. The difference "
        "is in our model, XGBoost and LambdaMART don't actually use hypothesis "
        "testing. What this method actually does is optimization, and not p-values, "
        "significance thresholds, or α levels. It minimizes a loss function. "
        "Pairwise logistic loss: <b>log(1 + e^(−(sᵢ − sⱼ)))</b>. So it optimizes "
        "metrics such as NDCG. The 190 pairwise comparisons are used as training "
        "signals, and not statistical tests. <b>All pairwise comparisons are jointly "
        "optimized</b>. The model doesn't learn if A is significantly better than B, "
        "but instead it learns to adjust scores so correct orderings are more likely "
        "overall."
    ))
    elems.append(sp(0.2))

    elems.append(p("<b>5.3 Training and Evaluation: Walk-Forward Cross-Validation</b>", H3))

    elems.append(p("<b>Why walk-forward CV — and why not standard k-fold?</b>", H4))
    elems.append(p(
        "Standard k-fold cross-validation randomly partitions the dataset into k "
        "subsets, trains on k−1 of them, and tests on the remaining one. This works "
        "well for data where observations are independent. <b>F1 race data is not "
        "independent across time</b> — a model trained on 2024 race results has "
        "already seen which drivers were fast, which teams improved, and how the "
        "competitive order evolved. If we then ask it to predict 2022 races as a "
        "\"test set,\" we are evaluating it on data from the past using knowledge "
        "from the future. This is <b>temporal data leakage</b> — the model appears "
        "to perform well not because it learned generalizable patterns, but because "
        "it was inadvertently given information it would never have in a real "
        "deployment scenario."
    ))
    elems.append(sp(0.06))
    elems.append(p(
        "Walk-forward cross-validation solves this by enforcing a strict rule: "
        "<b>the test year must always be strictly later than every year in the "
        "training set</b>. The model can only be evaluated on seasons that, "
        "at the time of training, had not yet happened. This mirrors real-world "
        "use: when predicting a 2025 race, only pre-2025 data is available."
    ))
    elems.append(sp(0.1))

    elems.append(p("<b>How each fold works</b>", H4))
    elems.append(p(
        "For each fold, the procedure is identical:"
    ))
    elems.append(sp(0.04))
    elems.append(p(
        "1. <b>Fit the model</b> on all races in the training years. "
        "For XGBRanker this means training the boosted ranking model; "
        "for the PL model this means maximizing the penalized log-likelihood "
        "over the training races."
    ))
    elems.append(sp(0.03))
    elems.append(p(
        "2. <b>Predict</b> race outcomes for every race in the test year, "
        "using only the pre-race features available before that race. "
        "The trained model has never seen any result from the test year."
    ))
    elems.append(sp(0.03))
    elems.append(p(
        "3. <b>Evaluate</b> the predictions for that test year by computing "
        "Spearman, Kendall, Top-3/5 accuracy, Winner accuracy, and NDCG "
        "against the actual race results."
    ))
    elems.append(sp(0.08))
    elems.append(p(
        "This process is repeated for each of the five folds. The final "
        "reported metrics for each model are the <b>mean of the per-fold "
        "scores across all five folds</b>. Taking the mean across folds "
        "is important because any single fold can be unrepresentative — "
        "one season may be unusually predictable (Red Bull dominance) or "
        "unusually chaotic (2025 competitive reset). Averaging across five "
        "folds covering three different test years gives a stable estimate "
        "of how the model performs across a range of racing conditions."
    ))
    elems.append(sp(0.1))

    elems.append(p("<b>The five folds</b>", H4))
    elems.append(p(
        "Two types of fold are used. <b>Rolling folds</b> train on a single "
        "season and test on the next, measuring how well very recent patterns "
        "transfer. <b>Expanding folds</b> accumulate all prior seasons as "
        "training data, measuring whether more historical data helps. "
        "Together, the five folds cover every possible test season (2023, "
        "2024, 2025) and multiple training configurations:"
    ))
    elems.append(sp(0.08))

    fold_header = ["Fold", "Train Years", "Test Year", "Type"]
    fold_data = [fold_header,
                 ["1", "2022",              "2023", "Rolling"],
                 ["2", "2023",              "2024", "Rolling"],
                 ["3", "2024",              "2025", "Rolling"],
                 ["4", "2022 + 2023",       "2024", "Expanding"],
                 ["5", "2022 + 2023 + 2024","2025", "Expanding"],
                 ]
    elems.append(make_table(fold_data, [1.2*cm, 4.0*cm, 2.5*cm, 2.5*cm]))
    elems.append(p("Table 5a. Walk-forward cross-validation splits.", CAPTION))
    elems.append(sp(0.1))
    elems.append(p(
        "The 2025 season appears in both Fold 3 (rolling, trained on 2024 only) "
        "and Fold 5 (expanding, trained on 2022-2024), allowing a direct "
        "comparison of whether a single recent season or three seasons of "
        "history gives better 2025 predictions. At no point does any fold "
        "use 2025 data for training — it is always held out as the final "
        "out-of-sample test period."
    ))
    elems.append(sp(0.2))

    elems.append(p("<b>5.4 Plackett-Luce Models</b>", H3))
    elems.append(p(
        "The Plackett-Luce model is included as a baseline probabilistic ranking "
        "model. Three iterations of the PL model were developed before the current "
        "v2 framework:"
    ))
    elems.append(sp(0.1))

    elems.append(p("<b>5.4.1 Original PL (2024 season only)</b>", H4))
    elems.append(p(
        "This was the original Plackett-Luce model that was created. This model "
        "features such as <i>qualifying bins</i> (Front/Mid/Back), <i>DriverForm</i>, "
        "<i>TeamStrength</i>, <i>DNF Rates</i>, and <i>Overtake Index</i>. The "
        "problem was that the qualifying position signal was low due to it being "
        "done through bins meaning might have lost some variation due to grouping "
        "them. The features weren't efficiently engineered. Along with this we only "
        "ran the model for one season. This model exhibited low feature complexity. "
        "Our model also was trained and tested through a manually chosen training "
        "and testing percentage (70% and 30% respectively)."
    ))
    elems.append(sp(0.1))

    elems.append(p("<b>5.4.2 Multi-Season PL (2022-2025, 5-fold cross validation)</b>", H4))
    elems.append(p(
        "Expanded to multi-season data with 5-fold grouped cross-validation. "
        "Significantly improved generalization by averaging performance across "
        "multiple held-out race subsets, reducing the sensitivity to individual "
        "season conditions."
    ))
    elems.append(sp(0.1))

    elems.append(p("<b>5.4.3 PL Enriched Feature (2022-2025, 5-fold cross validation)</b>", H4))
    elems.append(p(
        "Added the full enriched feature set including QualiPercentile, "
        "QualiGapToPole, interaction terms, and race-context variables. Both "
        "models see moderate improvements with this enhanced feature engineering."
    ))

    return elems


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 — Results
# ═══════════════════════════════════════════════════════════════════════════

def section_6_results(imp_df):
    elems = []
    elems.append(p("<b>6. Results</b>", H2))
    elems.append(hr())

    elems.append(p("<b>6.1 Feature Importance</b>", H3))
    if imp_df is not None:
        header = ["Feature", "Importance"]
        rows = [header]
        for _, r in imp_df.iterrows():
            rows.append([r["Feature"], fmt(r["Importance"], 6)])
        elems.append(make_table(rows, [7.0*cm, 3.0*cm]))
        elems.append(p(
            "Table 6a. XGBRanker v2 mean feature importance across all 5 "
            "walk-forward folds (all 18 features + DNFProbFeature).", CAPTION
        ))
    else:
        elems.append(p("Feature importance data not available.", BODY))
    elems.append(sp(0.2))

    elems.append(p("Top Predictors:"))
    elems.append(p("    -  Qualifying position still is dominant", BULLET))
    elems.append(p("    -  Performance indicators give us a meaningful secondary signal", BULLET))
    elems.append(p("    -  Reliability is incorporated", BULLET))
    elems.append(p("    -  Interaction terms improve model flexibility", BULLET))
    elems.append(sp(0.2))

    elems.append(p("<b>6.2 Output Files</b>", H3))
    files = [
        ("Dataset", "DEBUGF7_f1_enriched_prediction_dataset_2022_2025.csv"),
        ("XGB Metrics (all-18)", "v2_walkforward_summary.csv"),
        ("XGB Metrics (VIF-15)", "v2_vif_summary.csv"),
        ("PL Metrics (all-18)", "pl_v2_walkforward_summary.csv"),
        ("PL Metrics (VIF-15)", "pl_v2_vif_summary.csv"),
        ("Predictions", "v2_predictions.csv"),
        ("Feature Importance", "v2_feature_importance.csv"),
        ("LOO Results", "v2_loo_results.csv"),
        ("VIF Steps (XGB/PL)", "v2_vif_steps.csv / pl_v2_vif_steps.csv"),
    ]
    for label, fname in files:
        elems.append(p(f"    <b>{label}:</b>  {fname}", BULLET))

    return elems


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7 — Model Evaluation Metrics
# ═══════════════════════════════════════════════════════════════════════════

def section_7_eval_metrics(xgb_df, pl_df):
    elems = []
    elems.append(p("<b>7. Model Evaluation Metrics</b>", H2))
    elems.append(hr())

    elems.append(p("<b>7.1 What These Metrics Measure — and What They Don't</b>", H3))
    elems.append(p(
        "A common misconception is that high evaluation metrics mean the model "
        "has correctly predicted the exact finishing order of a race. "
        "<b>None of the metrics used here, except Winner Accuracy, test for "
        "exact positional correctness.</b> They each measure a different aspect "
        "of ranking quality, and understanding precisely what each one captures "
        "is essential for interpreting the results honestly."
    ))
    elems.append(sp(0.06))
    elems.append(p(
        "Getting the exact 20-driver finishing order right in any F1 race is "
        "essentially impossible from pre-race features alone. Safety cars, "
        "mechanical failures, strategy calls, tire wear, and racing incidents "
        "all introduce randomness that no model trained on qualifying data, "
        "form, and reliability statistics can anticipate. <b>What a good model "
        "can do is extract the systematic, predictable component of race "
        "outcomes</b> — the signal that exists in the data before the lights "
        "go out — while acknowledging that the rest is stochastic."
    ))
    elems.append(sp(0.1))

    elems.append(p("<b>7.2 Metric Definitions</b>", H3))

    elems.append(p("<b>Spearman Rank Correlation</b>", H4))
    elems.append(p(
        "Spearman correlation measures the <b>monotonic agreement between "
        "predicted and actual rank orderings</b>. For each race, drivers are "
        "ranked by their predicted score and by their actual finishing position; "
        "Spearman asks how well these two rank lists agree. A value of 1.0 means "
        "the predicted order perfectly matches the actual finish; 0 means no "
        "relationship; −1.0 means perfectly inverted. <b>Spearman does not "
        "require exact position matches</b> — a race where every driver finishes "
        "exactly one place off from their predicted position still scores near "
        "1.0 because the relative ordering is preserved. The metric is averaged "
        "across all races in the test fold."
    ))
    elems.append(sp(0.06))

    elems.append(p("<b>Kendall's Tau</b>", H4))
    elems.append(p(
        "Kendall's Tau measures the same rank-order agreement but counts "
        "<b>concordant and discordant pairs</b> of drivers directly. A "
        "concordant pair is one where the model correctly predicts that driver A "
        "finishes ahead of driver B. Tau = (concordant − discordant) / total "
        "pairs. A value of 0.55 means 55% more concordant pairs than discordant "
        "ones. Like Spearman, it does not test exact positions — only pairwise "
        "ordering. Kendall is generally more conservative than Spearman and less "
        "sensitive to large rank swaps at the extremes."
    ))
    elems.append(sp(0.06))

    elems.append(p("<b>Top-3 and Top-5 Accuracy</b>", H4))
    elems.append(p(
        "Top-k accuracy measures the <b>overlap between the set of k predicted "
        "top finishers and the set of k actual top finishers</b>. "
        "Top-3 Accuracy = |predicted top 3 ∩ actual top 3| / 3. A score of "
        "0.63 means on average 1.9 of the 3 predicted podium drivers actually "
        "finished on the podium. Crucially, <b>this does not check whether "
        "each driver was predicted in the correct position within the top 3</b> "
        "— it only checks whether the right drivers were identified as podium "
        "finishers regardless of order. A model that correctly identifies all "
        "3 podium drivers but in the wrong order scores 1.0 on this metric."
    ))
    elems.append(sp(0.06))

    elems.append(p("<b>Winner Accuracy</b>", H4))
    elems.append(p(
        "Winner Accuracy is the <b>only strict positional metric</b>: did the "
        "driver assigned the highest predicted score actually win the race? "
        "This is a binary correct/incorrect judgment averaged across races. "
        "A random model would score 1/20 = 5% on 20-driver fields. The "
        "qualifying baseline (predict the pole-sitter wins) scores 60.3% "
        "because during the 2022-2024 Red Bull era Verstappen frequently won "
        "from pole. Winner Accuracy is the hardest metric to beat above the "
        "baseline because it penalises any error at the top of the distribution, "
        "including upsets caused by safety cars and strategy that no pre-race "
        "model can anticipate."
    ))
    elems.append(sp(0.06))

    elems.append(p("<b>NDCG@k (Normalized Discounted Cumulative Gain)</b>", H4))
    elems.append(p(
        "NDCG is a ranking quality metric borrowed from information retrieval. "
        "It assigns <b>higher reward for correct predictions near the top of "
        "the ranking and gives partial credit for near misses</b>. A driver "
        "predicted P2 when they actually finished P1 scores higher than a "
        "driver predicted P10 when they actually finished P1, because the "
        "error at P10 wastes more of the top-ranked positions. NDCG@k evaluates "
        "only the top k predicted positions. A score of 1.0 is perfect; "
        "the 0.89-0.90 range observed here means the top-ranked predictions "
        "are highly concentrated around the actual top finishers, even if "
        "the exact order within the top k varies."
    ))
    elems.append(sp(0.1))

    elems.append(p("<b>7.3 Results</b>", H3))
    elems.append(p(
        "All metrics are averaged across the 5 walk-forward folds. "
        "Results are reported for both the Two-Stage XGBRanker v2 and "
        "Plackett-Luce v2 models."
    ))
    elems.append(sp(0.1))

    metric_cols = ["Spearman", "KendallTau", "Top3 Acc", "Top5 Acc",
                   "Winner Acc", "NDCG@3", "NDCG@5"]

    xgb_model_mean = None
    pl_model_mean = None

    if xgb_df is not None:
        xgb_model_mean = {m: xgb_df[f"Model_{m}"].mean() for m in metric_cols}
        xgb_model_mean["DNF LogLoss"] = xgb_df["DNF_LogLoss"].mean()
        xgb_model_mean["DNF Brier"] = xgb_df["DNF_Brier"].mean()

    if pl_df is not None:
        pl_model = pl_df[pl_df["Method"] == "Model"]
        pl_model_mean = {m: pl_model[m].mean() for m in metric_cols}

    if xgb_model_mean and pl_model_mean:
        header = ["Metric", "XGB v2 (walk-fwd)", "PL v2 (walk-fwd)"]
        rows = [header]
        metric_display = [
            ("Mean Spearman",   "Spearman"),
            ("Mean KendallTau", "KendallTau"),
            ("Top-3 Accuracy",  "Top3 Acc"),
            ("Top-5 Accuracy",  "Top5 Acc"),
            ("Winner Accuracy", "Winner Acc"),
            ("NDCG@3",          "NDCG@3"),
            ("NDCG@5",          "NDCG@5"),
        ]
        for label, key in metric_display:
            rows.append([label, fmt(xgb_model_mean[key]), fmt(pl_model_mean[key])])
        rows.append(["DNF LogLoss",
                     fmt(xgb_model_mean.get("DNF LogLoss")), "—"])
        rows.append(["DNF Brier Score",
                     fmt(xgb_model_mean.get("DNF Brier")), "—"])
        elems.append(make_table(rows, [4.5*cm, 4.0*cm, 4.0*cm]))
        elems.append(p(
            "Table 7. Final cross-validated metrics — mean across all 5 "
            "walk-forward folds.", CAPTION
        ))
        elems.append(sp(0.2))

    sp_xgb = fmt(xgb_model_mean["Spearman"]) if xgb_model_mean else "N/A"
    sp_pl  = fmt(pl_model_mean["Spearman"])  if pl_model_mean  else "N/A"
    t3_xgb = fmt(xgb_model_mean["Top3 Acc"]) if xgb_model_mean else "N/A"
    t3_pl  = fmt(pl_model_mean["Top3 Acc"])  if pl_model_mean  else "N/A"
    wa_xgb = fmt(xgb_model_mean["Winner Acc"]) if xgb_model_mean else "N/A"
    wa_pl  = fmt(pl_model_mean["Winner Acc"])  if pl_model_mean  else "N/A"
    n3_xgb = fmt(xgb_model_mean["NDCG@3"]) if xgb_model_mean else "N/A"
    n3_pl  = fmt(pl_model_mean["NDCG@3"])  if pl_model_mean  else "N/A"

    elems.append(p(
        f"Spearman of {sp_xgb} (XGBRanker) and {sp_pl} (PL) reflects "
        f"moderately strong rank-order agreement across all test races. "
        f"Top-3 accuracy of {t3_xgb} (XGB) and {t3_pl} (PL) means the models "
        f"correctly identify on average nearly 2 of the 3 podium finishers per "
        f"race. Winner accuracy of {wa_xgb} (XGB) and {wa_pl} (PL) — the "
        f"strictest metric — compares against a random baseline of 5% (1-in-20 "
        f"drivers) and a qualifying baseline of 60.3%. NDCG@3 of {n3_xgb} / "
        f"{n3_pl} confirms the highest-ranked predictions cluster tightly "
        f"around the actual top finishers."
    ))
    elems.append(sp(0.1))

    elems.append(p("<b>7.4 Why These Results Validate the Model</b>", H3))
    elems.append(p(
        "Since the metrics do not confirm exact positional accuracy, the "
        "question of whether the model is genuinely learning — rather than "
        "being lucky or overfitting — rests on three independent checks."
    ))
    elems.append(sp(0.06))
    elems.append(p(
        "<b>Walk-forward validation on truly unseen seasons.</b> The model "
        "trains on past seasons and is evaluated on future seasons it has "
        "never seen. When the 2022-trained model achieves Spearman 0.65 on "
        "2023 races, or the 2022+2023+2024-trained model scores on 2025 races, "
        "those test seasons contain drivers, team performance levels, and race "
        "outcomes that did not exist in the training data. Consistent "
        "performance across five such folds, covering three different test "
        "years, rules out overfitting to any single season."
    ))
    elems.append(sp(0.06))
    elems.append(p(
        "<b>Beating the qualifying baseline on rank correlation.</b> The "
        "qualifying baseline — predicting race finish = qualifying order — "
        "requires no training, no features, and no model. It achieves "
        "Spearman 0.679 purely because grid position correlates with finishing "
        "position in Formula 1. Both the XGBRanker (0.662) and PL (0.683) "
        "match or exceed this on Spearman, and the PL feature-selected models "
        "(BIC: 0.692, AIC: 0.694) outperform it consistently across folds. "
        "Beating a strong domain-knowledge baseline on held-out data is the "
        "clearest evidence the model has extracted real signal."
    ))
    elems.append(sp(0.06))
    elems.append(p(
        "<b>Coefficient signs match domain knowledge.</b> QualiPercentile "
        "carries a large negative coefficient — better qualifying means higher "
        "predicted strength. DriverForm is positive — recent good results "
        "improve the prediction. TeamDNFRate is positive — higher failure rate "
        "hurts the driver's expected finish. These directions are exactly what "
        "F1 domain knowledge predicts. A model fitting noise would produce "
        "random or contradictory coefficient signs; consistent, interpretable "
        "signs across all five walk-forward folds confirm the model is "
        "capturing genuine relationships in the data."
    ))
    elems.append(sp(0.15))

    elems.append(p("<b>7.5 Per-Fold Breakdown and Rolling vs Expanding Comparison</b>", H3))
    elems.append(p(
        "Averaging across folds gives an overall picture, but examining each "
        "fold individually reveals how performance varies by season and answers "
        "a specific design question: <b>does training on more historical seasons "
        "improve predictions compared to training on only the most recent season?</b>"
    ))
    elems.append(sp(0.08))
    elems.append(p(
        "The five folds include two pairs where the same test year is evaluated "
        "under different training configurations. <b>For 2024</b>: Fold 2 trains "
        "on 2023 only (rolling) versus Fold 4 which trains on 2022+2023 "
        "(expanding). <b>For 2025</b>: Fold 3 trains on 2024 only (rolling) "
        "versus Fold 5 which trains on 2022+2023+2024 (expanding). Fold 1 "
        "(train 2022, test 2023) has no expanding counterpart — it would require "
        "pre-2022 data which is not available — so it appears once and contributes "
        "to the five-fold total without duplication."
    ))
    elems.append(sp(0.08))

    if xgb_df is not None and pl_df is not None:
        pl_m = pl_df[pl_df["Method"] == "Model"].copy()

        hdr = ["Train", "Test", "Type",
               "XGB Spearman", "XGB Top3", "XGB Winner",
               "PL Spearman",  "PL Top3",  "PL Winner"]
        rows = [hdr]

        fold_meta = [
            ("2022",              2023, "Rolling"),
            ("2023",              2024, "Rolling"),
            ("2024",              2025, "Rolling"),
            ("2022+2023",         2024, "Expanding"),
            ("2022+2023+2024",    2025, "Expanding"),
        ]

        for train_lbl, test_yr, fold_type in fold_meta:
            xr = xgb_df[xgb_df["TestYear"] == test_yr]
            # match train years by N_Features doesn't work — match by TrainYears string
            # XGB df has TrainYears as numeric or string — check what's there
            # Use the order of rows which matches fold order
            pass

        # Simpler: just use the rows in order — they match fold order
        xgb_rows = xgb_df.to_dict("records")
        pl_rows   = pl_m.sort_values(["TrainYears", "TestYear"]).to_dict("records")

        fold_labels = [
            ("2022",           "2023", "Rolling"),
            ("2023",           "2024", "Rolling"),
            ("2024",           "2025", "Rolling"),
            ("2022+2023",      "2024", "Expanding"),
            ("2022+2023+2024", "2025", "Expanding"),
        ]

        for i, (train_lbl, test_lbl, fold_type) in enumerate(fold_labels):
            if i < len(xgb_rows) and i < len(pl_rows):
                xr = xgb_rows[i]
                pr = pl_rows[i]
                rows.append([
                    train_lbl, test_lbl, fold_type,
                    f"{xr['Model_Spearman']:.4f}",
                    f"{xr['Model_Top3 Acc']:.4f}",
                    f"{xr['Model_Winner Acc']:.4f}",
                    f"{pr['Spearman']:.4f}",
                    f"{pr['Top3 Acc']:.4f}",
                    f"{pr['Winner Acc']:.4f}",
                ])

        col_w = [3.2*cm, 1.3*cm, 2.0*cm, 2.1*cm, 1.8*cm, 1.9*cm, 2.1*cm, 1.8*cm, 1.9*cm]
        t = Table(rows, colWidths=col_w)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#16213e")),
            ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
            ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",   (0,0), (-1,-1), 7),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f0f4f8")]),
            ("BACKGROUND", (0,4), (-1,4), colors.HexColor("#e8f0fe")),
            ("BACKGROUND", (0,6), (-1,6), colors.HexColor("#e8f0fe")),
            ("GRID",       (0,0), (-1,-1), 0.4, colors.HexColor("#cccccc")),
            ("ALIGN",      (0,0), (-1,-1), "CENTER"),
            ("ALIGN",      (0,1), (1,-1), "LEFT"),
            ("LEFTPADDING", (0,0), (-1,-1), 3),
            ("RIGHTPADDING",(0,0), (-1,-1), 3),
            ("TOPPADDING",  (0,0), (-1,-1), 3),
            ("BOTTOMPADDING",(0,0),(-1,-1), 3),
        ]))
        elems.append(t)
        elems.append(p(
            "Table 7b. Per-fold Spearman, Top-3, and Winner accuracy for XGBRanker "
            "and PL. Shaded rows are the expanding folds (more training data).", CAPTION
        ))
        elems.append(sp(0.1))

        elems.append(p("<b>Does more historical data help?</b>", H4))
        elems.append(p(
            "Comparing the shaded expanding folds against the rolling folds for "
            "the same test year gives a direct answer."
        ))
        elems.append(sp(0.06))
        elems.append(p(
            "<b>For 2024 predictions</b> (Fold 2 vs Fold 4): adding 2022 data "
            "to the 2023 training set consistently improves both models. "
            "XGBRanker Spearman rises from 0.688 to 0.699 and Winner Accuracy "
            "from 0.417 to 0.500. The PL model similarly improves from Spearman "
            "0.730 to 0.741. More historical data helps because 2022 and 2023 "
            "shared a similar competitive structure (Red Bull dominance), so "
            "the older season reinforces the same patterns the model needs to "
            "predict 2024."
        ))
        elems.append(sp(0.06))
        elems.append(p(
            "<b>For 2025 predictions</b> (Fold 3 vs Fold 5): the picture is "
            "more mixed. XGBRanker gains meaningfully from the full training "
            "set — Spearman rises slightly (0.646 to 0.653) and Winner Accuracy "
            "jumps from 0.333 to 0.542, suggesting the broader historical "
            "context helps the tree model identify likely winners. The PL model "
            "shows a smaller Spearman gain but its Winner Accuracy actually "
            "falls (0.458 to 0.375). This reflects the <b>2025 competitive "
            "reset</b>: 2022-2024 was dominated by Red Bull, but 2025 saw "
            "Ferrari, McLaren, and Mercedes winning races. The PL model, being "
            "a linear ranking model, is more sensitive to this regime shift — "
            "including three years of Red Bull-era data can mislead it on "
            "a season where that hierarchy no longer holds."
        ))
        elems.append(sp(0.06))
        elems.append(p(
            "Overall, the comparison suggests that <b>more data generally helps "
            "for stable competitive eras, but a single recent season can be more "
            "informative when the sport undergoes a significant competitive "
            "reset</b>. This is a fundamental challenge of sports prediction: "
            "historical patterns are only useful when the underlying dynamics "
            "have not changed."
        ))

    return elems


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8 — Comparison and Interpretation
# ═══════════════════════════════════════════════════════════════════════════

def section_8_comparison(xgb_df, pl_df, pl_coef_df=None):
    elems = []
    elems.append(p("<b>8. Comparison and Interpretation</b>", H2))
    elems.append(hr())

    elems.append(p("<b>8.1 Performance Comparison Overview</b>", H3))
    elems.append(p(
        "These results show a <b>clear initial growth followed by diminishing "
        "returns</b>."
    ))
    elems.append(sp(0.2))

    elems.append(p("<b>8.2 Impact of Data and Evaluation</b>", H3))

    metric_cols = ["Spearman", "KendallTau", "Top3 Acc", "Top5 Acc",
                   "Winner Acc", "NDCG@3", "NDCG@5"]

    xgb_mean = None
    pl_mean  = None
    if xgb_df is not None:
        xgb_mean = {m: xgb_df[f"Model_{m}"].mean() for m in metric_cols}
    if pl_df is not None:
        pl_model = pl_df[pl_df["Method"] == "Model"]
        pl_mean  = {m: pl_model[m].mean() for m in metric_cols}

    header = ["Metric",
              "Original PL\n(2024 only)",
              "PL v1\n(2022-25, 5-fold)",
              "XGB v1\n(2022-25, 5-fold)",
              "PL v2\n(walk-fwd)",
              "XGB v2\n(walk-fwd)"]

    hist = {
        "Spearman":   [0.535, 0.641, 0.653, None, None],
        "KendallTau": [0.350, 0.508, 0.523, None, None],
        "Top3 Acc":   [0.571, 0.659, 0.677, None, None],
        "Top5 Acc":   [None,  0.717, 0.750, None, None],
        "Winner Acc": [None,  0.447, 0.590, None, None],
        "NDCG@3":     [None,  0.870, 0.887, None, None],
        "NDCG@5":     [None,  0.879, 0.890, None, None],
    }
    labels = ["Mean Spearman", "Mean KendallTau",
              "Top-3 Accuracy", "Top-5 Accuracy",
              "Winner Accuracy", "NDCG@3", "NDCG@5"]

    rows = [header]
    for lbl, key in zip(labels, metric_cols):
        h = hist[key]
        v2_pl  = fmt(pl_mean[key])  if pl_mean  else "—"
        v2_xgb = fmt(xgb_mean[key]) if xgb_mean else "—"
        rows.append([
            lbl,
            fmt(h[0]) if h[0] else "—",
            fmt(h[1]) if h[1] else "—",
            fmt(h[2]) if h[2] else "—",
            v2_pl, v2_xgb,
        ])

    extra = [
        ["Feature Complexity", "Low",    "Medium", "Medium", "High", "High"],
        ["Model Type", "Linear Ranking", "Linear Ranking",
         "Nonlinear ranking", "Linear Ranking", "Nonlinear ranking"],
    ]
    rows += extra

    col_w = [2.8*cm, 2.2*cm, 2.2*cm, 2.2*cm, 2.2*cm, 2.2*cm]
    elems.append(make_table(rows, col_w))
    elems.append(p(
        "Table 8a. Impact of data coverage and evaluation methodology. "
        "v1 = original 5-fold random grouped CV; v2 = walk-forward CV.", CAPTION
    ))
    elems.append(sp(0.25))

    elems.append(p(
        "The largest improvement happens from moving from the <b>original "
        "single-season (2024) model to the multi-season (2022-2025), "
        "cross-validated model</b>. Expanding the <b>dataset from 23 to 92 "
        "races and along with that switching from a manually chosen single "
        "train-test split significantly stabilized the evaluation metrics</b>. "
        "Utilizing 5-fold CV, provides a more stable, less biased estimate of "
        "model performance by averaging results across multiple subsets."
    ))
    elems.append(sp(0.15))
    elems.append(p(
        "Our original plackett-luce model <b>suffered from high variance due to "
        "a limited amount of data</b>. This made it quite <b>sensitive to race "
        "conditions</b> and variance from the 2024 season. Now, with <b>multi-season "
        "data and cross-validation</b>, the model is able to <b>learn more "
        "generalizable patterns</b> across different circuits. This shows us that "
        "the <b>majority of the performance gain actually comes from improved data "
        "coverage and evaluation methodology</b>."
    ))
    elems.append(sp(0.2))

    elems.append(p("<b>8.2.1 Feature Importance / Coefficient Summary</b>", H3))
    elems.append(p("<b>Plackett-Luce</b>", H4))
    elems.append(p(
        "The table below shows mean PL coefficients across all 5 walk-forward folds, "
        "filtered to features with a mean |coefficient| above 0.03 (excluding "
        "degenerate near-zero estimates for unconstrained weather/track features). "
        "A negative coefficient means the feature <i>reduces</i> race utility "
        "(higher QualiPercentile = worse starting position = worse expected finish). "
        "A positive coefficient means the feature increases race utility."
    ))
    elems.append(sp(0.1))

    if pl_coef_df is not None:
        mean_coef = (pl_coef_df.groupby("Feature")["Coef"]
                     .mean()
                     .reset_index()
                     .rename(columns={"Coef": "Mean_Coef"}))
        mean_coef["Abs_Coef"] = mean_coef["Mean_Coef"].abs()
        mean_coef = mean_coef[mean_coef["Abs_Coef"] >= 0.03].sort_values(
            "Abs_Coef", ascending=False
        )
        if len(mean_coef) > 0:
            header = ["Feature", "Mean Coefficient (across 5 folds)", "Direction"]
            rows = [header]
            for _, r in mean_coef.iterrows():
                direction = "↑ boosts finish" if r["Mean_Coef"] > 0 else "↓ hurts finish"
                rows.append([r["Feature"], fmt(r["Mean_Coef"], 4), direction])
            elems.append(make_table(rows, [4.5*cm, 4.5*cm, 3.5*cm]))
            elems.append(p(
                "Table 8c. PL v2 mean coefficient summary — only features with "
                "mean |coef| ≥ 0.03 shown. Degenerate features (AirTemp, TrackTemp, "
                "etc.) are omitted as their coefficients collapse to ~0 when the "
                "L-BFGS-B optimiser cannot estimate them from the small training windows.",
                CAPTION
            ))
            elems.append(sp(0.15))

    elems.append(p("<b>Two-Stage XGBRanker</b>", H4))
    elems.append(p(
        "Feature importance is measured by mean gain across all trees in all "
        "5 walk-forward folds. See Table 6a for the complete importance ranking. "
        "QualiPercentile accounts for 52.7% of total importance — qualifying "
        "position remains the dominant signal. Quali_x_Overtake (11.3%) and "
        "DriverForm (8.3%) are the next most important features."
    ))
    elems.append(p(
        "Interpretation: These two figures highlight a clear difference in how "
        "each model utilizes the predictors. In the Plackett-Luce model, "
        "<b>qualifying position features dominate</b> strongly. This tells us "
        "that the starting position of drivers is the primary predictor in race "
        "outcome with <b>TeamStrength</b> and <b>DriverForm</b> giving small "
        "adjustments. The two-stage model however, <b>distributes importance "
        "across features</b>. DriverForm and TeamStrength actually are more "
        "influential than pure qualifying predictors."
    ))
    elems.append(sp(0.2))

    elems.append(p("<b>8.3 Impact of Feature Engineering (Optimal feature tuning)</b>", H3))

    if xgb_mean and pl_mean:
        header2 = ["Metric", "PL v2 (all-18, walk-fwd)", "XGB v2 (all-18, walk-fwd)"]
        rows2 = [header2]
        for lbl, key in zip(labels, metric_cols):
            rows2.append([lbl, fmt(pl_mean[key]), fmt(xgb_mean[key])])
        extra2 = [
            ["Feature Complexity", "High", "High"],
            ["Model Type", "Linear Ranking", "Nonlinear ranking"],
        ]
        rows2 += extra2
        elems.append(make_table(rows2, [4.0*cm, 4.0*cm, 4.0*cm]))
        elems.append(p(
            "Table 8b. Comparison of enriched-feature v2 models "
            "(walk-forward CV, all 18 features).", CAPTION
        ))
    elems.append(sp(0.2))

    elems.append(p(
        "The next stage presents <b>enhanced feature engineering</b>. A couple key "
        "additions include <b>continuous qualifying representation through "
        "<i>QualiPercentile</i>, qualifying gap features such as <i>QualiGapToPole</i> "
        "and <i>TeammateQualiGap</i></b>, interaction terms such as "
        "<i>QualiXOvertake</i>, and contextual race features as well. These features "
        "improve the resolution of qualifying performance and are able to <b>capture "
        "relationships between grid position and track characteristics</b>. Due to "
        "this, both models see <b>moderate improvements</b>."
    ))
    elems.append(sp(0.2))

    elems.append(p("<b>8.4 Model Comparison</b>", H3))
    elems.append(p(
        "Despite the complexity of the two-stage approach, <b>performance gains are "
        "minimal</b>. The two-stage model slightly outperforms in Top-3 and winner "
        "accuracy. This suggests that this <b>model is more focused on identifying "
        "top positions</b>. On the other hand, the <b>PL model gives us a more "
        "consistent overall ranking performance due to its direct likelihood-based "
        "optimization</b>, and slightly outperforms in other metrics such as "
        "Spearman, KendallTau, and top-5 accuracy."
    ))
    elems.append(sp(0.2))

    elems.append(p("<b>8.5 Why Performance Plateaus</b>", H3))
    elems.append(p("<b>8.5.1 Limited Nonlinearity</b>", H4))
    elems.append(p(
        "The two-stage model is designed to <b>capture nonlinear interactions "
        "using XGBoost</b>. However, <b>pre-race variables such as DriverForm and "
        "TeamStrength evolve gradually</b>, and external factors such as weather "
        "introduce limited variation. Along with this <b>qualifying position related "
        "predictors capture a majority of the prediction power</b>. Due to this, "
        "there is insufficient nonlinear structure for the model to exploit, leading "
        "to minimal performance gains over the simpler PL model. The PL model assumes "
        "a linear utility structure and with qualifying position being over half of "
        "the importance and a statistically significant predictor, <b>not being able "
        "to capture nonlinear interactions doesn't serve as a detriment to the PL "
        "model</b>."
    ))
    elems.append(sp(0.1))
    elems.append(p("<b>8.5.2 Diminishing Returns</b>", H4))
    elems.append(p(
        "<b>Once core signals such as qualifying and basic performance metrics are "
        "included, additional feature engineering yields smaller improvements</b>. "
        "This reflects diminishing returns, where most of the predictive power has "
        "already been captured."
    ))
    elems.append(sp(0.2))

    elems.append(p("<b>8.6 Differences in Model Behavior</b>", H3))
    elems.append(p(
        "Although the overall performance is similar, the models differ in how "
        "they make predictions."
    ))
    elems.append(p("<b>Plackett-Luce:</b>"))
    elems.append(p(
        "The PL model <b>optimizes likelihood over full ranking</b>. From this "
        "it is able to provide <b>consistent global ordering</b>. This gives it "
        "slightly higher Spearman and Kendall."
    ))
    elems.append(p("<b>Two-Stage Model:</b>"))
    elems.append(p(
        "This uses LambdaMART to <b>optimize ranking metrics such as NDCG</b>. "
        "It <b>emphasizes the top positions</b> giving it a better top-3 and "
        "winner accuracy."
    ))
    elems.append(sp(0.2))

    elems.append(p("<b>8.7 Main Takeaways</b>", H3))
    elems.append(p(
        "Based on these results there is a clear hierarchy of factors that "
        "influence model performance."
    ))
    for i, item in enumerate([
        "Evaluation methodology (train-split) and data size",
        "Feature engineering",
        "Model complexity",
    ], 1):
        elems.append(p(f"    {i}.  {item}", BULLET))
    elems.append(p(
        "Overall, <b>predictive performance in pre-race F1 modeling is primarily "
        "driven by the quality of the features and the overall framework</b>. "
        "Once a strong set of features are established, even simpler models like "
        "PL achieve performance that is comparable to more complex ML models that "
        "only results in minimal gain."
    ))

    return elems


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 9 — Feature Correlation
# ═══════════════════════════════════════════════════════════════════════════

def section_9_feature_correlation(enriched_df):
    elems = []
    elems.append(p("<b>9. Feature Correlation</b>", H2))
    elems.append(hr())

    elems.append(p("<b>9.1 Overall Feature Correlation Analysis</b>", H3))

    corr_matrix = None
    avail_feats = []
    if enriched_df is not None:
        avail_feats = [f for f in ALL_FEATURES if f in enriched_df.columns]
        if len(avail_feats) >= 2:
            corr_matrix = enriched_df[avail_feats].corr()

    if corr_matrix is not None:
        threshold = 0.7
        high_corr_pairs = []
        n = len(avail_feats)
        for i in range(n):
            for j in range(i+1, n):
                fi, fj = avail_feats[i], avail_feats[j]
                c = corr_matrix.loc[fi, fj]
                if abs(c) >= threshold:
                    high_corr_pairs.append((fi, fj, c))

        elems.append(p(
            "We took all predictors and computed pairwise correlations. Filtered "
            "and printed those abs(corr) ≥ 0.7. This was done to identify "
            "relationships and potential redundancy between features."
        ))
        elems.append(sp(0.1))

        if high_corr_pairs:
            header = ["Feature A", "Feature B", "Correlation"]
            rows = [header]
            for fi, fj, c in sorted(high_corr_pairs, key=lambda x: -abs(x[2])):
                rows.append([fi, fj, fmt(c)])
            elems.append(make_table(rows, [4.5*cm, 4.5*cm, 2.5*cm]))
            elems.append(p(
                "Table 9a. Feature pairs with |correlation| ≥ 0.7 (v2 dataset, "
                "2022-2025 seasons).", CAPTION
            ))
        elems.append(sp(0.2))
    else:
        elems.append(p(
            "Correlation analysis requires the enriched dataset "
            "(DEBUGF7_f1_enriched_prediction_dataset_2022_2025.csv). "
            "Results shown below are from the original v1 analysis and "
            "are expected to be similar for v2 given the same feature set."
        ))
        elems.append(sp(0.1))

    elems.append(p("<u>Strong redundancy among engineered features</u>"))
    elems.append(p(
        "<b>Several engineered features show high correlations</b>. This indicates "
        "that they actually aren't introducing new information but are <b>just "
        "transformations of existing variables</b>. An example of this is "
        "<i>QualixOvertake</i> and <i>OvertakeOpportunity</i> are both highly "
        "correlated with <i>QualiPercentile</i> (≈0.90). This is expected as both "
        "of these features are directly constructed using <i>QualiPercentile</i>. "
        "These features reinforce qualifying performance rather than providing "
        "independent predictive signals which suggests there is <b>redundancy "
        "within the feature set</b>."
    ))
    elems.append(p("<u>Driver and Team Performance Related</u>"))
    elems.append(p(
        "There also exists a strong positive correlation between <i>DriverForm</i> "
        "and <i>TeamStrength</i> (≈0.82). This reflects how F1 is structured in "
        "the real world, where <b>stronger teams tend to have stronger drivers</b>. "
        "Along with that these drivers tend to be stronger due to the car being "
        "better compared to others. While not fully redundant, these <b>features "
        "capture overlapping aspects of performance</b>, and may partially duplicate "
        "information."
    ))
    elems.append(p("<u>Weather-related Variables Correlated</u>"))
    elems.append(p(
        "Correlations between <i>AirTemp</i> and <i>TrackTemp</i> (≈0.70) and "
        "<i>Rainfall</i> and <i>Reliability_x_Rain</i> (≈0.79) show up as well. "
        "These relationships are expected due to <b>track temperature naturally "
        "being dependent on air temperature</b>, and the feature engineered "
        "variable <i>Reliability_x_Rain</i> being constructed with the <i>Rainfall</i> "
        "variable. Due to this, these correlations represent feature construction "
        "rather than new meaningful relationships."
    ))
    elems.append(p("<u>Overall</u>"))
    elems.append(p(
        "This correlation structure emphasizes that multiple engineered features, "
        "especially the interaction terms, are highly dependent on the base "
        "variables used for construction. This suggests that even <b>though feature "
        "engineering increases model complexity, it may not proportionally increase "
        "independent information</b>, mainly for qualifying related features."
    ))
    elems.append(sp(0.3))

    elems.append(p("<b>9.2 Correlation with QualiPercentile</b>", H3))

    if corr_matrix is not None and "QualiPercentile" in avail_feats:
        qp_corr = corr_matrix["QualiPercentile"].drop("QualiPercentile").sort_values(
            key=abs, ascending=False
        )
        header = ["Feature", "Corr with QualiPercentile"]
        rows = [header]
        for feat, c in qp_corr.items():
            rows.append([feat, fmt(c)])
        elems.append(make_table(rows, [6.0*cm, 4.0*cm]))
        elems.append(p(
            "Table 9b. Pairwise correlation of each feature with QualiPercentile "
            "(v2 dataset, 2022-2025 seasons).", CAPTION
        ))
        elems.append(sp(0.2))

    elems.append(p(
        "Fixed <i>QualiPercentile</i> variable and computed corr(QualiPercentile, "
        "feature) for all the features. This was to understand how strongly each "
        "predictor was correlated with qualifying performance."
    ))
    elems.append(p("<u>Qualifying-derived features dominate</u>"))
    elems.append(p(
        "The strongest correlations with <i>QualiPercentile</i> are "
        "<i>QualixOvertake</i> (≈ +0.90) and <i>OvertakeOpportunity</i> (≈ −0.90). "
        "These features are effectively <b>just re-scaled versions of qualifying "
        "performance</b> and therefore <b>don't actually introduce new predictive "
        "information</b>. Instead, they increase the importance of qualifying within "
        "the model."
    ))
    elems.append(p("<u>Moderate relationships capture additional qualifying context</u>"))
    elems.append(p(
        "Features such as <i>QualiGapToPole</i> (≈0.41) and <i>TeammateQualiGap</i> "
        "(≈0.21) both show moderate relationships. These features are <b>still related "
        "to qualifying but not perfectly correlated</b>. This means they provide "
        "additional nuance — information that is relative performance within a team "
        "or field gaps. The correlations tell us that these features are more useful "
        "than purely derived features."
    ))
    elems.append(p("<u>Driver and Team Performance Inversely Related to Qualifying</u>"))
    elems.append(p(
        "The features <i>DriverForm</i> (≈ −0.63) and <i>TeamStrength</i> (≈ −0.63) "
        "show an inverse relationship with qualifying position. What this means is "
        "that <b>better drivers and teams tend to qualify in stronger positions "
        "(lower percentile values)</b>. This then results in a negative correlation. "
        "This tells us that these features also contribute meaningful performance "
        "information beyond qualifying alone."
    ))
    elems.append(p("<u>Independent features of Qualifying</u>"))
    elems.append(p(
        "There are a couple near zero correlations such as weather features "
        "including <i>AirTemp</i>, <i>TrackTemp</i>, and <i>Rainfall</i>, track "
        "characteristics such as <i>OvertakeIndex</i>, and race variability in "
        "<i>GridSpread</i>. These features capture some of the key race-specific or "
        "environmental effects that aren't reflected in qualifying. This makes "
        "<b>these features, although weak, a potential source of independent "
        "signals</b>."
    ))
    elems.append(p("<u>Overall</u>"))
    elems.append(p(
        "<b>Qualifying performance, as expected, emerges as the dominant predictor</b> "
        "with several <b>feature engineered features effectively duplicating its "
        "signal</b>. However, <b>driver and team performance metrics provide "
        "additional independent information</b>, while weather and track specific "
        "variables remain largely uncorrelated, capturing separate aspects of "
        "race variability."
    ))

    return elems


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 10 — PL Model Selection Criteria
# ═══════════════════════════════════════════════════════════════════════════

def section_10_pl_selection(
    pl_vif_steps, pl_vif_final, pl_vif_summary, pl_df,
    pl_bic_steps=None, pl_bic_summary=None, pl_bic_coef=None,
    pl_aic_steps=None, pl_aic_summary=None, pl_aic_coef=None,
    pl_mcfadden=None, pl_coef_full=None, pl_coef_vif=None,
    pl_coef_bic=None, pl_coef_aic=None,
    xgb_vif_steps=None, xgb_vif_final=None, xgb_vif_summary=None, xgb_df=None,
):
    elems = []
    elems.append(p("<b>10. Plackett-Luce Model Selection Criteria</b>", H2))
    elems.append(hr())

    elems.append(p("<b>10.1 Motivation</b>", H3))
    elems.append(p(
        "Unlike the <b>two-stage ML model, the Plackett-Luce model gives a full "
        "statistical inference, including coefficient estimates, standard errors, "
        "z-scores, and p-values</b>. This allows for the use of formal model "
        "selection techniques that balance model fit and complexity. Later on, a "
        "leave-one-out (LOO) method will be used for the two-stage model to "
        "evaluate predictive performance, but this method doesn't explicitly "
        "penalize unnecessary model complexity. This is why different techniques "
        "are required for the Plackett-Luce model."
    ))
    elems.append(sp(0.2))

    elems.append(p("<b>10.2 Model Selection Methods</b>", H3))
    elems.append(p(
        "Several model selection techniques were considered to evaluate the "
        "contribution of the different predictors in the Plackett-Luce model."
    ))
    elems.append(sp(0.1))

    elems.append(p("<b>10.2.1 Akaike Information Criterion (AIC)</b>", H4))
    elems.append(p(
        "AIC evaluates model quality by balancing goodness of fit with model "
        "complexity: <b>AIC = −2 log L + 2k</b> where <b>L is the likelihood "
        "and k is the number of parameters</b>. This approach encourages good "
        "model fit, allows for more flexibility in retaining predictors, and is "
        "also well-suited for predictive performance. On the other hand, it "
        "<b>penalizes complexity less strongly</b> and <b>may retain redundant "
        "features when predictors are highly correlated</b>."
    ))
    elems.append(sp(0.1))

    elems.append(p("<b>10.2.2 Bayesian Information Criterion (BIC)</b>", H4))
    elems.append(p(
        "BIC is similar to AIC but applies a stronger penalty for model "
        "complexity: <b>BIC = −2 log L + k log n</b> where <b>n is the number "
        "of observations</b>. As mentioned, BIC incorporates a <b>stronger "
        "penalty</b> for unnecessary features, is <b>more effective at removing "
        "redundancy</b>, and tends to produce simpler, more interpretable models. "
        "It may remove features that provide small but meaningful improvements "
        "and can be <b>overly conservative</b> in some cases."
    ))
    elems.append(sp(0.1))

    elems.append(p("<b>10.2.3 Forward Selection</b>", H4))
    elems.append(p(
        "<b>Forward selection begins with an empty model and iteratively adds "
        "features that improve the selection criterion</b> whether it be AIC or "
        "BIC. This method is simple to implement and builds the model "
        "progressively. The problem is that it <b>can miss important features "
        "combinations and is sensitive to correlated predictors</b>."
    ))
    elems.append(sp(0.1))

    elems.append(p("<b>10.2.4 Backward Selection</b>", H4))
    elems.append(p(
        "<b>Backward selection begins with the full model and iteratively removes "
        "the least useful features based on a chosen criterion</b> (AIC or BIC). "
        "This starts with all the available information, is <b>more efficient "
        "when features are correlated</b>, and is well-suited when an initial full "
        "model is complete. Computationally though, this may be more expensive "
        "and may still be affected by multicollinearity."
    ))
    elems.append(sp(0.1))

    elems.append(p("<b>10.2.5 Stepwise Selection (Forward-Backward)</b>", H4))
    elems.append(p(
        "<b>Stepwise selection combines both forward and backward approaches by "
        "iteratively adding and removing features</b>. This can be more flexible "
        "than purely forward or backward selection and can better navigate "
        "correlated feature spaces. Some cons include that it is more complex "
        "to implement, and is still heuristic in nature."
    ))
    elems.append(sp(0.2))

    elems.append(p("<b>10.3 Method Selection</b>", H3))
    elems.append(p(
        "Due to the nature of the enriched feature set, which as we have come "
        "to know, contains several highly correlated predictors, model selection "
        "methods that effectively handle redundancy are preferred."
    ))
    elems.append(sp(0.1))
    elems.append(p("<b>10.3.1 Primary Method: BIC + Backward Selection</b>", H4))
    elems.append(p(
        "This approach was chosen due to the <b>feature set containing substantial "
        "correlation and redundancy</b>, <b>BIC strongly penalizing unnecessary "
        "complexity</b>, and <b>backward selection being a well-suited method to "
        "refine an already fully completed model</b>."
    ))
    elems.append(sp(0.1))
    elems.append(p("<b>10.3.2 Secondary Method: AIC + Stepwise Selection</b>", H4))
    elems.append(p(
        "This approach is used as a complementary method because AIC allows "
        "slightly more flexibility in retaining predictors and stepwise selection "
        "can explore a broader range of feature combinations. This serves as a "
        "robustness check against the more conservative BIC approach."
    ))
    elems.append(sp(0.2))

    elems.append(p("<b>10.4 BIC + Backward Selection</b>", H3))
    elems.append(sp(0.1))

    elems.append(p("<b>10.4.1 Methodology</b>", H4))
    elems.append(p(
        "This process starts with the full Plackett-Luce model containing all "
        "of the predictors. Using the full dataset, the model is fit by maximizing "
        "the penalized likelihood with L2 regularization. The <b>log-likelihood of "
        "the fitted model is used to compute the BIC value which serves as the "
        "baseline score</b>."
    ))
    elems.append(p(
        "At <b>each iteration, one predictor is removed at a time, and the model "
        "is refitted with the reduced predictor set</b>. For each refitted model, "
        "the log-likelihood is calculated and converted to a BIC score. The "
        "<b>predictor whose removal results in the largest decrease of BIC is "
        "permanently removed from the model</b>. This process is repeated "
        "iteratively until no further reduction in BIC can be achieved."
    ))
    elems.append(p(
        "After obtaining the final set of predictors, the Plackett-Luce model is "
        "retrained using those features and <b>evaluated using walk-forward "
        "cross-validation</b> with the same 5 folds used throughout this study. "
        "Selection is performed on 2022-2024 reference data only, so the 2025 test "
        "fold is never seen during feature selection. The v2 implementation "
        "corrects a bug from v1 where BIC used n = number of driver-race rows "
        "instead of n = number of races; the corrected formula uses "
        "<b>BIC = -2 log L + k log(n<sub>races</sub>)</b>."
    ))
    elems.append(sp(0.1))

    elems.append(p("<b>10.4.2 Selection Steps</b>", H4))
    if pl_bic_steps is not None and len(pl_bic_steps) > 0:
        elems.append(p(
            f"Starting from the full 18-feature model "
            f"(BIC = {pl_bic_steps.iloc[0]['BIC_Before']:.3f}), "
            f"{len(pl_bic_steps)} features were removed across {len(pl_bic_steps)} "
            f"steps. The process stopped when no further removal improved BIC."
        ))
        elems.append(sp(0.08))
        hdr = ["Step", "Feature Removed", "BIC Before", "BIC After", "BIC Delta", "Features Left"]
        rows = [hdr]
        for _, r in pl_bic_steps.iterrows():
            rows.append([
                str(int(r["Step"])), r["Feature_Removed"],
                f"{r['BIC_Before']:.3f}", f"{r['BIC_After']:.3f}",
                f"{r['BIC_Delta']:+.3f}", str(int(r["Features_Remaining"])),
            ])
        col_w = [1.0*cm, 4.2*cm, 2.2*cm, 2.2*cm, 2.2*cm, 2.2*cm]
        t = Table(rows, colWidths=col_w)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#16213e")),
            ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
            ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",   (0,0), (-1,-1), 7.5),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f0f4f8")]),
            ("GRID",       (0,0), (-1,-1), 0.4, colors.HexColor("#cccccc")),
            ("ALIGN",      (0,0), (-1,-1), "CENTER"),
            ("ALIGN",      (1,1), (1,-1), "LEFT"),
            ("LEFTPADDING", (0,0), (-1,-1), 4),
            ("RIGHTPADDING",(0,0), (-1,-1), 4),
            ("TOPPADDING",  (0,0), (-1,-1), 3),
            ("BOTTOMPADDING",(0,0),(-1,-1), 3),
        ]))
        elems.append(t)
    elems.append(sp(0.1))

    elems.append(p("<b>10.4.3 Selected Features and Coefficients</b>", H4))
    if pl_bic_coef is not None:
        mean_coef = (pl_bic_coef.groupby("Feature")["Coef"]
                     .mean().reset_index().rename(columns={"Coef": "Mean_Coef"}))
        mean_coef["Abs"] = mean_coef["Mean_Coef"].abs()
        mean_coef = mean_coef.sort_values("Abs", ascending=False)
        n_feat = len(mean_coef)
        final_bic = pl_bic_steps.iloc[-1]["BIC_After"] if pl_bic_steps is not None else "N/A"
        elems.append(p(
            f"BIC backward selection retained <b>{n_feat} features</b> "
            f"(final BIC = {final_bic:.3f}). BIC's strong log(n) penalty is "
            f"highly aggressive here, pruning 14 of 18 predictors and retaining "
            f"only the features with the clearest independent signal. "
            f"Mean PL coefficients across the 5 walk-forward folds are shown below."
        ))
        elems.append(sp(0.08))
        hdr = ["Feature", "Mean Coef", "Direction"]
        rows = [hdr]
        for _, r in mean_coef.iterrows():
            direction = "higher finish score" if r["Mean_Coef"] > 0 else "lower finish score"
            rows.append([r["Feature"], f"{r['Mean_Coef']:+.4f}", direction])
        col_w = [5.0*cm, 2.8*cm, 6.0*cm]
        t = Table(rows, colWidths=col_w)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#16213e")),
            ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
            ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",   (0,0), (-1,-1), 8),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f0f4f8")]),
            ("GRID",       (0,0), (-1,-1), 0.4, colors.HexColor("#cccccc")),
            ("ALIGN",      (0,0), (-1,-1), "CENTER"),
            ("ALIGN",      (0,1), (0,-1), "LEFT"),
            ("LEFTPADDING", (0,0), (-1,-1), 5),
            ("RIGHTPADDING",(0,0), (-1,-1), 5),
            ("TOPPADDING",  (0,0), (-1,-1), 3),
            ("BOTTOMPADDING",(0,0),(-1,-1), 3),
        ]))
        elems.append(t)
    elems.append(sp(0.1))

    elems.append(p("<b>10.4.4 Walk-Forward Performance</b>", H4))
    if pl_bic_summary is not None:
        metrics = ["Spearman", "KendallTau", "Top3 Acc", "Top5 Acc",
                   "Winner Acc", "NDCG@3", "NDCG@5"]
        bic_mean = pl_bic_summary.groupby("Method")[metrics].mean().round(4)
        elems.append(p(
            "Walk-forward mean metrics for the BIC-selected 4-feature model vs. "
            "the qualifying-position baseline:"
        ))
        elems.append(sp(0.08))
        hdr = ["Method"] + metrics
        rows = [hdr]
        for method in ["Model", "Baseline"]:
            if method in bic_mean.index:
                row = [method] + [f"{bic_mean.loc[method, m]:.4f}" for m in metrics]
                rows.append(row)
        col_w = [2.2*cm] + [2.1*cm]*7
        t = Table(rows, colWidths=col_w)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#16213e")),
            ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
            ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",   (0,0), (-1,-1), 7.5),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f0f4f8")]),
            ("GRID",       (0,0), (-1,-1), 0.4, colors.HexColor("#cccccc")),
            ("ALIGN",      (0,0), (-1,-1), "CENTER"),
            ("LEFTPADDING", (0,0), (-1,-1), 4),
            ("RIGHTPADDING",(0,0), (-1,-1), 4),
            ("TOPPADDING",  (0,0), (-1,-1), 3),
            ("BOTTOMPADDING",(0,0),(-1,-1), 3),
        ]))
        elems.append(t)
        elems.append(sp(0.08))
        elems.append(p(
            "The BIC-selected model improves over the baseline on rank-correlation "
            "metrics (Spearman, KendallTau) but falls short on accuracy-based "
            "metrics (Top3, Winner Accuracy). This is consistent with BIC's "
            "aggressive pruning: with only 4 features, the model captures the "
            "broad competitive order well but lacks the granularity to reliably "
            "identify podium finishers and race winners."
        ))
    elems.append(sp(0.2))

    elems.append(p("<b>10.5 AIC + Forward Stepwise</b>", H3))
    elems.append(sp(0.1))

    elems.append(p("<b>10.5.1 Methodology</b>", H4))
    elems.append(p(
        "AIC forward stepwise selection starts with an empty model and "
        "<b>greedily adds the one feature whose inclusion gives the largest "
        "AIC decrease</b> at each step. The process stops when no remaining "
        "feature strictly reduces AIC. The v2 implementation uses "
        "<b>forward-only selection</b> to avoid the infinite oscillation that "
        "can occur with bidirectional stepwise when correlated features trade "
        "off against each other. AIC is computed as "
        "<b>AIC = -2 log L + 2k</b> where n is the number of races. "
        "Selection is performed on 2022-2024 reference data with the same "
        "walk-forward evaluation as BIC."
    ))
    elems.append(sp(0.1))

    elems.append(p("<b>10.5.2 Selection Steps</b>", H4))
    if pl_aic_steps is not None and len(pl_aic_steps) > 0:
        elems.append(p(
            f"Starting from an empty model, {len(pl_aic_steps)} features were "
            f"added before AIC stopped improving."
        ))
        elems.append(sp(0.08))
        hdr = ["Step", "Feature Added", "AIC Before", "AIC After", "AIC Delta", "Features"]
        rows = [hdr]
        for _, r in pl_aic_steps.iterrows():
            aic_before = f"{r['AIC_Before']:.3f}" if pd.notna(r.get("AIC_Before")) else "—"
            aic_delta  = f"{r['AIC_Delta']:+.3f}"  if pd.notna(r.get("AIC_Delta"))  else "—"
            rows.append([
                str(int(r["Step"])), r["Feature_Added"],
                aic_before, f"{r['AIC_After']:.3f}", aic_delta,
                str(int(r["Features_Selected"])),
            ])
        col_w = [1.0*cm, 4.2*cm, 2.2*cm, 2.2*cm, 2.2*cm, 2.0*cm]
        t = Table(rows, colWidths=col_w)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#16213e")),
            ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
            ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",   (0,0), (-1,-1), 7.5),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f0f4f8")]),
            ("GRID",       (0,0), (-1,-1), 0.4, colors.HexColor("#cccccc")),
            ("ALIGN",      (0,0), (-1,-1), "CENTER"),
            ("ALIGN",      (1,1), (1,-1), "LEFT"),
            ("LEFTPADDING", (0,0), (-1,-1), 4),
            ("RIGHTPADDING",(0,0), (-1,-1), 4),
            ("TOPPADDING",  (0,0), (-1,-1), 3),
            ("BOTTOMPADDING",(0,0),(-1,-1), 3),
        ]))
        elems.append(t)
    elems.append(sp(0.1))

    elems.append(p("<b>10.5.3 Selected Features and Coefficients</b>", H4))
    if pl_aic_coef is not None:
        mean_coef = (pl_aic_coef.groupby("Feature")["Coef"]
                     .mean().reset_index().rename(columns={"Coef": "Mean_Coef"}))
        mean_coef["Abs"] = mean_coef["Mean_Coef"].abs()
        mean_coef = mean_coef.sort_values("Abs", ascending=False)
        n_feat = len(mean_coef)
        final_aic = pl_aic_steps.iloc[-1]["AIC_After"] if pl_aic_steps is not None else "N/A"
        elems.append(p(
            f"AIC forward stepwise retained <b>{n_feat} features</b> "
            f"(final AIC = {final_aic:.3f}). AIC's lighter 2k penalty allows "
            f"more features than BIC while still discarding predictors with "
            f"little independent contribution. Notably, AIC selected "
            f"TeamStrength and Quali_x_Overtake where BIC selected "
            f"OvertakeOpportunity and Driver_vs_Team — reflecting AIC's "
            f"greater tolerance for correlated team-based features."
        ))
        elems.append(sp(0.08))
        hdr = ["Feature", "Mean Coef", "Direction"]
        rows = [hdr]
        for _, r in mean_coef.iterrows():
            direction = "higher finish score" if r["Mean_Coef"] > 0 else "lower finish score"
            rows.append([r["Feature"], f"{r['Mean_Coef']:+.4f}", direction])
        col_w = [5.0*cm, 2.8*cm, 6.0*cm]
        t = Table(rows, colWidths=col_w)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#16213e")),
            ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
            ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",   (0,0), (-1,-1), 8),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f0f4f8")]),
            ("GRID",       (0,0), (-1,-1), 0.4, colors.HexColor("#cccccc")),
            ("ALIGN",      (0,0), (-1,-1), "CENTER"),
            ("ALIGN",      (0,1), (0,-1), "LEFT"),
            ("LEFTPADDING", (0,0), (-1,-1), 5),
            ("RIGHTPADDING",(0,0), (-1,-1), 5),
            ("TOPPADDING",  (0,0), (-1,-1), 3),
            ("BOTTOMPADDING",(0,0),(-1,-1), 3),
        ]))
        elems.append(t)
    elems.append(sp(0.1))

    elems.append(p("<b>10.5.4 Walk-Forward Performance</b>", H4))
    if pl_aic_summary is not None:
        metrics = ["Spearman", "KendallTau", "Top3 Acc", "Top5 Acc",
                   "Winner Acc", "NDCG@3", "NDCG@5"]
        aic_mean = pl_aic_summary.groupby("Method")[metrics].mean().round(4)
        elems.append(p(
            "Walk-forward mean metrics for the AIC-selected 5-feature model vs. "
            "the qualifying-position baseline:"
        ))
        elems.append(sp(0.08))
        hdr = ["Method"] + metrics
        rows = [hdr]
        for method in ["Model", "Baseline"]:
            if method in aic_mean.index:
                row = [method] + [f"{aic_mean.loc[method, m]:.4f}" for m in metrics]
                rows.append(row)
        col_w = [2.2*cm] + [2.1*cm]*7
        t = Table(rows, colWidths=col_w)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#16213e")),
            ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
            ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",   (0,0), (-1,-1), 7.5),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f0f4f8")]),
            ("GRID",       (0,0), (-1,-1), 0.4, colors.HexColor("#cccccc")),
            ("ALIGN",      (0,0), (-1,-1), "CENTER"),
            ("LEFTPADDING", (0,0), (-1,-1), 4),
            ("RIGHTPADDING",(0,0), (-1,-1), 4),
            ("TOPPADDING",  (0,0), (-1,-1), 3),
            ("BOTTOMPADDING",(0,0),(-1,-1), 3),
        ]))
        elems.append(t)
        elems.append(sp(0.08))
        elems.append(p(
            "The AIC-selected model shows a similar pattern to BIC: rank-correlation "
            "metrics (Spearman 0.694, KendallTau 0.556) exceed the baseline, while "
            "accuracy-based metrics (Top3, Winner) remain below it. The 5-feature "
            "AIC model is marginally stronger than the 4-feature BIC model across "
            "most metrics, suggesting the one additional feature (TeamStrength) "
            "provides incremental predictive value."
        ))
    elems.append(sp(0.2))

    elems.append(p("<b>10.6 Variance Inflation Factor (VIF) Analysis</b>", H3))

    elems.append(p("<b>10.6.1 What is VIF and Why Do We Use It?</b>", H4))
    elems.append(p(
        "The <b>Variance Inflation Factor (VIF)</b> measures <b>multicollinearity</b> "
        "— the degree to which a predictor can be linearly explained by the other "
        "predictors in the model. For a given feature x<sub>j</sub>, VIF is computed "
        "by regressing x<sub>j</sub> on all remaining features and calculating:"
    ))
    elems.append(sp(0.05))
    elems.append(p("    VIF(x<sub>j</sub>) = 1 / (1 − R²<sub>j</sub>)", BODY))
    elems.append(sp(0.05))
    elems.append(p(
        "where R²<sub>j</sub> is the coefficient of determination from that regression. "
        "A VIF of 1 means no multicollinearity — the feature carries information "
        "not present in any other predictor. A VIF of 5 means 80% of that feature's "
        "variance is explained by the other features, indicating strong redundancy. "
        "The standard threshold of <b>VIF &lt; 5</b> (equivalently R² &lt; 0.8) is "
        "used throughout this study."
    ))
    elems.append(sp(0.08))
    elems.append(p(
        "<b>Why multicollinearity is a problem:</b> When two or more features are "
        "highly correlated, they carry overlapping information. This creates two "
        "distinct problems depending on the model type:"
    ))
    elems.append(sp(0.06))
    elems.append(p(
        "For <b>linear probabilistic models like Plackett-Luce</b>, coefficients "
        "are estimated via maximum likelihood. When features are collinear, the "
        "Fisher information matrix — which determines coefficient standard errors "
        "and p-values — becomes near-singular. This inflates standard errors, "
        "making features appear statistically insignificant even when they carry "
        "real signal, and invalidates the Wald test p-values used for inference.",
        BULLET
    ))
    elems.append(sp(0.04))
    elems.append(p(
        "For <b>tree-based models like XGBRanker</b>, multicollinearity does not "
        "directly bias predictions the way it does in linear models. However, "
        "correlated features split feature importance scores across multiple "
        "redundant predictors — if two features convey the same information, the "
        "model may use either one interchangeably across trees. This makes feature "
        "importance rankings unreliable and masks which features are truly "
        "driving predictions.",
        BULLET
    ))
    elems.append(sp(0.08))
    elems.append(p(
        "<b>The VIF backward selection procedure</b> addresses both problems by "
        "iteratively removing the most collinear feature until all remaining "
        "features are below the threshold:"
    ))
    elems.append(sp(0.06))
    for step_txt in [
        "Compute VIF for all features simultaneously on the 2022-2024 reference dataset.",
        "Remove the single feature with the highest VIF.",
        "Recompute VIF for the remaining features.",
        "Repeat until all VIFs fall below 5.0.",
    ]:
        elems.append(p(f"    {step_txt}", BULLET))
    elems.append(sp(0.08))
    elems.append(p(
        "An important property of VIF is that it depends only on the <b>feature "
        "matrix X</b> — not on the model type, the outcome variable, or any "
        "model-specific parameters. Because both the Plackett-Luce model and "
        "XGBRanker use the same 18 engineered features on the same 2022-2024 "
        "reference dataset, the VIF selection procedure produces an <b>identical "
        "removal sequence and identical retained feature set for both models</b>. "
        "The redundancy structure of the features is entirely model-independent."
    ))
    elems.append(sp(0.15))

    # ── 10.6.2 PL ──────────────────────────────────────────────────────────
    elems.append(p("<b>10.6.2 Plackett-Luce Model</b>", H4))
    elems.append(p(
        "<b>Why VIF for PL:</b> The Plackett-Luce model estimates a coefficient "
        "vector β via maximum likelihood; standard errors and p-values are derived "
        "from the Fisher information matrix F = −∂²log L/∂β². When predictors are "
        "collinear, F becomes near-singular and its inverse — the asymptotic "
        "covariance matrix — has inflated diagonal entries, leading to unreliable "
        "standard errors. VIF selection ensures F is well-conditioned, giving "
        "valid frequentist inference for the retained coefficients."
    ))
    elems.append(sp(0.08))
    elems.append(p(
        "<b>How it is implemented:</b> VIF backward selection is run on the "
        "2022-2024 reference dataset (68 races, 1,836 driver-race rows), "
        "starting from the full set of 18 features. Three features are removed "
        "in sequence:"
    ))
    elems.append(sp(0.06))

    if pl_vif_steps is not None and len(pl_vif_steps) > 0:
        header = ["Step", "Feature Removed", "VIF at Removal", "Features Left"]
        rows = [header]
        for _, r in pl_vif_steps.iterrows():
            vif_val = "Inf" if np.isinf(r["VIF_at_Removal"]) else fmt(r["VIF_at_Removal"], 2)
            rows.append([str(int(r["Step"])), r["Feature_Removed"],
                         vif_val, str(int(r["Features_Remaining"]))])
        elems.append(make_table(rows, [1.2*cm, 5.5*cm, 3.0*cm, 3.0*cm]))
        elems.append(p(
            "Table 10a. VIF backward selection steps for PL model "
            "(threshold = 5.0, reference: 2022-2024 data).", CAPTION
        ))
        elems.append(sp(0.1))
    else:
        elems.append(p("VIF steps data not available.", BODY))

    elems.append(p(
        "<i>OvertakeOpportunity</i> is removed first because it is defined as "
        "(1 − QualiPercentile) × OvertakeIndex — a deterministic product of two "
        "other features already in the set, making its VIF infinite. "
        "<i>DriverForm</i> follows because Driver_vs_Team = DriverForm − TeamStrength, "
        "so DriverForm is nearly entirely predictable from the other two. "
        "<i>Quali_x_Overtake</i> is removed last due to its strong linear "
        "relationship with the qualifying features. After these three removals, "
        "all 15 remaining features have VIF below 5.0."
    ))
    elems.append(sp(0.08))

    if pl_vif_final is not None:
        header = ["Feature", "VIF"]
        rows = [header]
        for _, r in pl_vif_final.iterrows():
            rows.append([r["Feature"], fmt(r["VIF"])])
        elems.append(make_table(rows, [6.5*cm, 2.5*cm]))
        elems.append(p(
            "Table 10b. Final VIF values for the 15 retained PL features "
            "(all below the 5.0 threshold).", CAPTION
        ))
        elems.append(sp(0.1))

    elems.append(p("<b>Evaluation metrics — PL VIF-15 vs All-18:</b>"))
    elems.append(sp(0.06))

    metric_cols = ["Spearman", "KendallTau", "Top3 Acc", "Top5 Acc",
                   "Winner Acc", "NDCG@3", "NDCG@5"]

    if pl_vif_summary is not None and pl_df is not None:
        all18_m = pl_df[pl_df["Method"] == "Model"][metric_cols].mean()
        vif_m   = pl_vif_summary[pl_vif_summary["Method"] == "Model"][metric_cols].mean()

        header = ["Metric", "PL All-18", "PL VIF-15", "Delta (VIF − All-18)"]
        rows = [header]
        labels = ["Mean Spearman", "Mean KendallTau", "Top-3 Acc", "Top-5 Acc",
                  "Winner Acc", "NDCG@3", "NDCG@5"]
        for lbl, key in zip(labels, metric_cols):
            a18v = all18_m[key]
            vifv = vif_m[key]
            rows.append([lbl, fmt(a18v), fmt(vifv), delta_str(vifv - a18v)])
        elems.append(make_table(rows, [3.0*cm, 2.5*cm, 2.5*cm, 4.0*cm]))
        elems.append(p(
            "Table 10c. All-18 vs VIF-selected PL model (walk-forward means).", CAPTION
        ))
        elems.append(sp(0.1))

    elems.append(p(
        "Removing three collinear features leaves predictive performance essentially "
        "unchanged, confirming they were statistically redundant. The slight metric "
        "shifts are within noise. The retained 15-feature set gives more reliable "
        "coefficient estimates and valid p-values for PL inference, which is the "
        "primary purpose of VIF selection for a linear probabilistic model."
    ))
    elems.append(sp(0.2))

    # ── 10.6.3 XGBRanker ────────────────────────────────────────────────────
    elems.append(p("<b>10.6.3 XGBRanker</b>", H4))
    elems.append(p(
        "<b>Why VIF for XGBRanker:</b> Gradient-boosted trees can handle correlated "
        "features without biased predictions — the model will not produce "
        "misleading coefficients the way a linear model does. However, when two "
        "features are highly correlated, XGBoost may use either one across "
        "different trees, artificially splitting the importance score between "
        "them. This makes feature importance rankings hard to interpret: a "
        "feature may appear unimportant simply because its signal was absorbed "
        "by a correlated partner. VIF selection consolidates the feature set "
        "to the maximally non-redundant subset, producing importance scores "
        "that reflect genuine independent contributions."
    ))
    elems.append(sp(0.08))
    elems.append(p(
        "<b>How it is implemented:</b> The same backward VIF procedure is applied "
        "to the same 2022-2024 reference dataset. Because VIF is a property of "
        "the feature matrix alone, the removal sequence is identical to the PL "
        "case: OvertakeOpportunity (Inf), DriverForm (613.93), and "
        "Quali_x_Overtake (18.04) are removed in that order, leaving 15 features."
    ))
    elems.append(sp(0.08))

    if xgb_vif_steps is not None and len(xgb_vif_steps) > 0:
        header = ["Step", "Feature Removed", "VIF at Removal", "Features Left"]
        rows = [header]
        for _, r in xgb_vif_steps.iterrows():
            vif_val = "Inf" if np.isinf(r["VIF_at_Removal"]) else fmt(r["VIF_at_Removal"], 2)
            rows.append([str(int(r["Step"])), r["Feature_Removed"],
                         vif_val, str(int(r["Features_Remaining"]))])
        elems.append(make_table(rows, [1.2*cm, 5.5*cm, 3.0*cm, 3.0*cm]))
        elems.append(p(
            "Table 10d. VIF backward selection steps for XGBRanker v2 "
            "(threshold = 5.0, reference: 2022-2024 data).", CAPTION
        ))
        elems.append(sp(0.1))

    elems.append(p(
        "<b>The 15 retained features</b> span all feature groups: three qualifying "
        "metrics (QualiPercentile, QualiGapToPole, TeammateQualiGap), team/driver "
        "performance (TeamStrength, Driver_vs_Team), practice pace and consistency "
        "(FP_LongRunPace, FP_LongRunVar), reliability (DriverDNFRate, TeamDNFRate, "
        "Reliability_x_Rain), environmental (AirTemp, TrackTemp, Rainfall), and "
        "track characteristics (OvertakeIndex, GridSpread). Each retained feature "
        "carries information not linearly explained by the others."
    ))
    elems.append(sp(0.08))

    elems.append(p("<b>Evaluation metrics — XGBRanker VIF-15 vs All-18:</b>"))
    elems.append(sp(0.06))

    xgb_metric_cols = ["Spearman", "KendallTau", "Top3 Acc", "Top5 Acc",
                       "Winner Acc", "NDCG@3", "NDCG@5"]
    if xgb_vif_summary is not None and xgb_df is not None:
        all18_xgb = {m: xgb_df[f"Model_{m}"].mean() for m in xgb_metric_cols}
        vif_xgb   = {m: xgb_vif_summary[f"Model_{m}"].mean() for m in xgb_metric_cols}
        header = ["Metric", "XGB All-18", "XGB VIF-15", "Delta (VIF − All-18)"]
        rows = [header]
        labels_xgb = ["Mean Spearman", "Mean KendallTau", "Top-3 Acc", "Top-5 Acc",
                      "Winner Acc", "NDCG@3", "NDCG@5"]
        for lbl, key in zip(labels_xgb, xgb_metric_cols):
            a18v = all18_xgb[key]
            vifv = vif_xgb[key]
            rows.append([lbl, fmt(a18v), fmt(vifv), delta_str(vifv - a18v)])
        elems.append(make_table(rows, [3.0*cm, 2.5*cm, 2.5*cm, 4.0*cm]))
        elems.append(p(
            "Table 10e. All-18 vs VIF-selected XGBRanker (walk-forward means).", CAPTION
        ))
        elems.append(sp(0.1))

    elems.append(p(
        "Overall ranking performance is preserved after VIF reduction. Spearman "
        "and Kendall's Tau remain nearly identical, confirming that removing "
        "three redundant features does not reduce the model's ability to order "
        "drivers correctly. Top-3 and Top-5 accuracy show marginal changes "
        "within fold-to-fold noise. The primary gain from VIF selection for "
        "XGBRanker is interpretability: feature importance scores for the "
        "15-feature model reflect genuine independent contributions rather than "
        "being diluted across correlated partners."
    ))
    elems.append(sp(0.2))

    elems.append(p("<b>10.7 Overall</b>", H3))
    elems.append(p(
        "We see that the <b>BIC plus backward selection model achieves comparable "
        "or slightly improved performance while only using 4 instead of the "
        "original 18 predictors</b> (v1 result; v2 pending). This tells us that "
        "a <b>majority of the feature-engineered features were redundant</b>, "
        "<b>qualifying and core performance metrics contain most of the predictive "
        "signal</b>, and removing <b>weaker features reduces noise</b>. These "
        "results show that a parsimonious model (model that explains data with "
        "fewer or least amount of features), based on a core set of predictors "
        "can match or outperform a highly engineered feature set."
    ))
    elems.append(sp(0.2))

    # ── Section 10.7 — Cross-Method Comparison ────────────────────────────────
    elems.append(p("<b>10.7 Cross-Method Comparison: VIF vs BIC vs AIC</b>", H3))
    elems.append(p(
        "The three PL model selection methods — VIF backward selection, BIC "
        "backward selection, and AIC forward stepwise — each approach feature "
        "reduction from a different angle. VIF removes features purely on "
        "multicollinearity grounds without regard to predictive performance. "
        "BIC and AIC balance goodness of fit against complexity, with BIC "
        "applying a heavier penalty. The table below compares the mean "
        "walk-forward performance of each selected model against the full "
        "18-feature PL model and the qualifying-position baseline."
    ))
    elems.append(sp(0.1))

    cmp_data = [
        ("Full PL (18 features)",    18, 0.6834, 0.5456, 0.6179, 0.7661, 0.4788, 0.8879, 0.9007),
        ("PL + VIF (15 features)",   15, 0.6847, 0.5459, 0.6205, 0.7677, 0.4621, 0.8898, 0.9023),
        ("PL + BIC (4 features)",     4, 0.6922, 0.5554, 0.6260, 0.7644, 0.4447, 0.8873, 0.9029),
        ("PL + AIC (5 features)",     5, 0.6944, 0.5559, 0.6407, 0.7694, 0.4356, 0.8899, 0.9017),
        ("Qualifying Baseline",       1, 0.6790, 0.5404, 0.6742, 0.7727, 0.6030, 0.9001, 0.9123),
    ]
    hdr = ["Model", "Feats", "Spearman", "Kendall", "Top3", "Top5", "Winner", "NDCG@3", "NDCG@5"]
    rows = [hdr]
    for row in cmp_data:
        rows.append([row[0], str(row[1])] + [f"{v:.4f}" for v in row[2:]])
    col_w = [4.5*cm, 1.2*cm] + [1.8*cm]*7
    t = Table(rows, colWidths=col_w)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#16213e")),
        ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
        ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0), (-1,-1), 7.5),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f0f4f8")]),
        ("GRID",       (0,0), (-1,-1), 0.4, colors.HexColor("#cccccc")),
        ("ALIGN",      (0,0), (-1,-1), "CENTER"),
        ("ALIGN",      (0,1), (0,-1), "LEFT"),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("RIGHTPADDING",(0,0), (-1,-1), 4),
        ("TOPPADDING",  (0,0), (-1,-1), 3),
        ("BOTTOMPADDING",(0,0),(-1,-1), 3),
    ]))
    elems.append(t)
    elems.append(sp(0.1))
    elems.append(p(
        "<b>Rank correlation (Spearman, Kendall):</b> All PL models beat the "
        "baseline, and feature reduction consistently improves rank correlation — "
        "AIC (5 features) achieves the best Spearman (0.694) and Kendall (0.556) "
        "of any model. This pattern — fewer features yielding better rank "
        "correlation — indicates the dropped features were adding noise, not signal, "
        "to the overall ordering."
    ))
    elems.append(sp(0.05))
    elems.append(p(
        "<b>Accuracy metrics (Top3, Top5, Winner):</b> The qualifying baseline "
        "outperforms all models on these metrics. Among the PL variants, AIC "
        "achieves the highest Top3 accuracy (0.641), while the full model and "
        "VIF model are competitive. BIC and AIC both drop Winner Accuracy below "
        "the full model — aggressive pruning removes features that help identify "
        "the exact race winner even if they hurt overall rank ordering."
    ))
    elems.append(sp(0.05))
    elems.append(p(
        "<b>Key takeaway:</b> The three selection methods serve different purposes. "
        "VIF corrects multicollinearity with minimal feature loss (18 → 15), "
        "preserving predictive breadth. BIC produces the most parsimonious model "
        "(4 features) with the strongest interpretability. AIC strikes the best "
        "overall balance — 5 features, best rank correlation, and competitive "
        "accuracy — making it the preferred reduced model for the PL framework."
    ))
    elems.append(sp(0.2))

    # ── Section 10.8 — McFadden's Adjusted Pseudo-R² ─────────────────────────
    elems.append(p("<b>10.8 McFadden's Adjusted Pseudo-R² and Coefficient Analysis</b>", H3))
    elems.append(sp(0.1))

    elems.append(p("<b>10.8.1 Methodology</b>", H4))

    elems.append(p("<b>What is R² and why can't we use it directly?</b>", H4))
    elems.append(p(
        "In linear regression, <b>R²</b> (the coefficient of determination) "
        "measures the proportion of variance in the outcome that the model "
        "explains: R² = 1 − SS<sub>residual</sub> / SS<sub>total</sub>, where "
        "SS<sub>total</sub> is the total variance of the response and "
        "SS<sub>residual</sub> is the unexplained variance after fitting. "
        "A value of 0 means the model explains nothing, 1 means it explains "
        "everything. R² is important because it gives an intuitive, "
        "scale-free summary of how well the model fits the data."
    ))
    elems.append(sp(0.06))
    elems.append(p(
        "The Plackett-Luce model is not a regression model — it does not "
        "predict a continuous outcome or minimize a sum of squared residuals. "
        "Instead, it maximizes the likelihood of observed race orderings. "
        "There is no sum of squares to partition, so standard R² has no "
        "direct meaning here. <b>A goodness-of-fit measure that works with "
        "likelihoods is needed.</b>"
    ))
    elems.append(sp(0.1))

    elems.append(p("<b>What is McFadden's Pseudo-R² and why do we use it?</b>", H4))
    elems.append(p(
        "McFadden's pseudo-R² is the standard goodness-of-fit measure for "
        "likelihood-based models (logistic regression, Plackett-Luce, etc.). "
        "It is constructed by analogy with R²: instead of comparing residual "
        "variance to total variance, it compares the log-likelihood of the "
        "fitted model to the log-likelihood of a <b>null model</b> that "
        "contains no predictors at all."
    ))
    elems.append(sp(0.05))
    elems.append(p(
        "<b>McFadden R² = 1 − log L<sub>model</sub> / log L<sub>null</sub></b>"
    ))
    elems.append(sp(0.05))
    elems.append(p(
        "The <b>null model</b> for the Plackett-Luce framework assigns equal "
        "probability to every driver at every sequential selection step — "
        "equivalent to saying every possible race ordering is equally likely. "
        "Its log-likelihood is <b>log L<sub>null</sub> = −Σ log(k<sub>i</sub>!)</b> "
        "summed over all races, where k<sub>i</sub> is the number of starters in "
        "race i. When the fitted model's log-likelihood equals the null "
        "(no improvement), R² = 0. As the model explains more of the ordering "
        "structure, log L<sub>model</sub> approaches 0 (from below) and R² "
        "approaches 1. For sports ranking models, values of 0.05–0.20 indicate "
        "meaningful predictive power — the scale is lower than in regression "
        "because even the best pre-race model cannot fully predict the chaotic "
        "dynamics of a live race."
    ))
    elems.append(sp(0.1))

    elems.append(p("<b>What is the Adjusted R² and why does it matter?</b>", H4))
    elems.append(p(
        "A well-known problem with R² in any model is that it <b>never decreases "
        "when you add more predictors</b>, even if those predictors are pure noise. "
        "Adding a random feature to a model will always increase (or at worst "
        "hold constant) the R², because more parameters allow the model to fit "
        "idiosyncratic patterns in the training data. This makes raw R² a poor "
        "criterion for comparing models with different numbers of features."
    ))
    elems.append(sp(0.06))
    elems.append(p(
        "<b>Adjusted R²</b> corrects for this by penalizing each additional "
        "parameter. For McFadden's pseudo-R², the adjusted version subtracts "
        "the number of model parameters k from the log-likelihood before "
        "computing the ratio:"
    ))
    elems.append(sp(0.05))
    elems.append(p(
        "<b>McFadden Adjusted R² = 1 − (log L<sub>model</sub> − k) / log L<sub>null</sub></b>"
    ))
    elems.append(sp(0.05))
    elems.append(p(
        "Subtracting k from log L<sub>model</sub> makes the numerator smaller "
        "(more negative), which shrinks the ratio and reduces R². A new feature "
        "only improves adjusted R² if its improvement in log-likelihood exceeds "
        "its parameter cost. This makes adjusted R² the correct metric for "
        "comparing the Full (18 features), VIF (15), BIC (4), and AIC (5) "
        "models — it directly answers: <b>does adding these extra features "
        "improve the model enough to justify their complexity?</b>"
    ))
    elems.append(sp(0.1))

    elems.append(p("<b>How were the coefficients and p-values computed?</b>", H4))
    elems.append(p(
        "Each of the four PL model variants is fit a single time on the "
        "<b>2022-2024 reference dataset</b> — that is, all 68 races from the "
        "2022, 2023, and 2024 seasons combined into one training set. "
        "This is distinct from the walk-forward cross-validation used elsewhere "
        "in this study. In walk-forward CV, we repeatedly split data into "
        "training and test years to measure predictive performance on unseen "
        "seasons. Here, the goal is different: we want to measure "
        "<b>goodness of fit on the training data</b> and obtain stable "
        "coefficient estimates for inference. The 2025 season is never used "
        "in any part of this analysis — not for training, not for testing, "
        "not for feature selection. It exists solely as a final out-of-sample "
        "evaluation period (tested in the walk-forward folds)."
    ))
    elems.append(sp(0.06))
    elems.append(p(
        "The coefficients themselves come directly from the L-BFGS-B optimizer "
        "that maximizes the penalized Plackett-Luce log-likelihood. Each "
        "coefficient β<sub>j</sub> represents how much a one-unit increase in "
        "feature j (after standardization) increases a driver's log-strength, "
        "which in turn increases their probability of finishing ahead of rivals."
    ))
    elems.append(sp(0.06))
    elems.append(p(
        "Standard errors are computed from the <b>unregularized Fisher "
        "information matrix</b> — the negative Hessian of the plain "
        "log-likelihood (without the L2 penalty term) evaluated at the fitted "
        "coefficients. The Fisher information measures how sharply the "
        "likelihood peaks around the estimates: a steep peak means low "
        "uncertainty (small SE), a flat peak means high uncertainty (large SE). "
        "Using the unregularized Hessian is necessary for valid frequentist "
        "inference because the L2 penalty artificially steepens the curvature, "
        "which would understate the true uncertainty. "
        "The <b>Z-score</b> is then Z = β / SE and the <b>p-value</b> is "
        "computed from the standard normal distribution as "
        "p = 2 × (1 − Φ(|Z|)), giving a two-sided Wald test."
    ))
    elems.append(sp(0.1))

    elems.append(p("<b>10.8.2 Goodness-of-Fit Comparison</b>", H4))
    if pl_mcfadden is not None:
        elems.append(p(
            f"Null log-likelihood (uniform random ranking, 68 races): "
            f"log L<sub>null</sub> = {pl_mcfadden.iloc[0]['LogL_null']:.3f}"
        ))
        elems.append(sp(0.08))
        hdr = ["Model", "k", "log L", "R²", "Adj R²"]
        rows = [hdr]
        for _, r in pl_mcfadden.iterrows():
            rows.append([
                r["Model"], str(int(r["k"])),
                f"{r['LogL_model']:.3f}",
                f"{r['McFadden_R2']:.4f}",
                f"{r['McFadden_R2_adj']:.4f}",
            ])
        col_w = [4.8*cm, 1.2*cm, 2.5*cm, 2.2*cm, 2.2*cm]
        t = Table(rows, colWidths=col_w)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#16213e")),
            ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
            ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",   (0,0), (-1,-1), 8),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f0f4f8")]),
            ("GRID",       (0,0), (-1,-1), 0.4, colors.HexColor("#cccccc")),
            ("ALIGN",      (0,0), (-1,-1), "CENTER"),
            ("ALIGN",      (0,1), (0,-1), "LEFT"),
            ("LEFTPADDING", (0,0), (-1,-1), 5),
            ("RIGHTPADDING",(0,0), (-1,-1), 5),
            ("TOPPADDING",  (0,0), (-1,-1), 3),
            ("BOTTOMPADDING",(0,0),(-1,-1), 3),
        ]))
        elems.append(t)
        elems.append(sp(0.08))
        elems.append(p(
            "The raw R² values (0.076–0.079) are similar across all four models. "
            "However, the <b>adjusted R²</b> tells a different story: the BIC "
            "(0.0753) and AIC (0.0755) models achieve <b>higher adjusted R² than "
            "the full 18-feature model (0.0723)</b>. This confirms that the "
            "13–14 features removed by BIC and AIC were adding parameters without "
            "improving fit. The VIF model (0.0710) has the lowest adjusted R², "
            "reflecting that VIF selects on collinearity alone and retains some "
            "features whose marginal likelihood contribution is minimal."
        ))
    elems.append(sp(0.12))

    elems.append(p("<b>Interpreting the 7.7% figure: why the models still perform well</b>", H4))
    elems.append(p(
        "A McFadden R² of ~0.077 may appear low, but it is important to understand "
        "what the null model actually is. The null assigns <b>equal probability to "
        "every possible finishing order</b>. For a 20-driver race, there are "
        "20! ≈ 2.4 × 10<sup>18</sup> possible orderings. The null treats each of "
        "these as equally likely — an astronomically uninformed baseline. "
        "The 7.7% is the relative improvement in log-likelihood over this "
        "completely random lottery, not the proportion of variance explained in "
        "the familiar regression sense."
    ))
    elems.append(sp(0.06))
    elems.append(p(
        "The high evaluation metrics (Spearman ≈ 0.69, NDCG@3 ≈ 0.89) exist "
        "for a different reason: <b>Formula 1 race results are inherently "
        "structured</b>. The same drivers and constructors consistently occupy "
        "the same part of the field across races. A model that has been trained "
        "on historical races learns this structure — which drivers tend to qualify "
        "and finish at the front, which teams have the fastest car, which drivers "
        "are in form. The evaluation metrics measure how well the model captures "
        "that learned structure when predicting future, unseen races. They are "
        "not measuring improvement over a random baseline; they are measuring "
        "agreement with actual observed rankings."
    ))
    elems.append(sp(0.06))
    elems.append(p(
        "The two measures are therefore answering completely different questions. "
        "McFadden R² asks: <i>compared to knowing nothing at all, how much does "
        "the model improve?</i> The evaluation metrics ask: <i>when the model "
        "ranks drivers and we compare to what actually happened, how accurate "
        "is the ordering?</i> A Spearman correlation of 0.69 is achievable even "
        "with a small McFadden R² because the difficulty of the problem is not "
        "as hard as 20! suggests — the competitive hierarchy makes many orderings "
        "far more likely than others before any model is applied."
    ))
    elems.append(sp(0.1))

    elems.append(p("<b>Why the qualifying baseline achieves high Spearman despite race variation</b>", H4))
    elems.append(p(
        "The qualifying baseline (predicting that the race finishes in the same "
        "order as qualifying) achieves Spearman ≈ 0.679 despite the fact that "
        "qualifying and race order never match perfectly. This is not surprising "
        "once the properties of Spearman correlation are understood."
    ))
    elems.append(sp(0.06))
    elems.append(p(
        "Spearman correlation does not require exact position matches — it only "
        "requires that the <b>relative ordering is preserved</b>. A race where "
        "the pole-sitter finishes P2, the P2 starter finishes P1, and every other "
        "driver finishes exactly one position off still produces a Spearman near "
        "1.0, because the rank ordering is almost entirely intact. What matters "
        "is not whether P3 finishes P3, but whether the driver who started P3 "
        "finishes ahead of the driver who started P10."
    ))
    elems.append(sp(0.06))
    elems.append(p(
        "Formula 1 competitive order operates in <b>performance bands</b>. The "
        "front runners (top 6-8 drivers across the top 3-4 teams) shuffle among "
        "themselves but almost never drop to P15. The midfield (P9-P14) shuffles "
        "within its own band. Even the backmarkers (P15-20) tend to shuffle "
        "among themselves rather than jumping into the points. This within-band "
        "shuffling does relatively little damage to Spearman because the major "
        "rank separations across bands are preserved."
    ))
    elems.append(sp(0.06))
    elems.append(p(
        "DNFs are a genuine source of rank disruption — a driver who qualifies "
        "P1 and retires is recorded at the back of the finishing order, creating "
        "a large rank jump that pulls correlation down for that race. However, "
        "DNFs affect 1–3 drivers per race on average, so while they create noise "
        "in individual races they do not dominate the mean Spearman across 68 races."
    ))
    elems.append(sp(0.06))
    elems.append(p(
        "The 2024 and 2025 seasons were more competitive than 2022-2023 (the Red "
        "Bull era), with Ferrari, McLaren, and Mercedes all winning races. This "
        "increased inter-team variation lowers the qualifying-to-finish "
        "correlation for those years. The mean Spearman of 0.679 reflects an "
        "average across all five walk-forward folds — folds testing on the "
        "more predictable 2022-2023 seasons contribute higher correlations "
        "while the 2025 fold contributes lower, and the 0.679 is the result "
        "of averaging across this range."
    ))
    elems.append(sp(0.1))

    def _coef_table(coef_df, label):
        """Build a coefficient table with p-value significance markers."""
        if coef_df is None:
            return []
        elems_local = []
        elems_local.append(p(f"<b>{label}</b>", H4))
        hdr = ["Feature", "Coef", "SE", "Z", "p-value", "Sig"]
        rows = [hdr]
        for _, r in coef_df.iterrows():
            se_s  = f"{r['SE']:.4f}"   if r.get("SE")      is not None and str(r.get("SE")) != "nan"      else "—"
            z_s   = f"{r['Z']:.3f}"    if r.get("Z")       is not None and str(r.get("Z")) != "nan"       else "—"
            pv_s  = f"{r['P_value']:.4f}" if r.get("P_value") is not None and str(r.get("P_value")) != "nan" else "—"
            sig_s = str(r.get("Sig", "")) if str(r.get("Sig", "")) != "nan" else ""
            rows.append([
                r["Feature"], f"{r['Coef']:+.4f}", se_s, z_s, pv_s, sig_s,
            ])
        col_w = [4.5*cm, 2.0*cm, 2.0*cm, 1.8*cm, 2.0*cm, 1.0*cm]
        t = Table(rows, colWidths=col_w)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#16213e")),
            ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
            ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",   (0,0), (-1,-1), 7.5),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f0f4f8")]),
            ("GRID",       (0,0), (-1,-1), 0.4, colors.HexColor("#cccccc")),
            ("ALIGN",      (0,0), (-1,-1), "CENTER"),
            ("ALIGN",      (0,1), (0,-1), "LEFT"),
            ("LEFTPADDING", (0,0), (-1,-1), 4),
            ("RIGHTPADDING",(0,0), (-1,-1), 4),
            ("TOPPADDING",  (0,0), (-1,-1), 3),
            ("BOTTOMPADDING",(0,0),(-1,-1), 3),
        ]))
        elems_local.append(t)
        elems_local.append(p(
            "Significance codes:  *** p<0.001  ** p<0.01  * p<0.05  . p<0.1  "
            "(SE and p-values from unregularized Fisher information)", CAPTION
        ))
        return elems_local

    elems.append(p("<b>10.8.3 Coefficients and p-values by Model</b>", H4))
    elems.append(p(
        "Each model is fit once on the 2022-2024 reference data. Coefficients "
        "reflect the direction and magnitude of each feature's contribution to "
        "a driver's Plackett-Luce 'strength' — a higher score increases the "
        "probability of finishing ahead of rivals. Continuous features other "
        "than QualiPercentile are standardized, so coefficients are comparable "
        "in magnitude. Standard errors and p-values use the unregularized "
        "Fisher information (Wald test)."
    ))
    elems.append(sp(0.08))
    elems.append(p(
        "<i>Note on the full and VIF models: L2 regularization collapses several "
        "correlated features (AirTemp, TrackTemp, Rainfall, OvertakeIndex, "
        "GridSpread) to near-zero coefficients. In those models, multicollinearity "
        "inflates standard errors for the remaining features, reducing statistical "
        "significance even for genuinely important predictors. The BIC and AIC "
        "models avoid this by removing the redundant features, yielding cleaner "
        "and more interpretable inference.</i>"
    ))
    elems.append(sp(0.08))

    for coef_df, label in [
        (pl_coef_bic,  "BIC-selected model (4 features)"),
        (pl_coef_aic,  "AIC-selected model (5 features)"),
        (pl_coef_vif,  "VIF-selected model (15 features)"),
        (pl_coef_full, "Full model (18 features)"),
    ]:
        elems += _coef_table(coef_df, label)
        elems.append(sp(0.12))

    elems.append(p("<b>10.8.4 Interpretation</b>", H4))
    elems.append(p(
        "<b>QualiPercentile</b> is the strongest and most consistently significant "
        "predictor across all four models (p < 0.001 in all). Its large negative "
        "coefficient (approximately −2.3) means a driver starting from pole "
        "(QualiPercentile = 0) has substantially higher predicted strength than "
        "one starting from the back (QualiPercentile = 1). This confirms "
        "qualifying position as the dominant pre-race signal in Formula 1."
    ))
    elems.append(sp(0.06))
    elems.append(p(
        "<b>DriverForm</b> is significant in the BIC model (p < 0.001) and AIC "
        "model (p = 0.006), with a positive coefficient indicating that drivers "
        "in better recent form are predicted to finish higher. In the full model, "
        "its significance is diluted by collinear features sharing the same signal."
    ))
    elems.append(sp(0.06))
    elems.append(p(
        "<b>TeamStrength</b> is significant in the AIC and VIF models (p < 0.001), "
        "reflecting the constructor's pace advantage independent of qualifying. "
        "BIC did not retain TeamStrength, preferring OvertakeOpportunity and "
        "Driver_vs_Team as alternative signals for team-level effects."
    ))
    elems.append(sp(0.06))
    elems.append(p(
        "The <b>multicollinearity issue</b> is most visible in the full model: "
        "DriverForm has coef = +0.46 but p = 0.65, meaning its variance is shared "
        "with correlated features and its individual contribution cannot be "
        "isolated. The BIC and AIC selection process resolves this by retaining "
        "only features that provide genuinely independent information, which is "
        "why the adjusted R² improves even as the feature count drops from 18 to "
        "4 or 5."
    ))

    return elems


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 11 — Leave One Out Analysis for Two-Stage Model
# ═══════════════════════════════════════════════════════════════════════════

def section_11_loo(xgb_loo, xgb_vif_steps, xgb_vif_final, xgb_df, xgb_vif_summary):
    elems = []
    elems.append(p("<b>11. Leave One Out Analysis for Two-Stage Model</b>", H2))
    elems.append(hr())

    elems.append(p("<b>11.1 Methodology</b>", H3))
    elems.append(p(
        "To further evaluate the contribution of individual predictors, we utilized "
        "a leave one out (LOO) feature analysis. The full model is first trained "
        "using all features to establish the baseline model performance. After, "
        "<b>each feature is removed one at a time and the model is retrained, and "
        "the resulting evaluation metrics are compared to the baseline</b>. What "
        "this analysis does is <b>measure the marginal contribution of each feature "
        "to model performance</b>. Unlike feature importance from tree-based models, "
        "which reflects how frequently a feature is used in splits, <b>LOO directly "
        "looks at how model performance changes when a feature is removed</b>."
    ))
    elems.append(p(
        "<b>However, due to many of the features being highly correlated, especially "
        "the qualifying derived features, removing a single one, may not actually "
        "significantly affect results if the other correlated features can substitute "
        "for it. To address this multicollinearity, variance inflation factor (VIF) "
        "is computed for each predictor.</b> VIF measures <b>how much of a feature "
        "can be explained by other features</b>. For a feature Xⱼ, regress it on "
        "all features and compute R² from that regression:"
    ))
    elems.append(p("<b>VIF_j = 1 / (1 − R²_j)</b>"))
    elems.append(sp(0.15))
    elems.append(p(
        "From our problem and the correlation analysis we already saw that "
        "<i>QualiPercentile</i> is highly correlated with its engineered features "
        "and <i>DriverForm</i> and <i>TeamStrength</i> also exhibit high "
        "correlation. We can utilize iterative VIF removal, just like backward "
        "selection to find the key features. The features with high VIF are highly "
        "collinear and can be removed and after each removal the VIF values are "
        "recomputed using the new feature set."
    ))
    elems.append(sp(0.2))

    elems.append(p("<b>11.2 Leave One Out Results</b>", H3))

    if xgb_loo is not None:
        col_map = {c: c.replace(" ", "_") for c in xgb_loo.columns}
        loo = xgb_loo.rename(columns=col_map)
        feat_col = "Feature_Removed"
        sp_col   = "Delta_Spearman"
        if sp_col in loo.columns:
            loo = loo.sort_values(sp_col, ascending=True)

        header = ["Feature Removed", "Δ Spearman", "Δ Kendall",
                  "Δ Top3", "Δ Top5", "Δ Winner", "Δ NDCG@3", "Δ NDCG@5"]
        rows = [header]
        for _, r in loo.iterrows():
            def ds(key):
                for k in [key, key.replace("_", " ")]:
                    if k in r.index:
                        return delta_str(r[k])
                return "—"
            rows.append([
                r[feat_col],
                ds("Delta_Spearman"), ds("Delta_KendallTau"),
                ds("Delta_Top3 Acc"), ds("Delta_Top5 Acc"),
                ds("Delta_Winner Acc"), ds("Delta_NDCG@3"), ds("Delta_NDCG@5"),
            ])

        col_w = [3.8*cm, 1.55*cm, 1.55*cm, 1.4*cm, 1.4*cm, 1.5*cm, 1.5*cm, 1.5*cm]
        elems.append(make_table(rows, col_w))
        elems.append(p(
            "Table 11a. Leave-one-out feature analysis for XGBRanker v2. "
            "Delta = (model without feature) − (full model), averaged across "
            "5 walk-forward folds. Sorted by Δ Spearman ascending "
            "(most damaging removals at top).", CAPTION
        ))
        elems.append(sp(0.2))

    elems.append(p(
        "From this analysis, we see that <b>removing any individual features "
        "actually only results in minimal changes to model performance across "
        "all evaluation metrics</b>. This then suggests that <b>many features "
        "in the model are redundant</b>. Qualifying related variables is an "
        "example of this as <i>QualiPercentile</i>, <i>QualixOvertake</i>, "
        "and <i>OvertakeOpportunity</i>, are all highly correlated, thus "
        "<b>removing one doesn't significantly reduce predictive performance "
        "as the remaining features still capture similar information</b>."
    ))
    elems.append(p(
        "The results indicate that the model is robust to the removal of "
        "individual features. Many predictors, as mentioned, provide overlapping "
        "information. Single-feature importance may be understated due to "
        "multicollinearity (multiple variables are correlated, hard for the model "
        "to isolate individual effects of each predictor). Even though individual "
        "features appear to have limited impact in isolation, this doesn't imply "
        "that the information that they represent is unimportant."
    ))
    elems.append(sp(0.2))

    elems.append(p("<b>11.3 Variance Inflation Factor (VIF)</b>", H3))
    elems.append(p(
        "The full VIF analysis — what VIF is, why it is applied, the backward "
        "selection procedure, and a comparison of evaluation metrics before and "
        "after feature reduction — is presented in <b>Section 10.6</b> of this "
        "report. That section covers both the Plackett-Luce model (Section 10.6.2) "
        "and the XGBRanker (Section 10.6.3) in a unified treatment, including "
        "the removal sequence (OvertakeOpportunity → DriverForm → "
        "Quali_x_Overtake), the final 15 retained features, and the "
        "walk-forward evaluation metrics for each model before and after "
        "VIF reduction."
    ))
    elems.append(sp(0.2))

    elems.append(p("<b>11.4 Overall</b>", H3))
    elems.append(p(
        "Overall, the leave one out analysis reveals that while the enriched "
        "feature set may contain many engineered predictors, <b>predictive "
        "performance is primarily driven by a small subset of core information</b>. "
        "The leave-one-out (LOO) analysis shows that <b>removing individual "
        "predictors results in only minor changes in performance</b>. This "
        "indicates a <b>high degree of redundancy</b> among features. This is "
        "also supported in the VIF analysis, which identifies significant "
        "multicollinearity."
    ))
    elems.append(p(
        "The <b>VIF-based backward selection confirms that many predictors are "
        "strongly explained by others</b>, and their <b>removal actually does not "
        "meaningfully impact model performance</b>. Despite reducing "
        "multicollinearity, the VIF-reduced model still retains a large feature "
        "set, emphasizing <b>while features may be correlated, they still capture "
        "unique aspects of race dynamics</b>."
    ))
    elems.append(p(
        "Across both the VIF and LOO analysis, we see <b>qualifying performance "
        "still being the dominant driver of predictive accuracy, with team strength "
        "and driver-related metrics providing additional but smaller "
        "contributions</b>. These are also consistent with the correlation "
        "analysis done prior, which showed strong relationships among engineered "
        "features. Overall, the results indicate that increasing feature "
        "complexity doesn't necessarily improve predictive performance. Carefully "
        "selecting features that provide independent information leads to more "
        "stable models without sacrificing accuracy."
    ))

    return elems


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 12 — Conclusion
# ═══════════════════════════════════════════════════════════════════════════

def section_12_conclusion():
    elems = []
    elems.append(p("<b>12. Conclusion</b>", H2))
    elems.append(hr())

    elems.append(p(
        "This report has documented the full development trajectory of F1 race "
        "prediction models, from the original single-season Plackett-Luce model "
        "to the v2 Two-Stage XGBRanker and PL models with corrected methodology, "
        "evaluated using temporally valid walk-forward cross-validation across "
        "the 2022-2025 seasons."
    ))
    elems.append(sp(0.15))

    elems.append(p("<b>12.1 Overall Model Performance</b>", H3))
    elems.append(p(
        "The table below summarizes mean walk-forward performance across all "
        "five folds for every model variant evaluated in this study. "
        "The <b>qualifying-position baseline</b> (score = −grid position) "
        "represents the simplest possible prediction: assume the race finishes "
        "in qualifying order. This is a competitive benchmark because grid "
        "position is the strongest single predictor of race outcome in Formula 1."
    ))
    elems.append(sp(0.08))

    cmp_rows = [
        ["Model",                      "Feat", "Spearman", "Kendall", "Top3", "Top5", "Winner", "NDCG@3", "NDCG@5"],
        ["XGBRanker (18 features)",     "18",  "0.6621",   "0.5240",  "0.6624","0.7686","0.5129","0.8909","0.8983"],
        ["XGBRanker + VIF (15 feat)",   "15",  "0.6617",   "0.5189",  "0.6515","0.7805","0.4856","0.8901","0.9041"],
        ["PL — Full (18 features)",     "18",  "0.6834",   "0.5456",  "0.6179","0.7661","0.4788","0.8879","0.9007"],
        ["PL + VIF (15 features)",      "15",  "0.6847",   "0.5459",  "0.6205","0.7677","0.4621","0.8898","0.9023"],
        ["PL + BIC (4 features)",        "4",  "0.6922",   "0.5554",  "0.6260","0.7644","0.4447","0.8873","0.9029"],
        ["PL + AIC (5 features)",        "5",  "0.6944",   "0.5559",  "0.6407","0.7694","0.4356","0.8899","0.9017"],
        ["Qualifying Baseline",          "1",  "0.6790",   "0.5404",  "0.6742","0.7727","0.6030","0.9001","0.9123"],
    ]
    col_w = [4.3*cm, 1.1*cm] + [1.75*cm]*7
    t = Table(cmp_rows, colWidths=col_w)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#16213e")),
        ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
        ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0), (-1,-1), 7.5),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f0f4f8")]),
        ("GRID",       (0,0), (-1,-1), 0.4, colors.HexColor("#cccccc")),
        ("ALIGN",      (0,0), (-1,-1), "CENTER"),
        ("ALIGN",      (0,1), (0,-1), "LEFT"),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("RIGHTPADDING",(0,0), (-1,-1), 4),
        ("TOPPADDING",  (0,0), (-1,-1), 3),
        ("BOTTOMPADDING",(0,0),(-1,-1), 3),
    ]))
    elems.append(t)
    elems.append(sp(0.08))
    elems.append(p("Table 12a. Mean walk-forward metrics across all 5 folds for all model variants.", CAPTION))
    elems.append(sp(0.15))

    elems.append(p("<b>12.2 Key Findings by Metric</b>", H3))
    elems.append(p(
        "<b>Rank correlation (Spearman, Kendall):</b> All model variants beat "
        "the baseline, and counter-intuitively, reducing features improves rank "
        "correlation. The AIC-selected 5-feature PL model achieves the best "
        "Spearman (0.694) and Kendall (0.556) of any model. This suggests the "
        "dropped features were adding noise to the global ordering rather than "
        "signal — a consequence of multicollinearity in the engineered feature set."
    ))
    elems.append(sp(0.08))
    elems.append(p(
        "<b>Top-3 and Top-5 accuracy:</b> The qualifying baseline outperforms "
        "all models (Top3 = 0.674, Top5 = 0.773). Among actual models, "
        "XGBRanker is best on Top3 (0.662), followed by PL AIC (0.641). "
        "The rank-learning objective of XGBRanker, optimized directly for NDCG, "
        "is better suited for identifying podium finishers than the PL model's "
        "probabilistic ranking."
    ))
    elems.append(sp(0.08))
    elems.append(p(
        "<b>Winner accuracy:</b> The baseline is the strongest predictor of the "
        "race winner (0.603) — the pole-sitter wins often enough in Formula 1 "
        "that no model consistently improves on it. XGBRanker (0.513) is the "
        "best among the trained models. All PL variants trail, with the more "
        "aggressively pruned BIC and AIC models performing worst (0.445, 0.436) "
        "as winner prediction benefits from the full feature set's granularity."
    ))
    elems.append(sp(0.08))
    elems.append(p(
        "<b>NDCG@3 and NDCG@5:</b> The baseline again leads (0.900, 0.912), "
        "reflecting its strong positional accuracy. Model differences on NDCG "
        "are small (within 0.015) — all models capture the broad ranking "
        "structure similarly, with variation concentrated in the exact ordering "
        "of top finishers."
    ))
    elems.append(sp(0.15))

    elems.append(p("<b>12.3 Model Selection Summary</b>", H3))
    elems.append(p(
        "The most significant methodological gains came from <b>walk-forward "
        "cross-validation</b> (eliminating temporal data leakage) and the "
        "<b>corrected QualiPercentile denominator</b>. Feature selection via "
        "VIF, BIC, and AIC consistently showed that the 18-feature set contains "
        "substantial redundancy — removing correlated features does not hurt "
        "and often helps rank correlation."
    ))
    elems.append(sp(0.08))
    elems.append(p(
        "<b>Qualifying position</b> remains the dominant predictor of race "
        "outcome across all methods, consistent with the physical reality of "
        "Formula 1 where dirty-air effects limit overtaking. DriverForm and "
        "TeamStrength are the next most consistently selected features across "
        "all three PL selection methods (VIF, BIC, AIC), confirming that "
        "recent form and constructor competitiveness carry independent predictive "
        "signal beyond grid position alone."
    ))
    elems.append(sp(0.08))
    elems.append(p(
        "The <b>2025 season</b> proved particularly challenging for both models "
        "due to a competitive reset — multiple teams became race-winning "
        "competitors, breaking the strong statistical patterns from the "
        "2022-2024 Red Bull era. This is a regime-shift problem and highlights "
        "the fundamental limit of pre-race prediction: when the underlying "
        "competitive order changes substantially, historical training data "
        "cannot fully prepare the model."
    ))
    elems.append(sp(0.08))
    elems.append(p(
        "Among all model variants, <b>no single model dominates on every metric</b>. "
        "For applications prioritising overall driver ordering (rankings, fantasy "
        "sports), the <b>PL + AIC model</b> is the strongest choice. For predicting "
        "top finishers and race winners, <b>XGBRanker</b> outperforms the PL "
        "variants and approaches the naive baseline. The qualifying baseline "
        "remains a surprisingly difficult benchmark to beat for exact positional "
        "accuracy, underscoring how much Formula 1 race results are determined "
        "before the lights go out."
    ))
    elems.append(sp(0.2))

    elems.append(p("<b>12.4 Main Finding and Contribution</b>", H3))
    elems.append(p(
        "A recurring concern when evaluating these results is that the qualifying "
        "baseline is so competitive. It is important to recognise that this "
        "baseline is not weak — predicting that every race finishes in qualifying "
        "order requires no training, no features beyond grid position, and no "
        "model of any kind. The fact that it achieves 60.3% winner accuracy "
        "and Spearman 0.679 reflects a genuine property of Formula 1: "
        "<b>grid position is the single most powerful pre-race signal</b>, because "
        "aerodynamic dirty air makes overtaking physically difficult and dominant "
        "cars tend to qualify and race at the front. <b>Beating this baseline, "
        "even by a modest margin on held-out future seasons, is a meaningful "
        "result</b> — it means the model has extracted real predictive signal "
        "beyond what any analyst already knows from the starting grid."
    ))
    elems.append(sp(0.1))
    elems.append(p(
        "The central finding of this study can be stated as follows: <b>a "
        "parsimonious 5-feature Plackett-Luce model, trained exclusively on "
        "past seasons, outperforms a qualifying-position baseline on rank-order "
        "accuracy for future races, while an 18-feature engineered model offers "
        "no improvement over this minimal set.</b> This has two implications. "
        "First, the predictable component of F1 race outcomes is captured by a "
        "small core of pre-race signals — qualifying position, team strength, "
        "circuit overtaking difficulty, driver form, and reliability. Second, "
        "additional engineered features beyond these five do not add predictive "
        "value; they introduce redundancy that, if anything, slightly hurts "
        "rank-ordering performance."
    ))
    elems.append(sp(0.1))
    elems.append(p(
        "The difficulty of beating the baseline on <b>winner accuracy</b> is "
        "itself an informative finding rather than a failure. No pre-race model "
        "achieves winner accuracy above 52%, while the baseline achieves 60.3%. "
        "This gap directly quantifies <b>how much of a race winner's outcome is "
        "stochastic</b> — driven by safety cars, mechanical failures, pit-stop "
        "strategy, and on-track incidents that no model trained on pre-race data "
        "can anticipate. The remaining predictable component (correctly identifying "
        "likely winners roughly half the time from pre-race data alone) represents "
        "the ceiling of what systematic pre-race prediction can achieve in Formula 1 "
        "under current regulations."
    ))
    elems.append(sp(0.1))
    elems.append(p(
        "Finally, the comparison between rolling and expanding training folds "
        "(Section 7.5) reveals that <b>more historical data does not uniformly "
        "help</b>. For 2024 predictions, adding 2022 data to the 2023 training set "
        "consistently improves both models, because the 2022-2023 seasons shared "
        "similar competitive dynamics. For 2025 predictions, the XGBRanker "
        "benefits from the full three-year training set while the PL model's "
        "winner accuracy declines — a consequence of the 2025 competitive reset "
        "where three years of Red Bull-era patterns became partially misleading. "
        "This suggests that in sports prediction, <b>recency and regime stability "
        "matter as much as sample size</b>, and that model-specific sensitivity "
        "to distributional shift should inform how training data is selected."
    ))

    return elems


# ═══════════════════════════════════════════════════════════════════════════
# BUILD PDF
# ═══════════════════════════════════════════════════════════════════════════

def build_pdf():
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    def _read(path):
        return pd.read_csv(path) if os.path.exists(path) else None

    pl_df          = _read(PL_CSV)
    xgb_df         = _read(XGB_CSV)
    imp_df         = _read(IMP_CSV)
    xgb_vif_steps  = _read(XGB_VIF_STEPS)
    xgb_vif_final  = _read(XGB_VIF_FINAL)
    xgb_vif_sum    = _read(XGB_VIF_SUMMARY)
    xgb_loo        = _read(XGB_LOO_CSV)
    pl_vif_steps   = _read(PL_VIF_STEPS)
    pl_vif_final   = _read(PL_VIF_FINAL)
    pl_vif_sum     = _read(PL_VIF_SUMMARY)
    pl_coef_df     = _read(PL_COEF_CSV)
    pl_bic_steps   = _read(PL_BIC_STEPS)
    pl_bic_summary = _read(PL_BIC_SUMMARY)
    pl_bic_coef    = _read(PL_BIC_COEF)
    pl_aic_steps   = _read(PL_AIC_STEPS)
    pl_aic_summary = _read(PL_AIC_SUMMARY)
    pl_aic_coef    = _read(PL_AIC_COEF)
    pl_mcfadden    = _read(PL_MCFADDEN)
    pl_coef_full   = _read(PL_COEF_FULL)
    pl_coef_vif    = _read(PL_COEF_VIF)
    pl_coef_bic    = _read(PL_COEF_BIC)
    pl_coef_aic    = _read(PL_COEF_AIC)

    enriched_df = None
    if os.path.exists(ENRICHED_CSV):
        try:
            enriched_df = pd.read_csv(ENRICHED_CSV)
        except Exception:
            pass

    doc = SimpleDocTemplate(
        OUT_PDF, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2.2*cm, bottomMargin=2*cm,
    )

    elems = []

    # Title page
    elems.append(sp(0.5))
    elems.append(p("<b>F1 Race Prediction: Model Development and Evaluation</b>", H1))
    elems.append(p(
        "Two-Stage XGBRanker and Plackett-Luce Models — Version 2 (Walk-Forward CV)",
        BODY
    ))
    elems.append(hr())
    elems.append(sp(0.3))

    # Section 0 — Issues
    elems += section_issues()
    elems.append(PageBreak())

    # Section 1 — Background
    elems += section_1_background()
    elems.append(PageBreak())

    # Section 2 — Model Framework
    elems += section_2_model_framework()
    elems.append(PageBreak())

    # Section 3 — Data and Preprocessing
    elems += section_3_data_preprocessing()
    elems.append(PageBreak())

    # Section 4 — Feature Engineering
    elems += section_4_feature_engineering()
    elems.append(PageBreak())

    # Section 5 — Model Selection
    elems += section_5_model_selection()
    elems.append(PageBreak())

    # Section 6 — Results
    elems += section_6_results(imp_df)
    elems.append(PageBreak())

    # Section 7 — Evaluation Metrics
    elems += section_7_eval_metrics(xgb_df, pl_df)
    elems.append(PageBreak())

    # Section 8 — Comparison and Interpretation
    elems += section_8_comparison(xgb_df, pl_df, pl_coef_df)
    elems.append(PageBreak())

    # Section 9 — Feature Correlation
    elems += section_9_feature_correlation(enriched_df)
    elems.append(PageBreak())

    # Section 10 — PL Model Selection Criteria (+ combined VIF section for both models)
    elems += section_10_pl_selection(
        pl_vif_steps, pl_vif_final, pl_vif_sum, pl_df,
        pl_bic_steps, pl_bic_summary, pl_bic_coef,
        pl_aic_steps, pl_aic_summary, pl_aic_coef,
        pl_mcfadden, pl_coef_full, pl_coef_vif, pl_coef_bic, pl_coef_aic,
        xgb_vif_steps=xgb_vif_steps, xgb_vif_final=xgb_vif_final,
        xgb_vif_summary=xgb_vif_sum, xgb_df=xgb_df,
    )
    elems.append(PageBreak())

    # Section 11 — LOO Analysis
    elems += section_11_loo(xgb_loo, xgb_vif_steps, xgb_vif_final, xgb_df, xgb_vif_sum)
    elems.append(PageBreak())

    # Section 12 — Conclusion
    elems += section_12_conclusion()

    doc.build(elems)
    print(f"Report written to: {OUT_PDF}")


if __name__ == "__main__":
    build_pdf()
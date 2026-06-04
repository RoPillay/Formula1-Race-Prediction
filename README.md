# Formula1-Race-Prediction

*Work in Progress — Ongoing Research Project*
##  Overview
This repository contains an ongoing undergraduate research project focused on statistical and machine learning approaches for Formula 1 race prediction and performance modeling. The project integrates telemetry-derived, qualifying, and race-context features across multiple Formula 1 seasons to model finishing position outcomes, evaluate race strategy, and analyze key performance drivers in stochastic racing environments.
Current work includes regularized Plackett–Luce ranking models, probabilistic outcome modeling, feature selection and multicollinearity analysis, and Monte Carlo race simulation frameworks for race strategy evaluation.

## Research Motivation
Qualifying performance is widely believed to be one of the strongest determinants of Formula 1 race outcomes, yet race-day execution, team strength, and driver form introduce substantial variability. This project aims to quantify these relationships, assess their statistical significance, and build predictive models that move beyond descriptive analysis toward actionable race performance forecasting.

## Key Research Questions 
- How effective is telemetry and race features for predicting Formula 1 race outcomes across multiple seasons?
- What is the relative impact of certain predictors such as qualifying position, driver form, team strength, and track specific factors in determining final race position?
- How do ranking approaches such as Plackett-Luce models compare to tradiitonal predictive modeling frameworks for race outcomes?
- How can stochastic race simulation and probabilistic modeling improve race strategy optimization under uncertainty?

## Current Progress
[Current work](https://docs.google.com/document/d/1vVFXrOgcapsdwYac7s4XwOXEpe7JpVclJWCF34v1KCA/edit?tab=t.0)
- Engineered enriched race-context and qualifying-based features across multiple Formula 1 seasons using FastF1 telemetry and official timing data
- Developed regularized Plackett–Luce ranking models with walk forward cross-validation for race outcome prediction and finishing-position ranking across multiple seasons
- Applied feature selection, VIF analysis, correlation analysis, and exploratory statistical evaluation to understand predictor importance and reduce feature redundancy
- Evaluated model performance using ranking and classification metrics including Spearman correlation, Kendall’s Tau, Top-3 accuracy, and NDCG
- Implemented Monte Carlo race simulations incorporating lap-time variability, tire degradation, pit strategy, and stochastic race events
  [Monte Carlo Current Work](https://docs.google.com/document/d/1ThiAwvgyUiQIdqrt58P2OycU6xiBj53fLNOCT1lKM00/edit?tab=t.5mmjmcy4iqi2#heading=h.u2aa18kntqmi)

## Repository Contents
- `analysis/` — PDFs summarizing exploratory analysis, modeling ideas, and research notes
- `interactive/` — Interactive HTML visualizations examining relationships between qualifying position, driver form, team strength, and finishing position
  - Interactive plots are hosted via GitHub Pages:
  -  [Qualifying vs Finishing Position](https://ropillay.github.io/Formula1-Race-Prediction/interactive/F1_2024_Qual_vs_Finish.html)
  -  [Driver Form vs Finishing Position](https://ropillay.github.io/Formula1-Race-Prediction/interactive/DriverForm_vs_Finish_PerRace.html)
  -  [Team Strength vs Finishing Position](https://ropillay.github.io/Formula1-Race-Prediction/interactive/TeamStrength_vs_Finish_PerRace.html)
- `src/` — Python scripts used for data collection, feature engineering, statistical analysis, and model development

## Data
- Official Formula 1 timing and session data via the **FastF1 API**
- Race results, qualifying data, and race-level context features
- Local caching enabled for reproducibility

## Planned Extensions
- Reinforcement learning/multi-agent dynamics to optimize strategy
- Deep learning models for race outcome prediction
- Integration of richer telemetry and strategy features

## Research Paper
A full academic-style LaTeX manuscript documenting the methodology, analysis, and results will be linked here upon completion.

## Contact
**Rohan Pillay**  
Statistics & Data Science — UC Davis

**Professor Maxime Guiffo Pouokam**  
UC Davis Department of Statistics

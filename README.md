# Formula1-Race-Prediction

*Work in Progress — Ongoing Research Project*
##  Overview
This repository contains an ongoing undergraduate research project focused on statistical and machine learning approaches for Formula 1 race prediction and performance modeling. The project integrates telemetry-derived, qualifying, and race-context features across multiple Formula 1 seasons to model finishing position outcomes, evaluate race strategy, and analyze key performance drivers in stochastic racing environments.
Current work includes regularized Plackett–Luce ranking models, probabilistic outcome modeling, feature selection and multicollinearity analysis, and Monte Carlo race simulation frameworks for race strategy evaluation.

## Research Motivation
Qualifying performance is widely believed to be one of the strongest determinants of Formula 1 race outcomes, yet race-day execution, team strength, and driver form introduce substantial variability. This project aims to quantify these relationships, assess their statistical significance, and build predictive models that move beyond descriptive analysis toward actionable race performance forecasting.

## Key Research Questions 
- How effectivel can telemetry and race features predict Formula 1 race outcomes across multiple seasons?
- What is the relative impact of certain predictors such as qualifying position, driver form, team strength, and track specific factors in determining final race position?
- How do ranking approaches such as Plackett-Luce models compare to tradiitonal predictive modeling frameworks for race outcomes?
- How can stochastic race simulation and probabilistic modeling improve race strategy optimization under uncertainty?

## Current Progress
[Current work](https://docs.google.com/document/d/1vVFXrOgcapsdwYac7s4XwOXEpe7JpVclJWCF34v1KCA/edit?tab=t.0)
- Exploratory data analysis and correlation studies
- Statistical evaluation of key predictors (qualifying position, driver form, team strength)
- Binary logistic regression model to predict top-10 finishes
- L2- regularized Plackett-Luce Model ML model (with PCA exploration)
- Realistic race simulation utilizing monte carlo methods

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

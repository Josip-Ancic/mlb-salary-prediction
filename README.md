# MLB Player Salary Prediction (CRISP-DM)

Regression models that predict a baseball player's salary from his season and career statistics. The use case is a club's management that wants a data-driven reference point when negotiating contracts.

> Course project: **Knowledge Discovery in Data** (Otkrivanje znanja u podacima), FOI, University of Zagreb, April 2026  
> Authors: **Lovro Alfirević & Josip Ančić**

## Result

**Gradient Boosting: R² = 0.728 on the test set (0.738 in 5-fold CV)**, which meets both success criteria set at the start (R² > 0.5 and RMSE below the target's standard deviation).

| Model | Notes |
|---|---|
| Linear / Ridge / Lasso | R² ≈ 0.32, trained on the VIF-reduced feature set |
| Random Forest | R² = 0.635 |
| **Gradient Boosting** | **R² = 0.728** (best) |

Career cumulative stats (`CAtBat`, `CRuns`, `CHits`, `CRBI`) account for **over 75 % of feature importance**, which matches the literature (Magel & Hoffman, 2015).

## What's inside the notebook

Everything follows all six CRISP-DM phases:

1. **Data understanding**: Hitters dataset (322 players, 20 attributes), distributions, correlations, missing values (18.32 % missing `Salary`)
2. **Data preparation**:
   - log transform of the target (skew 1.59 → −0.18) and of skewed features  
   - one-hot encoding  
   - **VIF** to remove multicollinearity  
   - **SelectKBest** and **RFE** for feature selection  
   - **RobustScaler** fitted only on the training set
3. **Modelling**: five regressors, hyperparameter search with `cross_val_score`
4. **Evaluation**: feature importance, Lasso coefficients, residual diagnostics
5. **Deployment demo**: predicted salaries for the 59 players with an unknown salary

## Tech stack

Python · pandas · NumPy · scikit-learn · statsmodels · Matplotlib · seaborn · Jupyter

## Data

*Hitters* dataset (MLB 1986–87, from *An Introduction to Statistical Learning*). Save it as `Projekt_13_hitters.csv` next to the notebook.

Full documentation (in Croatian) is in [`docs/`](docs/).

---

## Hrvatski

# Predviđanje plaća igrača bejzbola (CRISP-DM)

Regresijski modeli koji predviđaju plaću MLB igrača na temelju sezonskih i karijernih statistika. Alat je zamišljen kao referentna točka klubu u pregovorima o ugovorima.

> Projekt iz kolegija **Otkrivanje znanja u podacima**, FOI, travanj 2026. · Autori: **Lovro Alfirević i Josip Ančić**

### Rezultat

Najbolji je **Gradient Boosting s R² = 0,728** na testnom skupu (0,738 na petostrukoj križnoj validaciji). Zadovoljena su oba kriterija uspjeha. Na plaću najjače utječu karijerne kumulativne statistike, koje nose više od 75 % važnosti značajki.

### Postupak

Projekt prolazi svih šest faza CRISP-DM-a:

- razumijevanje i istraživanje podataka
- priprema podataka: log-transformacija, VIF, SelectKBest/RFE, RobustScaler
- pet regresijskih modela
- vrednovanje
- primjena modela na 59 igrača s nepoznatom plaćom

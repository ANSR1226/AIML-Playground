# 🏠 Boston House Price Prediction

Predicting **median house prices in the Boston area** using classical machine learning regression models.  
This project walks from **EDA → preprocessing → model training → evaluation**, and is part of my ML learning journey.

---

## 📂 Project Overview

- **Goal:** Learn and apply supervised regression to predict `MEDV` (Median value of owner‑occupied homes).  
- **Dataset:** Classic *Boston Housing* dataset (506 rows, 13 features + target).[web:577][web:580]  
- **Target:** `MEDV` – median house value (in \$1000s).  

This is a learning‑focused project: I focus on understanding the data, handling outliers, comparing models, and interpreting metrics.

---

## 🧮 Features Used

- **CRIM** – crime rate in the area. Higher value = more crimes per person in that town.
- **ZN** – percentage of land with big residential plots (large houses on big lots). 
- **INDUS** – percentage of land used for factories/industry (non‑retail business). 
- **CHAS** – is the area next to the Charles River? 1 = yes, 0 = no. 
- **NOX** – air pollution level (nitric oxide concentration). 
- **RM** – average number of rooms per house in the area. 
- **AGE** – percentage of houses that are old (built long ago, owner‑occupied). 
- **DIS** – average distance to major job centers (how far people must travel to work).
- **RAD** – rating of how easy it is to reach main highways (higher = better access).
- **TAX** – property tax rate in that area (per 10,000 dollars of value). 
- **PTRATIO** – number of students per teacher in local schools (pupil–teacher ratio). 
- **LSTAT** – percentage of people in the area with lower income or social status. 
- **MEDV** – target column: median value of houses in the area (in thousands of dollars).  

(Sensitive feature `B` is dropped for ethical reasons.)

---

## 🔁 Workflow

1. **Exploratory Data Analysis (EDA)**
   - Histograms, boxplots, scatterplots vs `MEDV`
   - Correlation heatmap to see which features matter most

2. **Data Preprocessing**
   - Train–test split
   - Outlier detection (boxplot / IQR) on skewed features (`CRIM`, etc.)
   - Optional log transforms on heavily right‑skewed predictors
   - Feature scaling with `StandardScaler` inside a `ColumnTransformer`

3. **Models Trained**
   - `RandomForestRegressor`

4. **Evaluation Metrics**
   - **MAE** (Mean Absolute Error)  
   - **MSE / RMSE** (Mean / Root Mean Squared Error)  
   - **R² score** (how much variance in `MEDV` is explained)

Best model (on my current setup):

*RandomForestRegressor with R² ≈ 0.89 and MAE ≈ 2.0 on the test set*  

***

# Food Demand Forecasting with Random Forest

In this project, I predict weekly food demand (`num_orders`) for a food delivery company based on historical orders, meal properties, fulfillment center attributes, and promotions. My goal is to help optimize inventory and reduce waste by forecasting how many orders each meal–center–week combination will receive.

***

## 1. Problem Overview

I am given:

- **Train data**: historical orders with features and target `num_orders`.
- **Test data**: future weeks with the same features but without `num_orders`.
- **Lookup tables**:
  - `fulfilment_center_info.csv` (center details)
  - `meal_info.csv` (meal details)
- **Sample submission**: template with `id` and `num_orders` for generating predictions.

**Task:** I build a regression model that predicts `num_orders` for each row in the test set.

***

## 2. Data Description

After merging train with the lookup tables, each row I work with represents a **(week, center_id, meal_id)** combination with:

- **id**: Row identifier (not used as a feature).
- **week**: Week index in the timeline (captures time trend).
- **center_id**: Fulfilment center ID (joins to center info).
- **meal_id**: Meal ID (joins to meal info).
- **checkout_price**: Final price paid by the customer.
- **base_price**: Original price before discounts.
- **emailer_for_promotion**: 1 if promoted via email, else 0.
- **homepage_featured**: 1 if featured on homepage, else 0.
- **category**: Meal category (beverages, snacks, etc.).
- **cuisine**: Cuisine type (e.g., Indian, Italian).
- **city_code**: Encoded city ID of the center.
- **region_code**: Encoded region ID.
- **center_type**: Type of center (capacity/scale).
- **op_area**: Operational area of the center.
- **num_orders**: Target – number of orders for that meal–center–week.

***

## 3. My Approach

### 3.1 Preprocessing & Feature Engineering

1. **Merge datasets**
   - I join `train` with `fulfilment_center_info` on `center_id`.
   - I join `train` with `meal_info` on `meal_id`.
   - I apply the same merges to `test`.

2. **Column handling**

   - **Drop**:
     - `id` (surrogate key, no predictive signal).

   - **Keep numeric as-is** (no scaling needed for trees):
     - `week`, `checkout_price`, `base_price`, `op_area`,
       `emailer_for_promotion`, `homepage_featured`,
       `city_code`, `region_code`.

   - **Derived features I engineer**:
     - `discount = base_price - checkout_price`
     - `weeks_since_start = week - week.min()`

   - **Label-encode high-cardinality / categorical columns** (one encoder per column):
     - `center_id`, `meal_id`, `category`, `cuisine`, `center_type`.

3. **Train/validation split**

   - From `train_full` (with `num_orders`), I split:
     - 80% → training set
     - 20% → validation set
   - I use `train_test_split(test_size=0.2, random_state=42)`.

### 3.2 Model

- **Algorithm**: `RandomForestRegressor` (tree-based ensemble).
- **Why I chose it**: It handles nonlinear relationships, interactions, and mixed numeric/categorical (encoded) features well — and requires no feature scaling.
- **Hyperparameters I tuned to**:
  - `n_estimators = 300`
  - `max_depth = None`
  - `min_samples_split = 2`
  - `n_jobs = -1`
  - `random_state = 42`

### 3.3 Pipeline

I use a `Pipeline` with a `ColumnTransformer` to keep preprocessing and modeling together cleanly:
```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

feature_cols = num_col + obj_col + encode_col + no_change  # excludes 'num_orders'

preprocess = ColumnTransformer(
    transformers=[
        ("pass", "passthrough", feature_cols)
    ],
    remainder="drop"
)

rfr = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    max_depth=None,
    min_samples_split=2
)

model = Pipeline(steps=[
    ("pre", preprocess),
    ("rfr", rfr)
])
```

***

## 4. Evaluation

On my 20% validation split, the model achieved:

- **Mean Absolute Error (MAE)**: `≈ 67.96`
- **Mean Squared Error (MSE)**: `≈ 20,585.43`
- **Root Mean Squared Error (RMSE)**: `≈ 143.48`
- **R² score**: `≈ 0.865`

An R² of ~0.86 tells me the model explains about 86% of the variance in `num_orders`, which I consider strong performance for a real-world demand forecasting problem.

***

## 5. Generating Predictions & Submission

1. **Retrain on full training data**

   After validating, I retrain `model` on all rows of `train_full`:
```python
X_full = train_full[feature_cols]
y_full = train_full["num_orders"]
model.fit(X_full, y_full)
```

2. **Prepare test features**

   I ensure `test_full` has the same preprocessing applied (merges + encodings + derived features), then:
```python
X_test = test_full[feature_cols]
test_pred = model.predict(X_test)
```

3. **Fill sample submission**
```python
import pandas as pd

sample_sub = pd.read_csv("sample_submission.csv")  # has 'id' and 'num_orders'
sample_sub["num_orders"] = test_pred
sample_sub.to_csv("submission.csv", index=False)
```

My final `submission.csv` matches the required format: one row per `id`, with predicted `num_orders` replacing the placeholder zeros.

***

## 6. Possible Improvements

Going forward, I could improve results by:

- Tuning hyperparameters more exhaustively via Random Search, Grid Search, or Optuna.
- Engineering richer time-based features such as lags and moving averages grouped by center or meal.
- Experimenting with alternative models like Gradient Boosting, XGBoost, LightGBM, or CatBoost.
- Applying target encoding for high-cardinality columns like `meal_id` and `center_id` instead of label encoding.
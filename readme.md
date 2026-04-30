# House Energy Usage Prediction  — ML Models (Linear & Logistic Regression)

This notebook trains two machine learning models on household energy usage data , built as part of the ML Module 1 hands-on activity.

---

## Models

### 1. Linear Regression — Predict Next Month's Bill
Predicts a household's monthly electricity bill based on usage in kWh.

**Input feature:** `monthly_kwh`  
**Target:** `monthly_bill`

**What it does:**
- Loads `household_dataset.csv` from Google Colab
- Splits data into 80/20 train-test sets
- Trains a `LinearRegression` model using scikit-learn
- Evaluates using MSE and RMSE
- Predicts next month's bill for a given kWh input (e.g. 300 kWh)
- Plots actual vs predicted values

---

### 2. Logistic Regression — Classify High/Low Usage Day
Classifies whether a day is a **High Usage** or **Low Usage** day based on consumption patterns.

**Input features:** `daily_kwh`, `peak_hour_usage`, `num_appliances_used`  
**Target:** `usage_label` (0 = Low Usage, 1 = High Usage)

**What it does:**
- Trains a `LogisticRegression` classifier (max_iter=1000)
- Reports accuracy and a full classification report
- Displays a confusion matrix heatmap using seaborn
- Tests a sample prediction: `daily_kwh=8.5, peak=4.2, appliances=6`

---

## Dataset

**File:** `household_dataset.csv`  
**Source:** VoltStream household energy usage data  
Upload via Google Colab's `files.upload()` when running the notebook.

**Columns used:**

| Column | Used In |
|---|---|
| `monthly_kwh` | Linear Regression (feature) |
| `monthly_bill` | Linear Regression (target) |
| `daily_kwh` | Logistic Regression (feature) |
| `peak_hour_usage` | Logistic Regression (feature) |
| `num_appliances_used` | Logistic Regression (feature) |
| `usage_label` | Logistic Regression (target) |

---

## Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
```



## Findings

- Linear Regression fits a straight line between monthly kWh usage and bill amount, giving a quick and interpretable prediction for next month's expected cost.
- Logistic Regression successfully classifies usage days as High or Low based on daily consumption, peak hours, and number of appliances — useful for detecting high-consumption patterns early.

---

## Deliverable

Notebook: `LinearRegression_LogisticRegression.ipynb`  

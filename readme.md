
# House Energy Usage Prediction — Linear & Logistic Regression

This notebook applies two supervised machine learning models to household electricity usage data to solve two practical energy analytics tasks:

1. **Predicting monthly electricity cost** using Linear Regression
2. **Classifying daily energy behavior** as Low Usage or High Usage using Logistic Regression

The project demonstrates how regression and classification can be used together on the same dataset for both cost forecasting and usage pattern detection.

---

## Project Overview

Household electricity data contains both continuous and behavioral patterns.

This notebook uses:

- **Linear Regression** to estimate a continuous value (**monthly electricity bill**)
- **Logistic Regression** to classify usage behavior into categories (**Low / High Usage**)

This creates a simple end-to-end ML workflow using:
- numerical prediction
- binary classification
- model evaluation
- result interpretation

---

## Models Used

## 1) Linear Regression — Monthly Bill Prediction

Linear Regression is used to predict a household's **monthly electricity bill** from total electricity usage.

### Objective
Estimate how much a household will be charged based on monthly energy consumption.

### Input
- `monthly_kwh`

### Target
- `monthly_bill`

### Model Idea
Linear Regression fits a straight-line relationship between electricity usage and billing:

$$y = mx + b$$

Where:
- $y$ = predicted monthly bill
- $x$ = monthly electricity usage (kWh)
- $m$ = slope (rate of bill increase per kWh)
- $b$ = intercept (base charge)

### What the model does
- learns the relationship between kWh usage and bill amount
- predicts expected bill for unseen monthly usage
- measures prediction error using regression metrics

### Evaluation Metrics
- **MSE (Mean Squared Error)** — average squared prediction error
- **RMSE (Root Mean Squared Error)** — interpretable average prediction error in bill units

### Results
- **MSE:** 2768.15
- **RMSE:** 52.61 — predictions are off by roughly ±₹52, which is realistic for real-world billing data

### Why it matters
- estimating future electricity cost
- budgeting expected monthly bills
- understanding how usage affects billing

---

## 2) Logistic Regression — Usage Classification

Logistic Regression is used to classify whether a household day is:

- **Low Usage (0)**
- **High Usage (1)**

### Objective
Identify whether daily energy behavior falls into a normal or high-consumption category.

### Inputs
- `peak_hour_usage`
- `num_appliances_used`

> **Note:** `daily_kwh` was intentionally excluded from the features.
> Initial analysis revealed that `usage_label` was directly derived from `daily_kwh`
> using a threshold rule, causing **target leakage** and an artificially high accuracy
> of 99.5%. Removing `daily_kwh` forces the model to learn genuine behavioral
> patterns from peak hour usage and appliance activity instead.

### Target
- `usage_label`

Where:
- `0` = Low Usage
- `1` = High Usage

### Model Idea
Logistic Regression predicts probability using the sigmoid function:

$$P(y=1) = \frac{1}{1 + e^{-(w_1x_1 + w_2x_2 + b)}}$$

Where:
- $x_1, x_2$ = input features
- $w_1, w_2$ = learned coefficients
- $b$ = intercept (bias)
- output = probability of High Usage

If probability > threshold, classify as **High Usage**, else **Low Usage**.

### What the model does
- learns patterns in daily consumption behavior from peak usage and appliance activity
- estimates whether a day belongs to low or high usage
- identifies high-consumption patterns without relying on direct energy totals

### Why it matters
- detecting excessive electricity usage
- monitoring high-consumption behavior
- identifying usage spikes before billing

---

## Dataset

**File:** `household_dataset.csv`

### Columns Used

| Column | Role |
|---|---|
| `monthly_kwh` | Linear Regression input |
| `monthly_bill` | Linear Regression target |
| `peak_hour_usage` | Logistic Regression input |
| `num_appliances_used` | Logistic Regression input |
| `usage_label` | Logistic Regression target |
| `daily_kwh` | Excluded — caused target leakage |

---

## Workflow

1. Load dataset in Google Colab
2. Explore feature and target variables
3. Split data into training and testing sets
4. Train Linear Regression model
5. Evaluate Linear Regression with MSE and RMSE
6. Identify and remove target leakage from Logistic Regression features
7. Train Logistic Regression on cleaned features
8. Evaluate with accuracy, classification report and confusion matrix
9. Interpret predictions and performance

---

## Evaluation Summary

### Linear Regression
- **MSE:** 2768.15
- **RMSE:** ±₹52.61
- Scatter plot confirms realistic spread around the regression line
- No issues — model captures the billing trend effectively

### Logistic Regression
- Features used: `peak_hour_usage`, `num_appliances_used`
- `daily_kwh` removed due to target leakage
- Accuracy reflects genuine pattern learning, not label rediscovery
- Confusion matrix shows reliable Low vs High usage separation

---

## Key Findings

- Monthly electricity bill increases predictably with total monthly energy consumption.
- Linear Regression captures this trend effectively with an RMSE of ±₹52.
- `daily_kwh` was found to directly determine `usage_label` through a threshold rule, causing target leakage and artificially inflated accuracy of 99.5%.
- After removing `daily_kwh`, the Logistic Regression model learns real behavioral patterns from peak hour usage and appliance activity.
- Peak hour usage is the strongest genuine predictor of high-consumption days.
- The combined approach demonstrates how one dataset can support both forecasting and classification tasks.

---

## Important Note on Target Leakage

During initial modelling, Logistic Regression achieved **99.5% accuracy** using
`daily_kwh`, `peak_hour_usage` and `num_appliances_used`.

Scatter plot analysis confirmed that `usage_label` was directly derived from
`daily_kwh` using a fixed threshold (~8.5 kWh):

```
daily_kwh < 8.5  →  Low Usage  (0)
daily_kwh > 8.5  →  High Usage (1)
```

This is a classic case of **target leakage** — the model was simply rediscovering
the labeling rule rather than learning meaningful patterns.

**Fix applied:** `daily_kwh` was removed from features. The model now uses only
`peak_hour_usage` and `num_appliances_used` to classify usage behavior.

---

## Skills Demonstrated

- supervised machine learning
- regression vs classification
- target leakage detection and correction
- feature-target separation
- train-test splitting
- model training using scikit-learn
- regression metrics (MSE, RMSE)
- classification metrics (accuracy, precision, recall, F1-score)
- confusion matrix interpretation
- model result interpretation

---

## Requirements

```bash
pandas
numpy
scikit-learn
matplotlib
seaborn
```
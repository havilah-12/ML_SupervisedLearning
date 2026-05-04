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

Linear Regression is used to predict a household’s **monthly electricity bill** from total electricity usage.

### Objective
Estimate how much a household will be charged based on monthly energy consumption.

### Input
- `monthly_kwh`

### Target
- `monthly_bill`

### Model Idea
Linear Regression fits a straight-line relationship between electricity usage and billing:

\[
y = mx + b
\]

Where:

- \(y\) = predicted monthly bill
- \(x\) = monthly electricity usage (kWh)
- \(m\) = slope (rate of bill increase per kWh)
- \(b\) = intercept (base charge)

This helps estimate how electricity cost grows with consumption.

### What the model does
- learns the relationship between kWh usage and bill amount
- predicts expected bill for unseen monthly usage
- measures prediction error using regression metrics

### Evaluation Metrics
- **MSE (Mean Squared Error)** — average squared prediction error
- **RMSE (Root Mean Squared Error)** — interpretable average prediction error in bill units

### Why it matters
This model is useful for:
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
- `daily_kwh`
- `peak_hour_usage`
- `num_appliances_used`

### Target
- `usage_label`

Where:

- `0` = Low Usage
- `1` = High Usage

### Model Idea
Logistic Regression predicts probability using the sigmoid function:

\[
P(y=1) = \frac{1}{1 + e^{-(w_1x_1 + w_2x_2 + w_3x_3 + b)}}
\]

Where:

- \(x_1, x_2, x_3\) = input features
- \(w_1, w_2, w_3\) = learned coefficients
- \(b\) = intercept (bias)
- output = probability of High Usage

If probability > threshold, classify as **High Usage**, else **Low Usage**.

### What the model does
- learns patterns in daily consumption behavior
- estimates whether a day belongs to low or high usage
- identifies high-consumption patterns early

### Why it matters
This model is useful for:
- detecting excessive electricity usage
- monitoring high-consumption behavior
- identifying usage spikes before billing

---

## Dataset

**File:** `household_dataset.csv`

The dataset contains household energy usage measurements used for both regression and classification tasks.

### Columns Used

| Column | Role |
|---|---|
| `monthly_kwh` | Linear Regression input |
| `monthly_bill` | Linear Regression target |
| `daily_kwh` | Logistic Regression input |
| `peak_hour_usage` | Logistic Regression input |
| `num_appliances_used` | Logistic Regression input |
| `usage_label` | Logistic Regression target |

---

## Workflow

The notebook follows this ML workflow:

1. Load dataset in Google Colab
2. Explore feature and target variables
3. Split data into training and testing sets
4. Train Linear Regression model
5. Train Logistic Regression model
6. Evaluate both models
7. Interpret predictions and performance

---

## Evaluation Summary

### Linear Regression
The Linear Regression model learns a direct mathematical relationship between electricity usage and billing.

It provides:
- simple and interpretable bill estimation
- low-complexity cost forecasting
- clear relationship between consumption and expense

### Logistic Regression
The Logistic Regression model classifies daily usage behavior with strong accuracy.

It provides:
- clear Low vs High usage classification
- interpretable feature importance
- strong confusion matrix performance
- reliable high-usage detection

---

## Key Findings

- Monthly electricity bill increases predictably with total monthly energy consumption.
- Linear Regression captures this trend effectively using a simple linear relationship.
- Daily energy behavior can be classified accurately using usage intensity and appliance activity.
- Logistic Regression clearly separates Low Usage and High Usage patterns.
- `daily_kwh` emerged as the strongest predictor in classification, showing that total daily consumption is the most influential factor in determining usage level.
- Peak hour usage also contributes positively to identifying high-consumption days.
- The combined approach demonstrates how one dataset can support both forecasting and classification tasks.

---

## Skills Demonstrated

This project demonstrates practical understanding of:

- supervised machine learning
- regression vs classification
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
# Credit Card Fraud Detection System

## Overview

This project builds an end-to-end **machine learning fraud detection system** that analyzes transaction data and predicts the probability of fraud.
The model is trained on an extremely imbalanced dataset and optimized using proper evaluation metrics and threshold tuning based on business cost.

The final model is deployed as a **Streamlit web application** where users can upload transaction datasets and receive fraud risk predictions.

---

## Live Demo

Upload a transaction dataset and receive fraud predictions.

**Demo:** https://credit-fraud-detection-demo.streamlit.app/

---

## Problem

Credit card fraud detection is challenging because fraudulent transactions are extremely rare.
Traditional metrics like accuracy are misleading in such cases.

This project focuses on:

* Handling **severe class imbalance**
* Using appropriate evaluation metrics
* Optimizing the decision threshold using business cost

---

## Dataset

Dataset used: Credit Card Fraud Detection dataset.

Features include:

* `Time`
* `Amount`
* `V1 – V28` (PCA-transformed features for privacy)
* `Class` (target variable)

Fraud cases represent only **~0.17% of all transactions**.

---

## Project Pipeline

### 1. Exploratory Data Analysis

* Analyzed fraud vs non-fraud distribution
* Investigated transaction amount patterns
* Verified extreme class imbalance

### 2. Baseline Model

A **Logistic Regression** model was trained as a baseline to confirm the dataset contains predictive signal.

### 3. Model Comparison

Multiple models were evaluated:

* Logistic Regression
* Random Forest
* Gradient Boosting

Evaluation metrics used:

* Precision
* Recall
* ROC-AUC
* PR-AUC

Random Forest provided the best performance.

### 4. Threshold Optimization

Instead of using the default threshold (0.5), the decision threshold was optimized using a **business cost model**:

* Missed fraud (False Negative): high cost
* False alarms (False Positive): lower cost

The optimal threshold minimizes total expected loss.

### 5. Deployment

The trained model is deployed using **Streamlit**, allowing users to:

1. Upload transaction datasets
2. Run fraud predictions
3. View fraud probabilities and predicted labels

---

## Tech Stack

Python
pandas
numpy
scikit-learn
Streamlit
joblib

---

## Project Structure

```
fraud-detection/

app.py                # Streamlit application
fraud_model.pkl       # Trained Random Forest model
notebook.ipynb        # Data analysis and model training
requirements.txt      # Dependencies
README.md
```

---

## Example Output

| Transaction | Fraud Probability | Prediction |
| ----------- | ----------------- | ---------- |
| 1           | 0.02              | Legit      |
| 2           | 0.91              | Fraud      |
| 3           | 0.73              | Fraud      |

---

## Key Learnings

* Handling highly imbalanced datasets
* Proper evaluation of fraud detection models
* Threshold tuning using business cost
* Deploying ML models as interactive applications

---

## Future Improvements

* Feature engineering on raw transaction data
* Real-time transaction scoring API
* Model monitoring and drift detection

---

## Author

Nadeem Ahmed

Data Science Student | Machine Learning & AI

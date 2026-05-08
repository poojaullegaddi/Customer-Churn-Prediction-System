# Customer Churn Prediction System

## Project Overview

This project is an end-to-end Machine Learning application that predicts whether a customer is likely to leave a telecom service company.

Customer churn prediction is one of the most important business use-cases in industries such as:

* Telecom
* Banking
* SaaS
* OTT Platforms
* Insurance
* E-commerce

The goal of this project is to help companies identify customers who are likely to churn so that retention strategies can be applied before losing them.

---

# Business Problem

Customer acquisition is expensive.

If companies can predict which customers are likely to leave, they can:

* Provide discounts
* Offer better plans
* Improve customer support
* Increase retention
* Reduce revenue loss

This project uses Machine Learning algorithms to predict customer churn based on customer behavior and service usage patterns.

---

# Project Architecture

```text
Customer Dataset
       ↓
Data Cleaning
       ↓
Feature Engineering
       ↓
Exploratory Data Analysis
       ↓
Train-Test Split
       ↓
Machine Learning Models
(Logistic Regression + Random Forest)
       ↓
Model Evaluation
(F1-score, Precision, Recall)
       ↓
Hyperparameter Tuning
       ↓
Save Best Model
```

---

# Features of This Project

* Data preprocessing and cleaning
* Missing value handling
* Feature engineering
* Exploratory Data Analysis (EDA)
* Logistic Regression model
* Random Forest model
* Hyperparameter tuning using GridSearchCV
* Model evaluation using:

  * Accuracy
  * Precision
  * Recall
  * F1-score
* Confusion Matrix
* Feature Importance Visualization
* Model saving using Joblib
* Resume-level project structure

---

# Tech Stack

| Technology   | Purpose                   |
| ------------ | ------------------------- |
| Python       | Programming Language      |
| Pandas       | Data manipulation         |
| NumPy        | Numerical operations      |
| Matplotlib   | Data visualization        |
| Seaborn      | Statistical visualization |
| Scikit-learn | Machine Learning          |
| Joblib       | Model serialization       |
| VS Code      | Development Environment   |

---

# Dataset Information

Dataset Used:

IBM Telecom Customer Churn Dataset

Dataset contains:

* 7043 customer records
* 21 features

Important Features:

| Feature         | Description                      |
| --------------- | -------------------------------- |
| tenure          | Number of months customer stayed |
| MonthlyCharges  | Monthly bill amount              |
| TotalCharges    | Total amount charged             |
| Contract        | Contract type                    |
| InternetService | Internet type                    |
| PaymentMethod   | Payment method                   |
| Churn           | Target column                    |

Target Variable:

| Value | Meaning               |
| ----- | --------------------- |
| Yes   | Customer left company |
| No    | Customer stayed       |

---

# Project Folder Structure

```text
customer-churn-project/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── model/
│   ├── churn_model.pkl
│   └── feature_names.pkl
│
├── venv/
│
├── train.py
├── requirements.txt
├── README.md
│
├── churn_distribution.png
├── confusion_matrix.png
├── correlation_heatmap.png
├── feature_importance.png
```

---
requirements.txt
```

Add:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
flask
joblib
```

---

# How to Run Project

## Step 1: Train Model

```bash
python train.py
```

This will:

* Load dataset
* Clean data
* Train models
* Evaluate performance
* Save model
* Generate charts

---

---

# Machine Learning Workflow

## 1. Data Cleaning

* Converted TotalCharges to numeric
* Handled missing values using median
* Removed unnecessary columns

---

## 2. Feature Encoding

Categorical features were converted into numerical values using:

```python
pd.get_dummies()
```

---

## 3. Train-Test Split

Dataset split:

* 80% Training
* 20% Testing

---

## 4. Logistic Regression

Used for baseline classification.

Advantages:

* Fast
* Interpretable
* Works well on structured data

---

## 5. Random Forest

Used ensemble learning with multiple decision trees.

Advantages:

* Handles non-linearity
* Reduces overfitting
* Better feature importance understanding

---

## 6. Hyperparameter Tuning

Implemented using:

```python
GridSearchCV
```

Used to find:

* Best number of trees
* Best depth
* Best performing configuration

---

# Evaluation Metrics

## Accuracy

Measures overall correctness.

Formula:

```text
Accuracy = Correct Predictions / Total Predictions
```

---

## Precision

Out of predicted churn customers, how many actually churned.

---

## Recall

Out of actual churn customers, how many were correctly identified.

---

## F1-score

Balance between Precision and Recall.

Important because churn datasets are usually imbalanced.

---

# Model Results

## Logistic Regression

| Metric   | Score |
| -------- | ----- |
| Accuracy | 82%   |
| F1-score | 0.64  |

---

## Random Forest

| Metric   | Score |
| -------- | ----- |
| Accuracy | 79%   |
| F1-score | 0.53  |

---

# Important Observation

Logistic Regression performed better than Random Forest on this dataset.

This indicates:

* Dataset had relatively linear patterns
* Simpler models can sometimes generalize better
* Complex models are not always superior

---

# Visualizations Generated

The project automatically generates:

* Churn Distribution
* Correlation Heatmap
* Confusion Matrix
* Feature Importance Graph

These visualizations help understand:

* Customer behavior
* Important features
* Model performance
* Business insights

---

# Conclusion

This project demonstrates how Machine Learning can be used to solve real business problems by identifying customers who are likely to churn.

The system can help businesses improve customer retention strategies and reduce revenue loss.

---

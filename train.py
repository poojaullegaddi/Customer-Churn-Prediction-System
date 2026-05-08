# =========================================================
# CUSTOMER CHURN PREDICTION SYSTEM
# RESUME LEVEL END-TO-END ML PROJECT
# =========================================================

# =========================
# STEP 1: IMPORT LIBRARIES
# =========================

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Evaluation Metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# =========================
# STEP 2: LOAD DATASET
# =========================

print("\n===== LOADING DATASET =====")

df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("\n===== FIRST 5 ROWS =====")
print(df.head())

# =========================
# STEP 3: DATASET INFO
# =========================

print("\n===== DATASET INFO =====")
print(df.info())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# =========================
# STEP 4: DATA CLEANING
# =========================

print("\n===== DATA CLEANING =====")

# Convert TotalCharges into numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Fill missing values
df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

# Remove unnecessary column
df.drop("customerID", axis=1, inplace=True)

print("Data Cleaning Completed")

# =========================
# STEP 5: TARGET ENCODING
# =========================

print("\n===== TARGET ENCODING =====")

df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

print(df["Churn"].head())

# =========================
# STEP 6: FEATURE ENCODING
# =========================

print("\n===== FEATURE ENCODING =====")

df = pd.get_dummies(df, drop_first=True)

print(df.head())

# =========================
# STEP 7: EXPLORATORY DATA ANALYSIS
# =========================

print("\n===== GENERATING VISUALIZATIONS =====")

# Churn Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x="Churn", data=df)
plt.title("Churn Distribution")
plt.savefig("churn_distribution.png")
plt.close()

# Correlation Heatmap
plt.figure(figsize=(18, 12))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.close()

print("Visualizations Saved")

# =========================
# STEP 8: SPLIT FEATURES & TARGET
# =========================

X = df.drop("Churn", axis=1)
y = df["Churn"]

# Save feature names for API later
joblib.dump(X.columns.tolist(), "model/feature_names.pkl")

# =========================
# STEP 9: TRAIN TEST SPLIT
# =========================

print("\n===== TRAIN TEST SPLIT =====")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("X_train Shape:", X_train.shape)
print("X_test Shape :", X_test.shape)

# =========================
# STEP 10: LOGISTIC REGRESSION PIPELINE
# =========================

print("\n===== LOGISTIC REGRESSION PIPELINE =====")

lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=2000))
])

# Train model
lr_pipeline.fit(X_train, y_train)

# Predictions
lr_pred = lr_pipeline.predict(X_test)

# Metrics
print("\n===== LOGISTIC REGRESSION RESULTS =====")

print("Accuracy :", accuracy_score(y_test, lr_pred))
print("Precision:", precision_score(y_test, lr_pred))
print("Recall   :", recall_score(y_test, lr_pred))
print("F1 Score :", f1_score(y_test, lr_pred))

# =========================
# STEP 11: RANDOM FOREST
# =========================

print("\n===== RANDOM FOREST =====")

rf_model = RandomForestClassifier(
    random_state=42
)

# Hyperparameter tuning
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None]
}

grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    cv=3,
    scoring="f1",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# Best model
best_rf = grid_search.best_estimator_

print("\nBest Parameters:")
print(grid_search.best_params_)

# Predictions
rf_pred = best_rf.predict(X_test)

# Metrics
print("\n===== RANDOM FOREST RESULTS =====")

print("Accuracy :", accuracy_score(y_test, rf_pred))
print("Precision:", precision_score(y_test, rf_pred))
print("Recall   :", recall_score(y_test, rf_pred))
print("F1 Score :", f1_score(y_test, rf_pred))

# =========================
# STEP 12: CONFUSION MATRIX
# =========================

print("\n===== CONFUSION MATRIX =====")

cm = confusion_matrix(y_test, rf_pred)

print(cm)

# Plot confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig("confusion_matrix.png")
plt.close()

# =========================
# STEP 13: CLASSIFICATION REPORT
# =========================

print("\n===== CLASSIFICATION REPORT =====")

print(classification_report(y_test, rf_pred))

# =========================
# STEP 14: FEATURE IMPORTANCE
# =========================

print("\n===== FEATURE IMPORTANCE =====")

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": best_rf.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print(importance_df.head(10))

# Plot feature importance
plt.figure(figsize=(10, 6))

sns.barplot(
    x="Importance",
    y="Feature",
    data=importance_df.head(10)
)

plt.title("Top 10 Important Features")

plt.savefig("feature_importance.png")
plt.close()

# =========================
# STEP 15: SAVE MODEL
# =========================

print("\n===== SAVING MODEL =====")

joblib.dump(best_rf, "model/churn_model.pkl")

print("Model Saved Successfully")

# =========================
# STEP 16: FINAL SUMMARY
# =========================

print("\n===== PROJECT COMPLETED =====")

print("""
Project Summary:
----------------
1. Loaded telecom churn dataset
2. Cleaned and preprocessed data
3. Handled missing values
4. Encoded categorical features
5. Performed EDA visualizations
6. Built Logistic Regression pipeline
7. Performed Random Forest hyperparameter tuning
8. Evaluated using F1-score
9. Generated confusion matrix
10. Saved trained model
11. Saved feature names for deployment

Files Generated:
----------------
1. model/churn_model.pkl
2. model/feature_names.pkl
3. churn_distribution.png
4. correlation_heatmap.png
5. confusion_matrix.png
6. feature_importance.png
""")
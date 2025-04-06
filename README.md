# Heart Disease Prediction using Machine Learning
 Overview
This project aims to predict the likelihood of heart disease in patients using a Logistic Regression model trained on clinical and demographic data. The model analyzes features such as cholesterol levels, blood pressure, and ECG readings to classify whether a patient is at risk (1) or not (0).

🔹 Problem Type: Binary Classification
🔹 Algorithm: Logistic Regression (Supervised ML)
🔹 Key Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn

Dataset
Source: Kagle Heart Disease Dataset.

Features:
Numerical: age, trestbps (blood pressure), chol (cholesterol), thalach (max heart rate).
Categorical: sex, cp (chest pain type), fbs (fasting blood sugar).
Target: target (0 = no disease, 1 = disease).
Size: 1024 records × 14 columns.

# Workflow
1)Exploratory Data Analysis (EDA)
  -Visualized distributions (histograms, pair plots).
  -Analyzed correlations (heatmap).
  -Checked for class imbalance.

2)Data Preprocessing
  -Handled missing values (none found).
  -Scaled features using StandardScaler.
  -Split data: 80% train, 20% test.

3)Model Training
  -Trained Logistic Regression with default hyperparameters.
  -Saved model and scaler using joblib.
  -Evaluation

4)Metrics:
  -Accuracy: 85%
  - Precision: 84%
  -Recall: 88%
  -F1-Score: 86%
  - AUC-ROC: 0.92

5)Generated:
  -Confusion matrix.
  -ROC curve.
  -Feature importance plot.

# Results & Insights
Top Predictors:
  -Chest pain type (cp): Strongest positive correlation.
  -Maximum heart rate (thalach): Higher values reduce risk.

Model Strengths:
  -High recall (88%) → Good at detecting true positives.
  -Interpretable coefficients (unlike black-box models).

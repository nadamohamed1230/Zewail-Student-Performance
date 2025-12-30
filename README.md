# 🎓 Smart Student Performance Prediction System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://zewail-student-performance-hz3v3eptqpz2vnamlpwunk.streamlit.app/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**A Machine Learning-powered analytics dashboard to predict academic performance based on 38 demographic, behavioral, and academic features.**

---

## 🚀 Live Demo
**Click to try the app:** [**Smart Student Predictor**](https://zewail-student-performance-hz3v3eptqpz2vnamlpwunk.streamlit.app/)

![App Screenshot](https://raw.githubusercontent.com/nadamohamed1230/Zewail-Student-Performance/main/app.png)

---

## 📌 Project Overview
This project was developed for the **CIE 417: Machine Learning** course at **Zewail City of Science and Technology**. The goal is to simulate a real-world educational analytics pipeline that helps institutions identify at-risk students early.

### 🎯 Objectives
The system predicts three key performance indicators for a student:
1.  **Final Score (0-100):** A regression prediction of the exact numerical score.
2.  **Pass/Fail Status:** A binary classification to flag at-risk students.
3.  **Final Letter Grade (A-F):** A multi-class classification for detailed academic standing.

---

## 🧠 Machine Learning Approach

We analyzed a dataset of **20,000 student records** and trained models using a robust pipeline.

### 🧹 Data Preprocessing Pipeline
To ensure robust predictions, the raw data undergoes a rigorous cleaning process before training:

1.  **Data Cleaning & Imputation**
    * **Missing Target Removal:** Rows with missing values in `final_score`, `final_grade`, or `pass_fail` are dropped to ensure ground-truth accuracy.
    * **Numerical Imputation:** Missing numerical values are filled with the **median** to be robust against outliers.
    * **Categorical Imputation:** Missing text values are filled with the **mode** (most frequent value).

2.  **Outlier Management (Winsorization)**
    * We identified extreme outliers in features like `parent_income` and `online_portal_usage_minutes`.
    * **Method:** Values below the **1st percentile** or above the **99th percentile** are capped at those limits to prevent model skewing.

3.  **Feature Transformation**
    * **Scaling:** Numerical features are standardized using `StandardScaler` (Mean = 0, Variance = 1).
    * **Encoding:** Categorical variables (`gender`, `part_time_job`, `course_type`) are converted into numerical vectors using `OneHotEncoder`.

4.  **Handling Class Imbalance**
    * **SMOTE (Synthetic Minority Over-sampling Technique):** Applied during the experimental phase to balance the distribution of "Pass" vs. "Fail" students.

---

### 📊 Model Performance
We evaluated multiple models (Logistic Regression, SVM, Random Forest, Decision Tree) and selected the best performers for the app:

| Target Variable | Prediction Type | Model Selected | Performance Metrics |
| :--- | :--- | :--- | :--- |
| **Final Score** | Regression | **Ridge Regression** | High stability and low variance |
| **Pass/Fail** | Binary Classification | **XGBoost Classifier** | Accuracy: **94.1%**, AUC: **0.93** |
| **Final Grade** | Multi-class Classification | **XGBoost Classifier** | Accuracy: **69.1%**, Weighted F1: **0.61** |

---

## 📂 Project Structure
This directory tree explains the organization of files in the repository:

```text
├── app.py                     # The main Streamlit application script
├── requirements.txt           # List of Python libraries required to run the app
├── Term_Project_Dataset_20K.csv # The raw dataset used for training the models
├── machine_project.ipynb      # Jupyter Notebook containing EDA, experiments & training
├── logo.png                   # Zewail City Logo used in the app interface
└── README.md                  # Project documentation (this file)

# 🏦 Credit Risk Prediction Using XGBoost, LightGBM & SHAP (Credit-G Dataset)

This project builds an interpretable machine learning system for
predicting credit default risk using the **Credit-G dataset from
OpenML**.\
The goal is not only to achieve high accuracy but also to **explain**
the model's behavior using **SHAP (Shapley Additive Explanations)**.

------------------------------------------------------------------------

## 🚀 Project Features

### ✅ 1. Data Loading & Preprocessing

-   Loaded the **Credit-G** dataset from OpenML\
-   Encoded the target variable (`good → 0`, `bad → 1`)
-   Selected optimal features based on domain knowledge
-   Applied preprocessing using:
    -   `StandardScaler` for numeric features\
    -   `OneHotEncoder` for categorical features\
    -   `ColumnTransformer` inside a `Pipeline`

------------------------------------------------------------------------

## 🤖 2. Models Trained

We build two strong gradient boosting models:

-   **XGBoost (XGBClassifier)**
-   **LightGBM (LGBMClassifier)**

Both are wrapped inside an end-to-end `Pipeline` with preprocessing.

------------------------------------------------------------------------

## 🔍 3. Robust Cross-Validation

Implemented **Stratified K-Fold Cross Validation** to ensure stable and
fair model evaluation:

-   AUC-ROC\
-   Precision\
-   Recall\
-   F1-score

------------------------------------------------------------------------

## 📊 4. Model Evaluation

Evaluated both models using:

-   AUC-ROC curves\
-   Precision-Recall curves\
-   Confusion matrix\
-   Classification report

------------------------------------------------------------------------

## 🔎 5. Explainability with SHAP

SHAP is used for both global and local interpretability:

### **Global Explanations**

-   SHAP summary plot\
-   SHAP dependence plots\
-   Feature importance ranking

### **Local Explanations**

-   Individual-level SHAP force plots\
-   High-risk (default) and low-risk (non-default) borrower explanations

------------------------------------------------------------------------

## 🧠 6. Key Counter-Intuitive Insights (from SHAP)

Three surprising findings revealed by SHAP:

1.  **Mid-range loan amounts (3,000--5,000 DM) are riskier than high
    loan amounts**\
2.  **Borrowers with no savings can be lower-risk than those with small
    savings**\
3.  **Younger borrowers with strong credit history can outperform
    middle-aged borrowers**

These insights reshape traditional credit policy rules.

------------------------------------------------------------------------

## 🏦 7. Policy Recommendations

-   Introduce refined risk tiers for mid-range loans\
-   Reduce penalty associated with "no savings" if employment is stable\
-   Prioritize **recent credit behavior** over age-based assumptions

------------------------------------------------------------------------

## 📁 Project Structure

    ├── notebooks/
    │   └── credit_risk_analysis.ipynb
    ├── shap_plots/
    │   ├── summary_plot.png
    │   ├── dependence_credit_amount.png
    │   ├── local_default_case.png
    │   └── local_non_default_case.png
    ├── data/
    │   └── credit-g.csv (optional export)
    ├── models/
    │   └── xgb_model.pkl
    │   └── lgbm_model.pkl
    ├── README.md
    └── requirements.txt

------------------------------------------------------------------------

## 🛠 Installation

    pip install -r requirements.txt

------------------------------------------------------------------------

## ▶ Run the Project

    python train.py

Or open the Jupyter notebook:

    jupyter notebook

------------------------------------------------------------------------

## 📚 Requirements

    pandas
    numpy
    scikit-learn
    xgboost
    lightgbm
    matplotlib
    shap

------------------------------------------------------------------------

## ✨ Author

**Thirumurugan B**\
Final-year ECE Student \| ML & Data Science\
Email: *thirumurugan1801@gmail.com*

------------------------------------------------------------------------

## ⭐ Contributions

Feel free to contribute!\
Fork this repo and submit a pull request.

------------------------------------------------------------------------

## 📄 License

This project is licensed under the **MIT License**.

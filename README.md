# Credit-Risk-Analysis

This project demonstrates how **machine learning can be applied to financial risk assessment**.  
By predicting loan defaults accurately, institutions can improve portfolio quality, reduce losses, and make data-driven lending decisions.

**Project Overview**

This project focuses on **Credit Risk Analysis** — predicting the likelihood of a borrower defaulting on a loan using **machine learning techniques**.  
The goal is to assist financial institutions in making better lending decisions by identifying high-risk applicants.

The notebook demonstrates the **end-to-end data science pipeline**:
- Data cleaning and preprocessing  
- Exploratory Data Analysis (EDA)  
- Feature engineering  
- Model training and evaluation

**Objective**

To build a **predictive model** that classifies loan applicants into **“low-risk”** or **“high-risk”** categories based on financial and personal attributes.

**Dataset**

- **Source:** Synthetic dataset (credit_risk_dataset.csv)  
- **Size:** 32,581 records and 12 features  
- **Target Variable:** `loan_status`  
  - `0` → Loan fully paid  
  - `1` → Loan defaulted  

**Key Features**
| Feature | Description |
|----------|--------------|
| person_age | Age of the borrower |
| person_income | Annual income |
| person_home_ownership | Home ownership type (RENT, OWN, MORTGAGE) |
| person_emp_length | Employment length in years |
| loan_intent | Purpose of the loan |
| loan_grade | Assigned loan grade |
| loan_amnt | Loan amount |
| loan_int_rate | Interest rate (%) |
| loan_percent_income | Loan amount as % of income |
| cb_person_default_on_file | Credit bureau record of default |
| cb_person_cred_hist_length | Credit history length (years) |

---
**Data Cleaning**

Steps performed:
1. Removed duplicate records  
2. Imputed missing values (`SimpleImputer`)  
3. Handled data types and standardized column names  
4. Verified data integrity and shape consistency  

**Final Clean Dataset:** 32,416 records × 12 features

**Exploratory Data Analysis (EDA)**

Visualizations and summary statistics were generated using:
- **Seaborn** and **Matplotlib** for plotting  
- **Correlation heatmap** to understand relationships  
- **Boxplots** to compare numeric features vs. loan status  
- **Countplots** for categorical variables  

**Key Insights**
- 21.8% of customers defaulted on loans.  
- Higher loan amounts and interest rates increase default risk.  
- Applicants with home ownership or mortgage showed lower default rates.  

**Feature Engineering**

Two new features were created:
- `income_to_loan` → Ratio of income to loan amount  
- `emp_age_ratio` → Employment length relative to age  

These features improved model interpretability and predictive power.

**Model Building**

### Data Split
| Set | Proportion | Purpose |
|------|-------------|----------|
| Training | 70% | Model learning |
| Validation | 15% | Hyperparameter tuning |
| Test | 15% | Final evaluation |

**Preprocessing Pipeline**

Using **Scikit-learn’s `ColumnTransformer`**:
- Numeric features → Imputation + Scaling  
- Categorical features → Imputation + One-Hot Encoding  

**Models Implemented**

1. **Logistic Regression**
   - Baseline model
   - Provides interpretability and clear feature coefficients  

2. **Histogram Gradient Boosting Classifier**
   - Advanced ensemble model
   - Captures complex non-linear relationships
   - Early stopping enabled for efficiency  

**Model Evaluation**

Metrics used:
- ROC-AUC Score  
- Confusion Matrix  
- Precision, Recall, F1-Score  
- Precision-Recall Curve  

**Gradient Boosting** outperformed Logistic Regression in overall predictive power and AUC score.


**Results Summary**
| Model | ROC-AUC | Key Strength |
|--------|----------|---------------|
| Logistic Regression | Moderate | Easy to interpret |
| Gradient Boosting | High | Best accuracy and recall |

**Model Saving with Joblib**

The project uses **`joblib`** to save and reload trained machine learning models efficiently.

Why Joblib?
- It’s optimized for serializing large **NumPy arrays** and **Scikit-learn objects**.  
- Faster and more memory-efficient than `pickle` for saving ML models.  
- Ensures that trained pipelines and preprocessing steps can be easily reused.

**Future Improvements**

- Handle class imbalance using **SMOTE / ADASYN**  
- Hyperparameter tuning with **RandomizedSearchCV**  
- Compare with **XGBoost**, **Random Forest**, and **LightGBM**  
- Build a **Streamlit dashboard** for live credit scoring  

**Tech Stack**

| Category | Tools Used |
|-----------|-------------|
| Language | Python |
| Libraries | pandas, numpy, scikit-learn, seaborn, matplotlib |
| Environment | Google Colab / Jupyter Notebook |
| Model Persistence | joblib |


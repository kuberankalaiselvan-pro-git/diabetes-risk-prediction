# 🩺 Diabetes Risk Prediction Using Health Indicators

Predicting diabetes risk using data analytics and machine learning techniques based on medical and lifestyle survey data.  
**University group project** for the *Introduction to Data Analytics* module at  
**Asia Pacific University of Technology and Innovation (APU)**.

---

## 📌 Project Overview

Diabetes is one of the most prevalent chronic diseases worldwide, affecting millions of adults and placing a growing burden on healthcare systems. Early identification of individuals at high risk is critical for enabling preventive care, improving patient outcomes, and optimizing healthcare resources.

This project applies **exploratory data analysis (EDA)** and **machine learning classification models** to predict whether an individual is likely to have diabetes based on health indicators such as BMI, age, blood pressure, and self‑reported health status.

Beyond predictive performance, the project emphasizes **interpretability, fairness, and responsible use of machine learning in healthcare contexts**.

---

## 🎯 Objectives

- Clean and preprocess a large‑scale health dataset  
- Explore relationships between health indicators and diabetes prevalence  
- Build and compare supervised machine learning models  
- Identify features with the strongest influence on diabetes risk  
- Recommend a suitable model for early screening use cases  
- Address ethical considerations in predictive healthcare analytics  

---

## 📊 Dataset

- **Name:** Diabetes Health Indicators Dataset (BRFSS 2015)  
- **Source:** Kaggle – Alex Teboul  
- **Records:** 253,680  
- **Target Variable:** `Diabetes_binary`  
  - `0` – Non‑Diabetic  
  - `1` – Diabetic  
- **Features:** 21 medical and lifestyle indicators  
  (e.g. BMI, Age, HighBP, GenHlth, PhysActivity)

> **Note:** The dataset is not included in this repository due to size and licensing considerations.

---

## 🔍 Exploratory Data Analysis (EDA)

EDA was conducted to understand the distributions, relationships, and trends within the dataset. Key findings include:

- Diabetes prevalence increases significantly with:
  - Higher **Body Mass Index (BMI)**
  - Older age groups
  - Poor self‑rated general health (**GenHlth**)
  - Presence of high blood pressure (**HighBP**)
- Significant **class imbalance** (~14% diabetic cases)
- Extreme BMI outliers were handled using **winsorization**

Visualizations such as histograms, boxplots, and correlation heatmaps provided statistical and clinical justification for downstream feature selection.

---

## 🧠 Feature Selection

To ensure robustness, multiple feature selection techniques were applied:

- **SelectKBest (Univariate Selection)**  
- **Recursive Feature Elimination (RFE)**  
- **Tree‑based Feature Importance**

Across all techniques, the most influential predictors were:

- **BMI**
- **General Health (GenHlth)**
- **Age**
- **High Blood Pressure (HighBP)**

These features align closely with established medical literature and public health research.

---

## 🤖 Machine Learning Models

Three supervised classification models were trained and evaluated:

| Model                | Purpose |
|---------------------|---------|
| Logistic Regression | Interpretable baseline model |
| Gradient Boosting   | High recall and strong predictive performance |
| **Random Forest ✅** | Best overall balance of performance and robustness |

### Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1‑Score  
- ROC‑AUC  

✅ **Random Forest** was selected as the final model due to:

- Balanced precision and recall  
- Strong ROC‑AUC (~0.84)  
- Reliable feature importance for interpretability  
- Resistance to overfitting  

Class imbalance was addressed using **SMOTE**, resulting in fairer and more reliable model training.

---

## 🧪 Methodology & Techniques

- CRISP‑DM framework  
- Data cleaning and preprocessing  
- SMOTE resampling  
- Feature selection and dimensionality reduction  
- Model training, comparison, and evaluation  
- Ethical risk assessment  

---

## 🚀 Demo Application (Streamlit)

A lightweight **Streamlit application** was developed to demonstrate how the trained Random Forest model can be used to predict diabetes risk based on user‑provided health indicators.

📂 Implementation available in the `app/` directory.

> ⚠️ **Important:**  
> This application is provided for demonstration purposes only and is **not deployed publicly**.  
> It is not intended for real‑world clinical decision‑making.

---

## ⚠️ Ethical Considerations

Given the healthcare domain, several ethical aspects were carefully considered:

- **Data privacy** and responsible handling  
- **Algorithmic bias** and fairness  
- **Model interpretability** and transparency  
- Clear acknowledgment that ML systems should **support—not replace—medical professionals**

The project aligns with principles of responsible and ethical AI usage.

---

## 👥 Team & Individual Contribution

This was a **group university project**.

### 🔹 My Individual Contributions

- Random Forest model development  
- Feature selection and sampling strategy  
- Model evaluation and comparison  
- Streamlit demo interface  
- Ethical considerations and interpretability analysis  

Other team members contributed primarily to **Gradient Boosting** and **Logistic Regression** modelling.

### 👥 Collaborators

- **Koo Junn Yuan**  
- **Low Wei Hao**

---

## 🛠 Tools & Technologies

- Python  
- Pandas & NumPy  
- Matplotlib & Seaborn  
- Scikit‑learn  
- Imbalanced‑learn (SMOTE)  
- Google Colab  
- Streamlit  

---

## 📁 Repository Structure


```text
diabetes-risk-prediction/
├── app/          # Streamlit demo application
├── notebooks/    # Jupyter notebook (EDA + ML workflow)
├── reports/      # Proposal, final report, and presentation slides
└── README.md
```

---

## 📜 Disclaimer

This project was developed strictly for **academic learning purposes**.  
It has **not been clinically validated** and should **not** be used as a medical diagnostic tool.

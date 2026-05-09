
# 🚀 Credit Risk Scoring System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Project-Completed-green)

---

## 🧠 Overview

This is an end-to-end **Machine Learning project** that predicts whether a loan applicant is high or low credit risk.

It helps financial institutions make data-driven lending decisions.

---

## 📊 Problem Statement

Given customer financial data, the model predicts:

- 🔴 Risk of default (Reject)
- 🟢 Low risk (Approve)

---

## ⚙️ Tech Stack

- Python 🐍  
- Pandas  
- NumPy  
- Scikit-learn  
- Joblib  

---

## 📁 Project Structure

```

app/
predict.py
example.py

data/
german_credit_data.csv
german_credit_with_risk.csv

models/
model.pkl

notebooks/
eda.ipynb

src/
train.py

````id="structure2"

---

## 🧪 How It Works

1. Data preprocessing  
2. Encoding categorical features  
3. Feature scaling  
4. Model training  
5. Prediction pipeline  

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python src/train.py
python app/example.py
````

---

## 🔮 Sample Prediction

```json id="sample2"
{
    "probability": 0.86,
    "decision": "Reject"
}
```

---

## 📊 Model Output Flow

```
Customer Data ➜ Preprocessing ➜ ML Model ➜ Probability ➜ Decision
```

---

## 📈 Future Improvements

* 🔥 Deploy using FastAPI
* 📊 Add Streamlit dashboard
* ⚡ Improve accuracy with XGBoost tuning
* 🌐 Real-time API integration

---


# Credit Risk Scoring System using Machine Learning

## 🧠 Overview
This project is a Machine Learning system that predicts credit risk for loan applicants.

It analyzes customer financial and personal data to determine:
- Probability of default
- Final decision: Approve or Reject

---

## 📊 Dataset
The project uses the German Credit Dataset:

- german_credit_data.csv
- german_credit_with_risk.csv

### Features:
- Age
- Job
- Credit amount
- Duration
- Housing
- Saving accounts
- Checking account
- Purpose

---

## ⚙️ Project Structure

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

````

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python src/train.py
python app/example.py
````

---

## 🔮 Sample Output

```json
{
    "probability": 0.86,
    "decision": "Reject"
}
```

---

## 📦 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib

---

## 📈 Future Improvements

* Deploy model using FastAPI
* Build dashboard with Streamlit
* Improve model using XGBoost tuning
* Add real-time API predictions

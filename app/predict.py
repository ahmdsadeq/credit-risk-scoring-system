import joblib
import os
import pandas as pd
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "model.pkl")

model = joblib.load(MODEL_PATH)

THRESHOLD = 0.3



def predict_credit_risk(input_json):
    """
    input_json: dict (JSON-like input)
    """

    input_df = pd.DataFrame([input_json])

    expected_cols = model.named_steps['preprocessor'].feature_names_in_
    input_df = input_df.reindex(columns=expected_cols, fill_value=0)

    prob = model.predict_proba(input_df)[0][1]

    decision = "Reject" if prob >= THRESHOLD else "Approve"

    return {
        "probability": float(prob),
        "decision": decision
    }



if __name__ == "__main__":

    print("Predict script is running...\n")

    sample_json = {
        "Job": 2,
        "Duration": 12,
        "Purpose": "car",
        "Housing": "own",
        "Sex": "male",
        "Saving accounts": "little",
        "Checking account": "moderate",
        "Credit amount": 5000,
        "Age": 35,
        "Unnamed: 0": 0
    }

    result = predict_credit_risk(sample_json)

    print("Prediction Result:")
    print(json.dumps(result, indent=4))
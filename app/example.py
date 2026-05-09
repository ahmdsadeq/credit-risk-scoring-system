from predict import predict_credit_risk

print("Running example test...\n")


sample = {
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

try:
    result = predict_credit_risk(sample)
    print("Prediction Result:")
    print(result)

except Exception as e:
    print("Error occurred:")
    print(str(e))
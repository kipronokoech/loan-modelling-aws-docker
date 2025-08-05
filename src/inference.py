import argparse
import pandas as pd
import json
import joblib
import os
from sklearn.metrics import classification_report
from processing import preprocess_inference

def run_inference(input_path, model_path, feature_path, output_dir):
    model = joblib.load(model_path)
    feature_columns = joblib.load(feature_path)

    df = pd.read_csv(input_path)
    X = preprocess_inference(df.copy(), feature_columns)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    results = classification_report(df["Personal_Loan"], predictions, output_dict=True, digits=4)
    print(json.dumps(results, indent=3))

    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    with open(os.path.join(output_dir, "output_test.json"), "w") as output_file:
        json.dump(results, output_file, indent=3)

    df["Predicted_Label"] = predictions
    df["Probability"] = probabilities

    # Write prediction results
    df.to_csv(os.path.join(output_dir, "inference_results.csv"), index=False)
    print(f"Inference complete. Results saved to: {output_dir}")

    return df

if __name__ == "__main__":

    input_path = "../data/test.csv"
    output_dir = "../output/"
    model_file = "../models/loan_model.joblib"
    features_file = "../models/feature_columns.joblib"

    run_inference(input_path=input_path, model_path=model_file, feature_path=features_file, output_dir=output_dir)

# loan_model/inference.py
import argparse
import os
import json
import pandas as pd
import joblib
from sklearn.metrics import classification_report
# from loan_model.processing import preprocess_inference
import numpy as np

OUTLIER_COLUMNS = ["Income", "CCAvg", "Mortgage"]
DROP_COLUMNS = ["ID", "ZIPCode"]

def treat_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    Lower_Whisker = Q1 - 1.5 * IQR
    Upper_Whisker = Q3 + 1.5 * IQR
    df[col] = np.clip(df[col], Lower_Whisker, Upper_Whisker)
    return df

def preprocess_inference(df, feature_columns):
    for c in OUTLIER_COLUMNS:
        if c in df.columns:
            df = treat_outliers(df, c)

    X = df.drop(DROP_COLUMNS, axis=1, errors="ignore")
    X = pd.get_dummies(X, drop_first=True)

    missing = set(feature_columns) - set(X.columns)
    if missing:
        print(f"[INFO] Adding {len(missing)} missing feature columns: {missing}")

    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_columns]

    return X


def run_inference(input_path, model_path, feature_path, output_dir):
    df = pd.read_csv(input_path)
    model = joblib.load(model_path)
    feature_columns = joblib.load(feature_path)

    X = preprocess_inference(df.copy(), feature_columns)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    results = classification_report(df["Personal_Loan"], predictions, output_dict=True, digits=4)
    print(json.dumps(results, indent=3))

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "output_test.json"), "w") as f:
        json.dump(results, f, indent=3)

    df["Predicted_Label"] = predictions
    df["Probability"] = probabilities
    df.to_csv(os.path.join(output_dir, "inference_results.csv"), index=False)

    print(f"Inference complete. Output written to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--features", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    run_inference(args.input, args.model, args.features, args.output)

# src/train.py

import os
import argparse
import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from processing import preprocess_training

def train(input_path: str, model_dir: str) -> None:
    os.makedirs(model_dir, exist_ok=True)

    df = pd.read_csv(input_path)
    X, y = preprocess_training(df)

    model = DecisionTreeClassifier(max_depth=3, ccp_alpha=0.0014, random_state=42)
    model.fit(X, y)

    joblib.dump(model, os.path.join(model_dir, "loan_model.joblib"))
    joblib.dump(X.columns.tolist(), os.path.join(model_dir, "feature_columns.joblib"))
    print(f"Model and features saved to: {model_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, default="/opt/ml/input/data/train/train.csv")
    parser.add_argument("--model-dir", type=str, default="/opt/ml/model")

    args = parser.parse_args()
    train(args.input_path, args.model_dir)

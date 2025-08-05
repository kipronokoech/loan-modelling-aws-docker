# src/train.py

import os
import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
# from processing import preprocess_training
# from loan_model.processing import preprocess_training

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

def preprocess_training(df):
    for c in OUTLIER_COLUMNS:
        if c in df.columns:
            df = treat_outliers(df, c)

    y = df["Personal_Loan"].astype("int")
    X = df.drop(["Personal_Loan"] + DROP_COLUMNS, axis=1, errors="ignore")
    X = pd.get_dummies(X, drop_first=True)
    return X, y

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

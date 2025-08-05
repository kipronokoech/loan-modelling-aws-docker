import os
import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from processing import preprocess_training

def train(input_path: str, models_dir: str = "output") -> None:
    os.makedirs(models_dir, exist_ok=True)

    df = pd.read_csv(input_path)
    X, y = preprocess_training(df)

    model = DecisionTreeClassifier(max_depth=3, ccp_alpha=0.0014, random_state=42)
    model.fit(X, y)

    joblib.dump(model, os.path.join(models_dir, "loan_model.joblib"))
    joblib.dump(X.columns.tolist(), os.path.join(models_dir, "feature_columns.joblib"))
    print(f"Model and features saved to: {models_dir}")

if __name__ == "__main__":
    input_path = "../data/train.csv"
    models_dir = "../models"

    train(input_path, models_dir)


import pandas as pd
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

def preprocess_training(df):
    for c in OUTLIER_COLUMNS:
        if c in df.columns:
            df = treat_outliers(df, c)

    y = df["Personal_Loan"].astype("int")
    X = df.drop(["Personal_Loan"] + DROP_COLUMNS, axis=1, errors="ignore")
    X = pd.get_dummies(X, drop_first=True)
    return X, y

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


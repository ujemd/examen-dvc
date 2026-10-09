from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = Path("data/raw_data/raw.csv")
OUT_DIR = Path("data/processed_data")


def main():
    df = pd.read_csv(RAW_PATH)

    # The date column is a string, so it can't be used as a model feature
    df = df.drop(columns=["date"])

    # The target is the last column (silica_concentrate)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(OUT_DIR / "X_train.csv", index=False)
    X_test.to_csv(OUT_DIR / "X_test.csv", index=False)
    y_train.to_csv(OUT_DIR / "y_train.csv", index=False)
    y_test.to_csv(OUT_DIR / "y_test.csv", index=False)

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")


if __name__ == "__main__":
    main()
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path("data/processed_data")


def main():
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")

    # Fit on the training set only, then apply to both sets.
    # Fitting on the test set would leak information into training.
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns
    )

    X_train_scaled.to_csv(DATA_DIR / "X_train_scaled.csv", index=False)
    X_test_scaled.to_csv(DATA_DIR / "X_test_scaled.csv", index=False)

    print("Train means (should be ~0):", X_train_scaled.mean().round(3).tolist())
    print("Train stds (should be ~1):", X_train_scaled.std().round(3).tolist())


if __name__ == "__main__":
    main()
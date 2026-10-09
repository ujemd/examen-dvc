from pathlib import Path
import pickle

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

DATA_DIR = Path("data/processed_data")
MODELS_DIR = Path("models")


def main():
    X_train = pd.read_csv(DATA_DIR / "X_train_scaled.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze("columns")

    with open(MODELS_DIR / "best_params.pkl", "rb") as f:
        best_params = pickle.load(f)

    model = GradientBoostingRegressor(random_state=42, **best_params)
    model.fit(X_train, y_train)

    with open(MODELS_DIR / "gbr_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Trained with params:", best_params)
    print("Saved models/gbr_model.pkl")


if __name__ == "__main__":
    main()
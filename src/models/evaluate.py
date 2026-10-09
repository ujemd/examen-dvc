from pathlib import Path
import json
import pickle

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

DATA_DIR = Path("data/processed_data")
MODELS_DIR = Path("models")
METRICS_DIR = Path("metrics")


def main():
    X_test = pd.read_csv(DATA_DIR / "X_test_scaled.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze("columns")

    with open(MODELS_DIR / "gbr_model.pkl", "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)

    scores = {
        "mse": mean_squared_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_DIR / "scores.json", "w") as f:
        json.dump(scores, f, indent=4)

    predictions = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
    predictions.to_csv(Path("data") / "prediction.csv", index=False)

    print("Scores:", scores)


if __name__ == "__main__":
    main()
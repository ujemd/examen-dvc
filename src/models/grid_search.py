from pathlib import Path
import pickle

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

DATA_DIR = Path("data/processed_data")
MODELS_DIR = Path("models")


def main():
    X_train = pd.read_csv(DATA_DIR / "X_train_scaled.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze("columns")

    param_grid = {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [2, 3, 4],
    }

    search = GridSearchCV(
        GradientBoostingRegressor(random_state=42),
        param_grid,
        cv=3,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / "best_params.pkl", "wb") as f:
        pickle.dump(search.best_params_, f)

    print("Best params:", search.best_params_)
    print("Best CV MSE:", -search.best_score_)


if __name__ == "__main__":
    main()
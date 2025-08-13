from utils import PROCESSED_DIR, save_model, read_csv
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def main():
    train = read_csv(f"{PROCESSED_DIR}/train_features.csv")
    drop_cols = ["unit", "cycle", "RUL", "failed"]
    X = train.drop(columns=[c for c in drop_cols if c in train.columns])
    y = train["RUL"]

    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=350, max_depth=None, random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)

    pred = model.predict(X_va)
    mae  = mean_absolute_error(y_va, pred)
    rmse = mean_squared_error(y_va, pred) ** 0.5   # <-- changed here

    print(f"RUL validation MAE: {mae:.3f}, RMSE: {rmse:.3f}")
    save_model(model, "rul_random_forest.joblib")

if __name__ == "__main__":
    main()

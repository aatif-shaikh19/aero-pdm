from utils import PROCESSED_DIR, read_csv, load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd

def main():
    test = read_csv(f"{PROCESSED_DIR}/test_features.csv")
    drop_cols = ["unit", "cycle", "RUL", "failed"]
    X = test.drop(columns=[c for c in drop_cols if c in test.columns])
    y = test["RUL"]

    rul_model = load_model("rul_random_forest.joblib")
    y_pred = rul_model.predict(X)

    mae  = mean_absolute_error(y, y_pred)
    rmse = mean_squared_error(y, y_pred) ** 0.5     # <-- changed here

    print(f"TEST RUL MAE: {mae:.3f}, RMSE: {rmse:.3f}")

    iso = load_model("anomaly_isolation_forest.joblib")
    scores = -iso.score_samples(X)
    out = test[["unit","cycle"]].copy()
    out["true_RUL"] = y
    out["pred_RUL"] = y_pred
    out["anomaly_score"] = scores
    out.to_csv(f"{PROCESSED_DIR}/test_predictions.csv", index=False)
    print("Wrote predictions to data/processed/test_predictions.csv")

if __name__ == "__main__":
    main()

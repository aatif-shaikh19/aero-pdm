
from utils import PROCESSED_DIR, save_model, read_csv
from sklearn.ensemble import IsolationForest

def main():
    train = read_csv(f"{PROCESSED_DIR}/train_features.csv")
    # Healthy data: RUL > some threshold (e.g., > 30 cycles)
    healthy = train[train["RUL"] > 30].copy()

    drop_cols = ["unit", "cycle", "RUL", "failed"]
    X = healthy.drop(columns=[c for c in drop_cols if c in healthy.columns])
    iso = IsolationForest(n_estimators=300, contamination=0.05, random_state=42)
    iso.fit(X)

    save_model(iso, "anomaly_isolation_forest.joblib")
    print("IsolationForest trained on healthy data.")

if __name__ == "__main__":
    main()

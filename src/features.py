import argparse
import pandas as pd
import numpy as np
from utils import RAW_DIR, PROCESSED_DIR, read_csv, write_csv

def window_features(df, window=20, sensor_cols=None):
    if sensor_cols is None:
        sensor_cols = [c for c in df.columns if c.startswith("s")]

    out = []
    for unit, grp in df.groupby("unit"):
        grp = grp.sort_values("cycle").reset_index(drop=True)

        # Compute rolling aggregates ONCE per window
        roll = grp[sensor_cols].rolling(window=window, min_periods=1)
        roll_mean = roll.mean().add_suffix("_mean")
        roll_std  = roll.std().fillna(0).add_suffix("_std")
        roll_min  = roll.min().add_suffix("_min")
        roll_max  = roll.max().add_suffix("_max")

        feats = pd.concat([roll_mean, roll_std, roll_min, roll_max], axis=1)

        # Add identifiers / settings / targets (if present)
        feats["unit"] = unit
        feats["cycle"] = grp["cycle"].values
        if "setting1" in grp.columns:
            feats["setting1"] = grp["setting1"].values
        if "setting2" in grp.columns:
            feats["setting2"] = grp["setting2"].values
        if "RUL" in grp.columns:
            feats["RUL"] = grp["RUL"].values
        feats["failed"] = grp["failed"].values if "failed" in grp.columns else 0

        out.append(feats)

    return pd.concat(out, ignore_index=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=20)
    args = parser.parse_args()

    train = read_csv(f"{RAW_DIR}/train.csv")
    test  = read_csv(f"{RAW_DIR}/test.csv")

    sensor_cols = [c for c in train.columns if c.startswith("s")]
    if not sensor_cols:
        raise ValueError("No sensor columns found that start with 's'. Did data generation run successfully?")

    train_feats = window_features(train, window=args.window, sensor_cols=sensor_cols)
    test_feats  = window_features(test,  window=args.window, sensor_cols=sensor_cols)

    write_csv(train_feats, f"{PROCESSED_DIR}/train_features.csv")
    write_csv(test_feats,  f"{PROCESSED_DIR}/test_features.csv")
    print("Feature engineering complete.")
    print(f"Train features shape: {train_feats.shape}, Test features shape: {test_feats.shape}")

if __name__ == "__main__":
    main()

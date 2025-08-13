
import argparse
import numpy as np
import pandas as pd
from utils import RAW_DIR, write_csv, ensure_dirs

def simulate_engine(unit_id, max_cycles, n_sensors, seed=None):
    rng = np.random.default_rng(seed + unit_id if seed is not None else None)

    # Initialize baseline sensor states
    baseline = rng.normal(0, 1, size=n_sensors)
    drift = rng.normal(0.001, 0.002, size=n_sensors)  # slow degradation per cycle
    noise_scale = rng.uniform(0.02, 0.08, size=n_sensors)

    # Operating settings (e.g., altitude/temp proxy)
    setting1 = rng.uniform(0.8, 1.2)
    setting2 = rng.uniform(0.9, 1.1)

    # Random failure cycle (ensures some fail earlier, some later)
    fail_cycle = rng.integers(int(max_cycles*0.5), max_cycles)

    rows = []
    for c in range(1, max_cycles+1):
        # gradual degradation + noise
        sensors = baseline + drift*c + rng.normal(0, noise_scale, size=n_sensors)

        rows.append({
            "unit": unit_id,
            "cycle": c,
            **{f"s{i+1}": sensors[i] for i in range(n_sensors)},
            "setting1": setting1,
            "setting2": setting2,
        })

    df = pd.DataFrame(rows)
    # RUL = time to failure
    df["RUL"] = np.maximum(fail_cycle - df["cycle"], 0)

    # Label 'failed' at failure cycle (RUL=0)
    df["failed"] = (df["RUL"] == 0).astype(int)
    return df

def split_train_test(df, test_frac=0.2, seed=42):
    rng = np.random.default_rng(seed)
    units = df["unit"].unique()
    test_units = set(rng.choice(units, size=int(len(units)*test_frac), replace=False))
    train = df[~df["unit"].isin(test_units)].copy()
    test = df[df["unit"].isin(test_units)].copy()
    return train, test

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_engines", type=int, default=80)
    parser.add_argument("--max_cycles", type=int, default=250)
    parser.add_argument("--n_sensors", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensure_dirs()

    dfs = [simulate_engine(u, args.max_cycles, args.n_sensors, seed=args.seed) for u in range(1, args.n_engines+1)]
    data = pd.concat(dfs, ignore_index=True)

    train, test = split_train_test(data, test_frac=0.2, seed=args.seed)

    write_csv(train, f"{RAW_DIR}/train.csv")
    write_csv(test, f"{RAW_DIR}/test.csv")

    print(f"Wrote {len(train)} train rows and {len(test)} test rows.")

if __name__ == "__main__":
    main()

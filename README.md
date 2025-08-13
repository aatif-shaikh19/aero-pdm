
# Aero-PDM: Predictive Maintenance for Turbofan Engines (Synthetic)

End-to-end project for **Remaining Useful Life (RUL) prediction** and **anomaly detection** on synthetic turbofan engine telemetry, built to mirror aerospace predictive maintenance workflows.

## Features
- Synthetic multivariate time-series generator (engines, cycles, sensors, operating settings)
- Windowed feature engineering
- RUL regression (RandomForest)
- Anomaly detection (IsolationForest) on healthy data
- Evaluation (MAE/RMSE)
- Streamlit dashboard for quick exploration & predictions
- Dockerfile + Makefile for reproducible runs

> You can swap the synthetic data with NASA C-MAPSS easily (same column format).

## Project Structure
```
aero-pdm/
├─ README.md
├─ requirements.txt
├─ Dockerfile
├─ Makefile
├─ .gitignore
├─ data/
│  ├─ raw/               # generated synthetic or external (e.g., C-MAPSS) csv files
│  └─ processed/         # engineered features
├─ models/               # saved sklearn models
├─ src/
│  ├─ generate_data.py   # synthetic data generator
│  ├─ features.py        # windowed features
│  ├─ train_rul.py       # trains RUL regressor
│  ├─ train_anomaly.py   # trains IsolationForest
│  ├─ evaluate.py        # evaluates regressor
│  └─ utils.py           # helpers
└─ app/
   └─ streamlit_app.py   # dashboard
```

## Quickstart

### 1) Install
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Generate synthetic data
```bash
python src/generate_data.py --n_engines 80 --max_cycles 250 --n_sensors 12 --seed 42
```

### 3) Create features
```bash
python src/features.py --window 20
```

### 4) Train RUL model
```bash
python src/train_rul.py
```

### 5) Train anomaly detector
```bash
python src/train_anomaly.py
```

### 6) Evaluate
```bash
python src/evaluate.py
```

### 7) Run the dashboard
```bash
streamlit run app/streamlit_app.py
```

## Using NASA C-MAPSS (Optional)
If you download a C-MAPSS subset (e.g., FD001), reformat it to match `data/raw/train.csv` & `data/raw/test.csv`:
- Required columns: `unit,cycle,s1,...,sK,setting1,setting2,RUL` for train; same minus `RUL` for test if you want to predict.
Then skip `generate_data.py` and continue from `features.py`.

## Makefile Cheatsheet
```bash
make setup        # install deps
make data         # generate synthetic data
make features     # build features
make train        # train both models
make eval         # evaluate
make app          # run Streamlit
```

## Notes
- Models are intentionally lightweight (sklearn) for easy portability.
- For deep learning (LSTM/GRU), replace `train_rul.py` with a PyTorch implementation.

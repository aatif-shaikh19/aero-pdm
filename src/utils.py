
import os
import joblib
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

def ensure_dirs():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

def save_model(model, name: str):
    ensure_dirs()
    joblib.dump(model, os.path.join(MODELS_DIR, name))

def load_model(name: str):
    return joblib.load(os.path.join(MODELS_DIR, name))

def read_csv(path):
    return pd.read_csv(path)

def write_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

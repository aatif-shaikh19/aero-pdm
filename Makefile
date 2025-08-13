
.PHONY: setup data features train eval app all

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt || true

data:
	python src/generate_data.py --n_engines 80 --max_cycles 250 --n_sensors 12 --seed 42

features:
	python src/features.py --window 20

train:
	python src/train_rul.py && python src/train_anomaly.py

eval:
	python src/evaluate.py

app:
	streamlit run app/streamlit_app.py

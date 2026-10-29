# Predictive Maintenance (LSTM) - Remaining Useful Life Prediction

Production-ready Remaining Useful Life (RUL) prediction for turbofan engines using
the NASA C-MAPSS dataset. Features stacked LSTM/GRU deep learning models, XGBoost/
Random Forest baselines, NASA asymmetric scoring, FastAPI inference endpoint, and
a Streamlit health-monitoring dashboard.

## Problem

Machines fail unexpectedly, and unplanned downtime is expensive. This model reads
multivariate sensor data (temperature, pressure, vibration) cycle by cycle and
predicts how many cycles an engine has left before failure — its Remaining Useful
Life (RUL). Maintenance can then be scheduled before the breakdown.

## Dataset

NASA C-MAPSS turbofan engine degradation dataset (FD001–FD004). Each engine has
multivariate sensor readings recorded cycle by cycle until failure.

- `train_FD001.txt` — training engine trajectories
- `test_FD001.txt` — test engine trajectories (last cycle of each engine)
- `RUL_FD001.txt` — ground-truth RUL at the last cycle for each test engine

## Repository structure

```
predictive-maintenance-lstm/
├── configs/config.yaml          # Central configuration (YAML)
├── data/                        # raw + processed data (gitignored if large)
├── notebooks/                   # EDA and experimentation
├── reports/                     # figures, evaluation_report.json
├── src/
│   ├── __init__.py
│   ├── data/                    # loader.py, sequences.py
│   ├── features/                # rolling.py
│   ├── models/                  # baseline.py, lstm_model.py
│   ├── training/                # trainer.py
│   ├── evaluation/              # metrics.py, plots.py
│   └── api/                     # FastAPI app
├── dashboard/
│   └── app.py                   # Streamlit dashboard
├── tests/                       # pytest suite
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

## Tech stack

Python, Pandas, NumPy, Scikit-learn, PyTorch, XGBoost, Matplotlib/Plotly,
FastAPI, Streamlit.

## Quick start

```bash
# Install dependencies
pip install -e ".[dev]"

# Download C-MAPSS data into data/raw/ (see data/README.md)

# Phase 1-2: Data + baselines
python -m src.data.loader --config configs/config.yaml
python -m src.features.rolling --config configs/config.yaml
python -m src.models.baseline --config configs/config.yaml

# Phase 3: Train LSTM
python -m src.training.trainer --config configs/config.yaml

# Phase 4: Evaluate
python -m src.evaluation.metrics --config configs/config.yaml

# Phase 5: Deploy
docker compose up
# API: http://localhost:8000/docs
# Dashboard: streamlit run dashboard/app.py
```

## Results (example)

| Model | RMSE | MAE | NASA Score |
|-------|------|-----|------------|
| Naive (mean) | ... | ... | ... |
| XGBoost | ... | ... | ... |
| LSTM | ... | ... | ... |
| Attention-LSTM | ... | ... | ... |

*Fill in after training. The NASA asymmetric score is the primary business metric —
it penalizes late (dangerous) RUL predictions more than early ones.*

## License

MIT
"# Predictive-Maintenance-LSTM-" 

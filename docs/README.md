# Documentation

This directory contains all project-level documentation for the **Spatio-Temporal Traffic Forecasting** project using GNN-variant models on METR-LA and PEMS-BAY datasets.


## Project Structure
```
spatiotemporal-traffic-forecasting/
├── src/                        # core ML code (data, models, training, inference, utils)
├── app/                        # backend and frontend application logic (can be part of a separate repo if needed)
├── notebooks/                  # exploratory notebooks for data and model POCs
├── docs/                       # project documentation
├── datasets/                   # not tracked by git — download from Google Drive
├── Dockerfile                  # container setup
├── pyproject.toml              # dependencies managed via uv
└── README.md                  
```

## Environment Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

**Prerequisites:** Python version is pinned in `.python-version` — uv handles this automatically.
```bash
# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# install dependencies and create virtual environment
uv sync

# activate virtual environment
source .venv/bin/activate        # mac/linux
.venv\Scripts\activate           # windows
```

All dependencies and their exact versions are locked in `uv.lock` for reproducible environments across platforms.


## Train
Set [Config](../src/configs/config.py) and run trainer.
```bash
cd spatiotemporal-traffic-forecasting
python -m src.trainer
```

To run mlflow server: 
```bash
mlflow ui --backend-store-uri ./mlruns
```
Navigate to http://localhost:5000 for UI.

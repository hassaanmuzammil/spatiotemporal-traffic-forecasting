# spatiotemporal-traffic-forecasting
Predicting Traffic Congestion Using Spatio-Temporal Graph Neural Networks

## Project Structure
```
spatiotemporal-traffic-forecasting/
├── src/                        # core ML code (data, models, training, inference, utils)
├── app/                        # backend and frontend application logic (can be part of a separate repo if needed)
├── configs/                    # model/dataset config YAMLs (metr_la, pems_bay)
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

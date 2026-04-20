import torch

# =========================
# Data
# =========================
DATA_DIR = "datasets"
dataset = "METR-LA"  # or "PEMS-BAY"

train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

window_size = 12
horizon = 12
batch_size = 32
USE_TEMPORAL_FEATURES = True   # True for speed + temporal features
DATASET_CONFIG = {
    "METR-LA": {
        "group_key": "df",
        "h5_path":  f"{DATA_DIR}/raw/metr_la/metr_la.h5",
        "adj_path": f"{DATA_DIR}/raw/metr_la/adj_metr_la.pkl",
        "meta_path": None,
        "pyg_path": f"{DATA_DIR}/processed/metr_la/metr_la.pt",
    },
    "PEMS-BAY": {
        "group_key": "speed",
        "h5_path":  f"{DATA_DIR}/raw/pems_bay/pems_bay.h5",
        "adj_path": f"{DATA_DIR}/raw/pems_bay/adj_mx_bay.pkl",
        "meta_path": f"{DATA_DIR}/raw/pems_bay/pems_bay_meta.h5",
        "pyg_path": f"{DATA_DIR}/processed/pems_bay/pems_bay.pt",
    },
}

# =========================
# Model
# =========================

# --- CURRENT SETTING: no extra temporal features ---
# Only raw traffic speed is used as input
#in_feats = 1

# --- If you want to enable temporal features later ---
# Input becomes:
# 1 speed feature
# + 6 temporal features:
#   time_of_day_sin, time_of_day_cos,
#   day_of_week_sin, day_of_week_cos,
#   is_weekend, is_holiday
in_feats = 7
temporal_feats = 32
hidden_feats = 64
out_feats = horizon
num_layers = 3
num_heads = 4
dropout = 0.3

# =========================
# Training
# =========================
CKPT_DIR = "ckpts"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lr = 1e-3
epochs = 10
print_every = 100
save_every = 5

# =========================
# MLflow
# =========================
ENABLE_MLFLOW = False   # turn on later after mlflow setup
MLFLOW_TRACKING_URI = "./mlruns"
MLFLOW_EXPERIMENT_NAME = "spatiotemporal-traffic-forecasting"
import torch

# data
DATA_DIR = "datasets"
dataset = "PEMS-BAY" # PEMS-BAY
train_ratio = 0.6
val_ratio = 0.2
test_ratio  = 0.2
window_size  = 12
horizon = 12
batch_size = 32

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

# model
in_feats = window_size
temporal_feats = 32   # updated from 32
hidden_feats = 64     # updated from 64
out_feats = horizon
num_layers = 3
num_heads = 4
dropout = 0.3

# train
CKPT_DIR = "ckpts"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 1e-3
epochs = 10
print_every = 100
save_every = 5

# mlflow
ENABLE_MLFLOW = True
MLFLOW_TRACKING_URI = "./mlruns"
MLFLOW_EXPERIMENT_NAME = "spatiotemporal-traffic-forecasting"
import pickle
from typing import Literal
import h5py
import pandas as pd

from src.configs.config import DATASET_CONFIG


def load_traffic_h5(dataset: Literal["METR-LA", "PEMS-BAY"]) -> pd.DataFrame:
    config = DATASET_CONFIG[dataset]

    with h5py.File(config["h5_path"], "r") as f:
        group = f[config["group_key"]]
        values = group["block0_values"][:]
        sensor_ids = group["axis0"][:]
        timestamps = group["axis1"][:]

    times = pd.to_datetime(timestamps, unit="ns")
    N = len(sensor_ids)
    if values.shape[0] == N:
        values = values.T

    df = pd.DataFrame(values, index=times, columns=sensor_ids)
    df.columns = df.columns.astype(str)

    return df


def load_adj_data(dataset: Literal["METR-LA", "PEMS-BAY"]):
    config = DATASET_CONFIG[dataset]

    with open(config["adj_path"], "rb") as f:
        adj_data = pickle.load(f, encoding="latin1")

    sensor_ids, sensor_id_to_idx, adj_mx = adj_data
    return sensor_ids, sensor_id_to_idx, adj_mx
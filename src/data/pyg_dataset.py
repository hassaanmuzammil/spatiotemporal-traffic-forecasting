import os
from typing import Literal
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader

from src.configs.config import (
    DATASET_CONFIG,
    train_ratio,
    val_ratio,
    test_ratio,
    window_size,
    horizon,
    batch_size
)

from src.data.loader import load_traffic_h5, load_adj_data
from src.data.preprocess import interpolate_speed


def build_edge_index_and_attr(adj_mx: np.ndarray):
    adj_tensor = torch.tensor(adj_mx, dtype=torch.float)
    edge_index = adj_tensor.nonzero().t().contiguous()

    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]

    edge_attr = adj_tensor[edge_index[0], edge_index[1]].unsqueeze(1)

    print(f"edge_index: {edge_index.shape}")
    print(f"edge_attr: {edge_attr.shape}")
    return edge_index, edge_attr


def build_pyg_data_list(
    speed: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    window_size: int,
    horizon: int,
) -> list:
    data_list = []
    T = speed.shape[0]

    for t in range(window_size - 1, T - horizon):
        x = speed[t - window_size + 1 : t + 1].T
        y = speed[t + 1 : t + 1 + horizon].T
        data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))

    print(f"total samples: {len(data_list)}")
    print(f"x shape: {data_list[0].x.shape}")
    print(f"y shape: {data_list[0].y.shape}")

    return data_list


class TrafficDataset(InMemoryDataset):
    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


def build_dataset(dataset: Literal["METR-LA", "PEMS-BAY"], save=False):
    pyg_path = DATASET_CONFIG[dataset]["pyg_path"]
    os.makedirs(os.path.dirname(pyg_path), exist_ok=True)

    print(f"\n{'='*40}")
    print(f" {dataset}")
    print(f"{'='*40}")

    print("\n[1] Loading speed data")
    df = load_traffic_h5(dataset)

    print("\n[2] Interpolating missing values")
    df = interpolate_speed(df)

    print("\n[3] Splitting")
    n = len(df)
    t_end = int(n * train_ratio)
    v_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:t_end]
    val_df   = df.iloc[t_end:v_end]
    test_df  = df.iloc[v_end:]

    print(f"train: {len(train_df)} | val: {len(val_df)} | test: {len(test_df)}")

    print("\n[4] Normalizing")
    mean = train_df.values.mean()
    std  = train_df.values.std()
    print(f"mean={mean:.3f}  std={std:.3f}")

    train_df = (train_df - mean) / std
    val_df   = (val_df   - mean) / std
    test_df  = (test_df  - mean) / std

    print("\n[5] Loading adjacency matrix")
    _, _, adj_mx = load_adj_data(dataset)
    edge_index, edge_attr = build_edge_index_and_attr(adj_mx)

    print("\n[6] Building sliding windows")
    def to_tensor(d): return torch.tensor(d.values, dtype=torch.float32)

    train_list = build_pyg_data_list(to_tensor(train_df), edge_index, edge_attr, window_size, horizon)
    val_list   = build_pyg_data_list(to_tensor(val_df),   edge_index, edge_attr, window_size, horizon)
    test_list  = build_pyg_data_list(to_tensor(test_df),  edge_index, edge_attr, window_size, horizon)

    if save:
        torch.save({
            "train": train_list,
            "val":   val_list,
            "test":  test_list,
            "mean":  mean,
            "std":   std,
        }, pyg_path)

        print(f"\nSaved → {pyg_path}")

    return TrafficDataset(train_list), TrafficDataset(val_list), TrafficDataset(test_list), mean, std


if __name__ == "__main__":
    metr_df = load_traffic_h5("METR-LA")
    metr_sensor_ids, metr_sensor_id_to_idx, metr_adj_mx = load_adj_data("METR-LA")
    metr_df_interp = interpolate_speed(metr_df)

    train_metr, val_metr, test_metr, mean, std = build_dataset("METR-LA")

    train_loader = DataLoader(train_metr, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_metr, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_metr, batch_size=batch_size, shuffle=False)
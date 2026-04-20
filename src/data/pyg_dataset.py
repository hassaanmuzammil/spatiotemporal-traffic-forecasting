import os
from typing import Literal, Optional

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
    batch_size,
    USE_TEMPORAL_FEATURES,
)
from src.data.loader import load_traffic_h5, load_adj_data
from src.data.preprocess import interpolate_speed, get_temporal_features


def build_edge_index_and_attr(adj_mx: np.ndarray):
    adj_tensor = torch.tensor(adj_mx, dtype=torch.float32)
    edge_index = adj_tensor.nonzero().t().contiguous()

    # remove self-loops
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]

    edge_attr = adj_tensor[edge_index[0], edge_index[1]].unsqueeze(1)

    print(f"edge_index: {edge_index.shape}")
    print(f"edge_attr: {edge_attr.shape}")
    return edge_index, edge_attr


def split_temporal_features(temporal_df, t_end, v_end):
    train_temporal = temporal_df.iloc[:t_end]
    val_temporal = temporal_df.iloc[t_end:v_end]
    test_temporal = temporal_df.iloc[v_end:]
    return train_temporal, val_temporal, test_temporal


def combine_speed_and_temporal_window(
    speed_window: torch.Tensor,
    temporal_window: torch.Tensor,
) -> torch.Tensor:
    """
    speed_window:    [window_size, num_nodes]
    temporal_window: [window_size, num_temporal_features]

    returns:
        x: [num_nodes, window_size * (1 + num_temporal_features)]
    """
    num_nodes = speed_window.shape[1]

    speed_expanded = speed_window.unsqueeze(-1)
    # [window_size, num_nodes, 1]

    temporal_expanded = temporal_window.unsqueeze(1).repeat(1, num_nodes, 1)
    # [window_size, num_nodes, num_temporal_features]

    combined = torch.cat([speed_expanded, temporal_expanded], dim=-1)
    # [window_size, num_nodes, 1 + num_temporal_features]

    x = combined.permute(1, 0, 2).reshape(num_nodes, -1)
    # [num_nodes, window_size * (1 + num_temporal_features)]

    return x


def speed_only_window(speed_window: torch.Tensor) -> torch.Tensor:
    """
    speed_window: [window_size, num_nodes]

    returns:
        x: [num_nodes, window_size]
    """
    return speed_window.T


def build_pyg_data_list(
    speed: torch.Tensor,
    temporal: Optional[torch.Tensor],
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    window_size: int,
    horizon: int,
    use_temporal_features: bool = True,
) -> list:
    data_list = []
    total_steps = speed.shape[0]

    for t in range(window_size - 1, total_steps - horizon):
        speed_window = speed[t - window_size + 1: t + 1]

        if use_temporal_features:
            if temporal is None:
                raise ValueError("temporal tensor is None but use_temporal_features=True")

            temporal_window = temporal[t - window_size + 1: t + 1]
            x = combine_speed_and_temporal_window(speed_window, temporal_window)
        else:
            x = speed_only_window(speed_window)

        y = speed[t + 1: t + 1 + horizon].T
        # [num_nodes, horizon]

        data_list.append(
            Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
            )
        )

    print(f"total samples: {len(data_list)}")
    if len(data_list) > 0:
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

    print(f"\n{'=' * 40}")
    print(f" {dataset}")
    print(f"{'=' * 40}")

    print("\n[1] Loading speed data")
    df = load_traffic_h5(dataset)

    print("\n[2] Interpolating missing values")
    df = interpolate_speed(df)

    if USE_TEMPORAL_FEATURES:
        print("\n[3] Building temporal features")
        temporal_df = get_temporal_features(df)
        print(f"temporal feature shape: {temporal_df.shape}")
        print(f"temporal columns: {temporal_df.columns.tolist()}")
    else:
        print("\n[3] Skipping temporal features (speed only)")
        temporal_df = None

    print("\n[4] Splitting")
    n = len(df)
    t_end = int(n * train_ratio)
    v_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:t_end]
    val_df = df.iloc[t_end:v_end]
    test_df = df.iloc[v_end:]

    if USE_TEMPORAL_FEATURES:
        train_temporal_df, val_temporal_df, test_temporal_df = split_temporal_features(
            temporal_df, t_end, v_end
        )
    else:
        train_temporal_df = None
        val_temporal_df = None
        test_temporal_df = None

    print(f"train: {len(train_df)} | val: {len(val_df)} | test: {len(test_df)}")

    print("\n[5] Normalizing speed")
    mean = train_df.values.mean()
    std = train_df.values.std()
    print(f"mean={mean:.3f}  std={std:.3f}")

    train_df = (train_df - mean) / std
    val_df = (val_df - mean) / std
    test_df = (test_df - mean) / std

    print("\n[6] Loading adjacency matrix")
    _, _, adj_mx = load_adj_data(dataset)
    edge_index, edge_attr = build_edge_index_and_attr(adj_mx)

    print("\n[7] Building sliding windows")

    def to_tensor(d):
        return torch.tensor(d.values, dtype=torch.float32)

    train_list = build_pyg_data_list(
        speed=to_tensor(train_df),
        temporal=to_tensor(train_temporal_df) if train_temporal_df is not None else None,
        edge_index=edge_index,
        edge_attr=edge_attr,
        window_size=window_size,
        horizon=horizon,
        use_temporal_features=USE_TEMPORAL_FEATURES,
    )

    val_list = build_pyg_data_list(
        speed=to_tensor(val_df),
        temporal=to_tensor(val_temporal_df) if val_temporal_df is not None else None,
        edge_index=edge_index,
        edge_attr=edge_attr,
        window_size=window_size,
        horizon=horizon,
        use_temporal_features=USE_TEMPORAL_FEATURES,
    )

    test_list = build_pyg_data_list(
        speed=to_tensor(test_df),
        temporal=to_tensor(test_temporal_df) if test_temporal_df is not None else None,
        edge_index=edge_index,
        edge_attr=edge_attr,
        window_size=window_size,
        horizon=horizon,
        use_temporal_features=USE_TEMPORAL_FEATURES,
    )

    if save:
        torch.save(
            {
                "train": train_list,
                "val": val_list,
                "test": test_list,
                "mean": mean,
                "std": std,
            },
            pyg_path,
        )
        print(f"\nSaved -> {pyg_path}")

    return (
        TrafficDataset(train_list),
        TrafficDataset(val_list),
        TrafficDataset(test_list),
        mean,
        std,
    )


if __name__ == "__main__":
    from src.configs.config import dataset

    train_dataset, val_dataset, test_dataset, mean, std = build_dataset(dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
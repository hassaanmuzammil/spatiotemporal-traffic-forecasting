import os
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from src.data.pyg_dataset import build_dataset
from src.models.gcn_baseline import GCNBaseline
from src.models.gcn_attention import GCNSpatioTemporalAttention
from src.models.stgcn import SpatioTemporalGNN
from src.train.train import train
from src.train.evaluate import test
from src.train.utils import plot_losses, create_run_dir
import src.configs.config as cfg
 
 
def main():
    # ── mlflow setup ──
    mlflow_run = None
    if cfg.ENABLE_MLFLOW:
        import mlflow
        mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(cfg.MLFLOW_EXPERIMENT_NAME)
        mlflow_run = mlflow.start_run()
 
        # log full config — skip dunder attrs, dicts (DATASET_CONFIG), and modules
        config_params = {
            k: str(v)
            for k, v in vars(cfg).items()
            if not k.startswith("_") and not isinstance(v, (dict, type(__builtins__)))
        }
        mlflow.log_params(config_params)
 
    # ── load dataset ──
    train_ds, val_ds, test_ds, mean, std = build_dataset("METR-LA")
 
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)
 
    # log dataset info
    if mlflow_run is not None:
        import mlflow
        mlflow.log_params({
            "dataset": cfg.dataset,
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "test_samples": len(test_ds),
            "data_mean": round(float(mean), 6),
            "data_std": round(float(std),  6),
        })
 
    # ── model ──
    # model = GCNBaseline(
    #     cfg.in_feats,
    #     cfg.hidden_feats,
    #     cfg.out_feats,
    # ).to(cfg.device)

    # model = GCNSpatioTemporalAttention(
    #     1, 
    #     cfg.temporal_feats, 
    #     cfg.hidden_feats,
    #     cfg.out_feats,
    #     cfg.num_heads,
    # )

    model = SpatioTemporalGNN(
        cfg.in_feats,
        cfg.hidden_feats,
        cfg.out_feats,
        cfg.num_layers,
        cfg.num_heads,
        cfg.dropout
    )
 
    # ── optimizer / loss / scheduler ──
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    criterion = nn.HuberLoss(delta=1.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=5,
        factor=0.5
    )
 
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
 
    if mlflow_run is not None:
        import mlflow
        mlflow.log_param("model_parameters", num_params)
 
    # ── create checkpoint folder for this run ──
    run_ckpt_dir = create_run_dir()
    print(f"Checkpoints will be saved to → {run_ckpt_dir}")
 
    # ── train ──
    train_losses, val_losses = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=cfg.device,
        epochs=cfg.epochs,
        ckpt_dir=run_ckpt_dir,
        print_every=cfg.print_every,
        save_every=cfg.save_every,
        scheduler=scheduler,
        mlflow_run=mlflow_run,
    )
    plot_losses(train_losses, val_losses, run_ckpt_dir)

    # ── test ──
    print("\nTesting best model...")
    best_model_path = os.path.join(run_ckpt_dir, "best_model.pt")
    model.load_state_dict(torch.load(best_model_path))
    test_metrics = test(model, test_loader, mean, std, cfg.device)
 
    # log test metrics if test() returns a dict; otherwise this is a no-op
    if mlflow_run is not None and isinstance(test_metrics, dict):
        import mlflow
        mlflow.set_tags({f"test_{k}": v for k, v in test_metrics.items()})
 
    # ── end mlflow run ──
    if mlflow_run is not None:
        import mlflow
        mlflow.end_run()
 
 
if __name__ == "__main__":
    torch.manual_seed(42)
    main()
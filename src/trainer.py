import os
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from src.data.pyg_dataset import build_dataset
from src.models.gcn_baseline import GCNBaseline
from src.train.train import train
from src.train.evaluate import test
from src.train.utils import plot_losses, create_run_dir
from src.configs.config import (
    in_feats,
    hidden_feats,
    out_feats,
    device,
    batch_size,
    lr,
    epochs,
    print_every,
    save_every
)

def main():
    # ── load dataset ──
    train_ds, val_ds, test_ds, mean, std = build_dataset("METR-LA")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # ── model ──
    model = GCNBaseline(
        in_feats,
        hidden_feats,
        out_feats,
    ).to(device)

    # ── optimizer / loss / scheduler ──
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.HuberLoss(delta=1.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=5,
        factor=0.5
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

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
        device=device,
        epochs=epochs,
        ckpt_dir=run_ckpt_dir,
        print_every=print_every,
        save_every=save_every,
        scheduler=scheduler
    )
    plot_losses(train_losses, val_losses, run_ckpt_dir)

    # ── test ──
    print("\nTesting best model...")
    best_model_path = os.path.join(run_ckpt_dir, "best_model.pt")
    model.load_state_dict(torch.load(best_model_path))
    test(model, test_loader, mean, std, device)


if __name__ == "__main__":
    main()
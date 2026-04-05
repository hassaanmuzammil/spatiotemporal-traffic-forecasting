import os
import torch


def train(
    model, 
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    epochs,
    ckpt_dir,
    print_every=100, 
    save_every=5,
    scheduler=None,
    mlflow_run=None,
):
    """
    Trains the model and saves checkpoints in the provided ckpt_dir.
    """
    if ckpt_dir is None:
        raise ValueError("ckpt_dir must be provided. Use create_run_ckpt_dir() to create one.")

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        # ── train ──
        model.train()
        total_loss = 0.0
        running_loss = 0.0

        for step, batch in enumerate(train_loader, 1):
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch), batch.y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            running_loss += loss.item()

            if step % print_every == 0:
                print(f"step {step:4d} | loss {running_loss / print_every:.4f}")
                running_loss = 0.0

        train_loss = total_loss / len(train_loader)

        # ── eval ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                val_loss += criterion(model(batch), batch.y).item()
        val_loss /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if scheduler is not None:
            scheduler.step(val_loss)
 
        # ── log to mlflow ──
        if mlflow_run is not None:
            import mlflow
            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss},
                step=epoch,
            )
 
        # ── save best model ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            ckpt_path = os.path.join(ckpt_dir, "best_model.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved new best model → {ckpt_path}")

        # ── save every N epochs ──
        if epoch % save_every == 0:
            ckpt_epoch_path = os.path.join(ckpt_dir, f"epoch_{epoch}.pt")
            torch.save(model.state_dict(), ckpt_epoch_path)
            print(f"Saved checkpoint for epoch {epoch} → {ckpt_epoch_path}")

        print(f"train loss: {train_loss:.4f}")
        print(f"val loss: {val_loss:.4f}  {'← best' if epoch == best_epoch else ''}")

    print(f"\nbest val loss: {best_val_loss:.4f} at epoch {best_epoch}")
 
    return train_losses, val_losses
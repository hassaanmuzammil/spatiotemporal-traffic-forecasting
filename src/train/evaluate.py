import torch


def test(model, test_loader, mean, std, device):
    all_preds, all_targets = [], []

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            all_preds.append(out.cpu())
            all_targets.append(batch.y.cpu())

    all_preds = torch.cat(all_preds,   dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # denormalize
    all_preds_denorm   = all_preds * std + mean
    all_targets_denorm = all_targets * std + mean

    # metrics
    mse  = torch.mean((all_preds_denorm - all_targets_denorm) ** 2)
    rmse = torch.sqrt(mse)
    mae  = torch.mean(torch.abs(all_preds_denorm - all_targets_denorm))

    print(f"rmse: {rmse:.4f}")
    print(f"mae: {mae:.4f}")

    return all_preds_denorm, all_targets_denorm
import os
from datetime import datetime
import matplotlib.pyplot as plt

from src.configs.config import CKPT_DIR

def create_run_dir():
    """Create a unique directory for this training run inside ckpts/."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(CKPT_DIR, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def plot_losses(train_losses, val_losses, save_dir: str = None):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "train.png"), dpi=150)

    plt.show()
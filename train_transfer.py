"""
DeepGuard AI - Model Training with Transfer Learning
===================================================
Fine-tunes ResNet18 on the video-partitioned deepfake dataset.
Key Features:
- Backbone: Pretrained ResNet18
- Classification Head: Dropout(0.5) + Linear(512 -> 2)
- Loss: CrossEntropyLoss
- Optimizer: Adam (lr=5e-4, weight_decay=1e-4)
- Learning Rate Scheduler: ReduceLROnPlateau
- Early Stopping: Prevents overfitting based on validation loss
- Checkpoint: Automatically saves best performing model to models/best_resnet18_deepfake.pth
- Metric Logging: Generates training curve plot at results/training_history.png
"""

import os
import time
import json
import argparse
import copy
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

from model_def import get_model, LABEL_MAP
from dataset_loader import get_dataloaders


def plot_and_save_history(
    history: Dict[str, List[float]],
    output_png: str = "results/training_history.png"
) -> None:
    """Plots training & validation loss and accuracy curves."""
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss", marker="o", color="#e74c3c")
    plt.plot(epochs, history["val_loss"], label="Val Loss", marker="s", color="#3498db")
    plt.title("Loss vs. Epochs", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [a * 100 for a in history["train_acc"]], label="Train Acc", marker="o", color="#2ecc71")
    plt.plot(epochs, [a * 100 for a in history["val_acc"]], label="Val Acc", marker="s", color="#9b59b6")
    plt.title("Accuracy vs. Epochs", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"[OK] Training curves saved to: {output_png}")


def train_model(
    dataset_dir: str = "dataset",
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    dropout: float = 0.5,
    patience: int = 5,
    save_path: str = "models/best_resnet18_deepfake.pth",
    results_dir: str = "results"
) -> Dict[str, List[float]]:
    """
    Executes the training and validation loop.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print(" DeepGuard AI - ResNet18 Transfer Learning Training Pipeline")
    print("=" * 65)
    print(f"[*] Compute Device:      {device}")
    print(f"[*] Dataset Directory:   {dataset_dir}")
    print(f"[*] Hyperparameters:     Epochs={epochs}, BatchSize={batch_size}, LR={lr}")
    print(f"[*] Regularization:      WeightDecay={weight_decay}, Dropout={dropout}")
    print(f"[*] Early Stopping:      Patience={patience} epochs")
    print(f"[*] Model Checkpoint:    {save_path}")

    # Load datasets
    dataloaders, dataset_sizes, class_names = get_dataloaders(
        dataset_dir=dataset_dir,
        batch_size=batch_size
    )

    print(f"[*] Dataset Splits:      Train={dataset_sizes['train']} images | Val={dataset_sizes['val']} images")
    if dataset_sizes["train"] == 0 or dataset_sizes["val"] == 0:
        raise ValueError("Cannot train with empty train or validation datasets.")

    # Initialize model
    model = get_model(num_classes=2, dropout_rate=dropout, pretrained=True)
    model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")
        epoch_metrics = {}

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            epoch_metrics[f"{phase}_loss"] = epoch_loss
            epoch_metrics[f"{phase}_acc"] = epoch_acc

        # Record history
        history["train_loss"].append(epoch_metrics["train_loss"])
        history["val_loss"].append(epoch_metrics["val_loss"])
        history["train_acc"].append(epoch_metrics["train_acc"])
        history["val_acc"].append(epoch_metrics["val_acc"])

        # Learning rate schedule update
        scheduler.step(epoch_metrics["val_loss"])

        val_loss = epoch_metrics["val_loss"]
        val_acc = epoch_metrics["val_acc"]
        train_loss = epoch_metrics["train_loss"]
        train_acc = epoch_metrics["train_acc"]

        status_flag = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            
            # Save checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": best_model_wts,
                "val_loss": best_val_loss,
                "val_acc": best_val_acc,
                "class_names": class_names,
                "dropout_rate": dropout,
                "architecture": "ResNet18"
            }, save_path)
            status_flag = " [BEST SAVED]"
        else:
            patience_counter += 1
            status_flag = f" [Patience: {patience_counter}/{patience}]"

        print(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:6.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:6.2f}%{status_flag}"
        )

        # Early stopping condition
        if patience_counter >= patience:
            print(f"\n[!] Early stopping triggered: Validation loss did not improve for {patience} consecutive epochs.")
            break

    elapsed = time.time() - start_time
    print("\n" + "=" * 65)
    print(f" Training Complete in {elapsed // 60:.0f}m {elapsed % 60:.0f}s")
    print(f" Best Val Loss:  {best_val_loss:.4f} at Epoch {best_epoch}")
    print(f" Best Val Acc:   {best_val_acc * 100:.2f}%")
    print(f" Model saved to: {os.path.abspath(save_path)}")
    print("=" * 65)

    # Save training curves & history JSON
    history_png = os.path.join(results_dir, "training_history.png")
    plot_and_save_history(history, history_png)

    history_json = os.path.join(results_dir, "training_history.json")
    with open(history_json, "w") as f:
        json.dump(history, f, indent=2)

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet18 for deepfake detection.")
    parser.add_argument("--dataset_dir", type=str, default="dataset", help="Dataset directory")
    parser.add_argument("--epochs", type=int, default=15, help="Maximum epochs (default: 15)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate (default: 5e-4)")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay (default: 1e-4)")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate (default: 0.5)")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience (default: 4)")
    parser.add_argument("--save_path", type=str, default="models/best_resnet18_deepfake.pth", help="Checkpoint save path")
    parser.add_argument("--results_dir", type=str, default="results", help="Results output directory")
    args = parser.parse_args()

    train_model(
        dataset_dir=args.dataset_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        patience=args.patience,
        save_path=args.save_path,
        results_dir=args.results_dir
    )

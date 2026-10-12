"""
DeepGuard AI - Training Entrypoint Alias
========================================
Delegates directly to train_transfer.py to support both:
    python train.py
and
    python train_transfer.py
"""

import sys
from train_transfer import train_model
import argparse

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

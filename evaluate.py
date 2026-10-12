"""
DeepGuard AI - Image-Level Model Evaluation
===========================================
Evaluates the trained ResNet18 model on the unseen test dataset.
Computes and reports:
- Overall Accuracy
- Precision, Recall, F1-Score (per class and macro)
- Confusion Matrix (Saved as image to results/confusion_matrix.png)
- Detailed metrics saved to results/metrics.txt

EXPLICIT CLASS LABELS:
    Class 0 = FAKE
    Class 1 = REAL
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

from model_def import load_model_checkpoint, LABEL_MAP
from dataset_loader import get_data_transforms


def plot_confusion_matrix_heatmap(
    cm: np.ndarray,
    output_png: str = "results/confusion_matrix.png"
) -> None:
    """Plots and saves an academic confusion matrix heatmap."""
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    
    classes = ["FAKE (0)", "REAL (1)"]
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.85)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > cm.max() / 2.0 else "black"
            ax.text(j, i, f"{val:d}", ha="center", va="center", color=color, fontsize=14, fontweight="bold")

    fig.colorbar(cax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_yticklabels(classes, fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("True / Ground Truth Label", fontsize=12, fontweight="bold")
    ax.set_title("Confusion Matrix - Test Dataset", fontsize=13, fontweight="bold", pad=15)

    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"[OK] Confusion matrix plot saved to: {output_png}")


def evaluate_test_set(
    dataset_dir: str = "dataset",
    model_path: str = "models/best_resnet18_deepfake.pth",
    results_dir: str = "results",
    batch_size: int = 16
) -> dict:
    """
    Evaluates the model on the test partition.
    """
    print("=" * 65)
    print(" DeepGuard AI - Comprehensive Model Evaluation")
    print("=" * 65)
    print(f"[*] Model Checkpoint:  {model_path}")
    print(f"[*] Test Dataset:      {os.path.join(dataset_dir, 'test')}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Inference Device:  {device}")

    test_dir = os.path.join(dataset_dir, "test")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test split directory not found: {test_dir}")

    # Load model
    model = load_model_checkpoint(model_path, device=device)
    model.eval()

    # Load test dataset
    transforms_dict = get_data_transforms()
    test_dataset = datasets.ImageFolder(test_dir, transform=transforms_dict["test"])
    
    if len(test_dataset) == 0:
        raise ValueError("Test dataset is empty. Cannot evaluate.")

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"[*] Test Samples:      {len(test_dataset)} images")
    print(f"[*] Class Index Map:   {test_dataset.class_to_idx} (0=fake, 1=real)")

    y_true = []
    y_pred = []
    y_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            y_probs.extend(probs.cpu().numpy().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    p_per_class, r_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # Print Report
    print("\n" + "=" * 65)
    print(f"{'Class':<12} | {'Precision':<12} | {'Recall':<12} | {'F1-Score':<12} | {'Support':<10}")
    print("-" * 65)
    print(f"{'FAKE (0)':<12} | {p_per_class[0]*100:10.2f}% | {r_per_class[0]*100:10.2f}% | {f1_per_class[0]*100:10.2f}% | {support_per_class[0]:<10}")
    print(f"{'REAL (1)':<12} | {p_per_class[1]*100:10.2f}% | {r_per_class[1]*100:10.2f}% | {f1_per_class[1]*100:10.2f}% | {support_per_class[1]:<10}")
    print("-" * 65)
    print(f"{'OVERALL ACCURACY':<12} : {acc * 100:.2f}%")
    print("=" * 65)

    print("\n[*] Confusion Matrix Breakdown:")
    print(f"    True Fakes correctly identified (TN): {cm[0, 0]}")
    print(f"    True Fakes misclassified as Real (FP): {cm[0, 1]}")
    print(f"    True Reals misclassified as Fake (FN): {cm[1, 0]}")
    print(f"    True Reals correctly identified (TP): {cm[1, 1]}")

    # Save metrics to text file
    os.makedirs(results_dir, exist_ok=True)
    metrics_txt = os.path.join(results_dir, "metrics.txt")
    with open(metrics_txt, "w") as f:
        f.write("DeepGuard AI - Test Evaluation Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Model: ResNet18 Transfer Learning\n")
        f.write(f"Overall Accuracy: {acc * 100:.2f}%\n\n")
        f.write(f"Class 0 (FAKE):\n")
        f.write(f"  Precision: {p_per_class[0] * 100:.2f}%\n")
        f.write(f"  Recall:    {r_per_class[0] * 100:.2f}%\n")
        f.write(f"  F1-Score:  {f1_per_class[0] * 100:.2f}%\n")
        f.write(f"  Support:   {support_per_class[0]}\n\n")
        f.write(f"Class 1 (REAL):\n")
        f.write(f"  Precision: {p_per_class[1] * 100:.2f}%\n")
        f.write(f"  Recall:    {r_per_class[1] * 100:.2f}%\n")
        f.write(f"  F1-Score:  {f1_per_class[1] * 100:.2f}%\n")
        f.write(f"  Support:   {support_per_class[1]}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"  [TN={cm[0, 0]}, FP={cm[0, 1]}]\n")
        f.write(f"  [FN={cm[1, 0]}, TP={cm[1, 1]}]\n\n")
        f.write("Full Classification Report:\n")
        f.write(classification_report(y_true, y_pred, target_names=["FAKE", "REAL"], zero_division=0))

    print(f"[OK] Metrics report saved to: {metrics_txt}")

    # Save confusion matrix plot
    cm_png = os.path.join(results_dir, "confusion_matrix.png")
    plot_confusion_matrix_heatmap(cm, cm_png)

    return {
        "accuracy": acc,
        "precision_fake": p_per_class[0],
        "recall_fake": r_per_class[0],
        "f1_fake": f1_per_class[0],
        "precision_real": p_per_class[1],
        "recall_real": r_per_class[1],
        "f1_real": f1_per_class[1],
        "confusion_matrix": cm.tolist()
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate deepfake detection model on test set.")
    parser.add_argument("--dataset_dir", type=str, default="dataset", help="Dataset directory")
    parser.add_argument("--model_path", type=str, default="models/best_resnet18_deepfake.pth", help="Model path")
    parser.add_argument("--results_dir", type=str, default="results", help="Results directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Evaluation batch size")
    args = parser.parse_args()

    evaluate_test_set(
        dataset_dir=args.dataset_dir,
        model_path=args.model_path,
        results_dir=args.results_dir,
        batch_size=args.batch_size
    )

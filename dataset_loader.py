"""
DeepGuard AI - PyTorch Dataset and DataLoaders
==============================================
Standardizes data loading, augmentation, and normalization for ResNet18.
Guarantees class mapping alignment:
    0 = FAKE
    1 = REAL

ImageNet Normalization:
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
Target Resolution:
    224 x 224 pixels
"""

import os
from typing import Dict, Tuple, Optional
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = (224, 224)


def get_data_transforms() -> Dict[str, transforms.Compose]:
    """
    Returns standard PyTorch transformation pipelines for train, val, and test splits.
    Train split uses gentle data augmentation to avoid overfitting.
    Val and Test splits apply deterministic resizing and normalization.
    """
    return {
        "train": transforms.Compose([
            transforms.Resize(INPUT_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ]),
        "val": transforms.Compose([
            transforms.Resize(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ]),
        "test": transforms.Compose([
            transforms.Resize(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    }


def get_inference_transform() -> transforms.Compose:
    """
    Returns transform pipeline used for real-time inference on single image/frame.
    """
    return transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_dataloaders(
    dataset_dir: str = "dataset",
    batch_size: int = 32,
    num_workers: int = 0
) -> Tuple[Dict[str, DataLoader], Dict[str, int], Dict[int, str]]:
    """
    Loads ImageFolder datasets and wraps them in PyTorch DataLoaders.
    Verifies that class indices match: 0 -> FAKE, 1 -> REAL.
    """
    data_transforms = get_data_transforms()
    image_datasets = {}
    dataloaders = {}
    dataset_sizes = {}
    
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Missing required dataset split: {split_dir}")

        dataset = datasets.ImageFolder(split_dir, transform=data_transforms[split])
        
        # Verify class index alignment
        expected_mapping = {"fake": 0, "real": 1}
        if dataset.class_to_idx != expected_mapping:
            raise ValueError(
                f"Unexpected class mapping in {split} split: {dataset.class_to_idx}. "
                f"Expected {expected_mapping} (0: fake, 1: real)."
            )

        image_datasets[split] = dataset
        dataset_sizes[split] = len(dataset)
        
        shuffle = True if split == "train" else False
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )

    class_names = {0: "FAKE", 1: "REAL"}
    return dataloaders, dataset_sizes, class_names


if __name__ == "__main__":
    print("[*] Testing dataset_loader.py transformations...")
    t = get_inference_transform()
    dummy_img = Image.new("RGB", (300, 300), color=(128, 128, 128))
    tensor = t(dummy_img)
    print(f"[*] Input image size:   (300, 300)")
    print(f"[*] Transformed tensor: {tensor.shape} (Expected: [3, 224, 224])")
    assert tensor.shape == (3, 224, 224), "Transform shape mismatch!"
    print("[OK] dataset_loader.py tests PASSED successfully!")

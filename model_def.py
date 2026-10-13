"""
DeepGuard AI - Model Architecture Definition
===========================================
Defines the transfer learning CNN architecture based on ResNet18.
Outputs 2 classes:
    Class 0: FAKE
    Class 1: REAL

Architecture:
    ResNet18 Backbone (Feature Extractor)
         ↓
    Dropout (p=0.5)
         ↓
    Linear (512 -> 2)
"""

import os
import pickle
import torch
import torch.nn as nn
import torchvision.models as models

# Canonical class index mapping (ImageFolder maps alphabetically: 'fake'=0, 'real'=1)
CLASS_NAMES = ["FAKE", "REAL"]
LABEL_MAP = {0: "FAKE", 1: "REAL"}
CLASS_TO_IDX = {"fake": 0, "real": 1}


class DeepfakeResNet18(nn.Module):
    """
    ResNet18 with modified classification head and dropout for binary deepfake detection.
    """
    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.5, pretrained: bool = True):
        super(DeepfakeResNet18, self).__init__()
        
        # Load weights
        if pretrained:
            weights = models.ResNet18_Weights.DEFAULT
            self.backbone = models.resnet18(weights=weights)
        else:
            self.backbone = models.resnet18(weights=None)
            
        num_features = self.backbone.fc.in_features  # 512 for ResNet18
        
        # Replace the final fully connected layer with Dropout + Linear
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def get_model(num_classes: int = 2, dropout_rate: float = 0.5, pretrained: bool = True) -> DeepfakeResNet18:
    """
    Factory function to instantiate the DeepfakeResNet18 model.
    """
    return DeepfakeResNet18(num_classes=num_classes, dropout_rate=dropout_rate, pretrained=pretrained)


def load_model_checkpoint(
    checkpoint_path: str,
    device: torch.device = None,
    dropout_rate: float = 0.5
) -> DeepfakeResNet18:
    """
    Load a trained model checkpoint safely.
    Ensures exact architectural match and loads weights onto the requested device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except (pickle.UnpicklingError, EOFError) as exc:
        raise ValueError(
            f"Invalid or unsupported model checkpoint format at: {checkpoint_path}"
        ) from exc

    if not isinstance(checkpoint, dict):
        raise ValueError(
            "Invalid checkpoint format: expected a state dictionary or a checkpoint "
            "dictionary containing 'model_state_dict'."
        )

    if "architecture" in checkpoint and checkpoint["architecture"] != "ResNet18":
        raise ValueError(
            "Incompatible checkpoint architecture: expected 'ResNet18', "
            f"got {checkpoint['architecture']!r}."
        )

    if "class_names" in checkpoint:
        class_names = checkpoint["class_names"]
        if not isinstance(class_names, dict):
            raise ValueError("Invalid checkpoint metadata: 'class_names' must be a dictionary.")
        try:
            normalized_class_names = {int(index): str(name).upper() for index, name in class_names.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid checkpoint metadata: 'class_names' must map class indexes to names.") from exc
        if normalized_class_names != LABEL_MAP:
            raise ValueError(
                "Incompatible checkpoint class mapping: expected {0: 'FAKE', 1: 'REAL'}, "
                f"got {normalized_class_names!r}."
            )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError(
            "Invalid checkpoint format: the model weights must be a non-empty state dictionary."
        )
    if not all(isinstance(value, torch.Tensor) for value in state_dict.values()):
        raise ValueError(
            "Invalid checkpoint format: the selected state dictionary contains non-tensor values."
        )

    checkpoint_dropout = checkpoint.get("dropout_rate", dropout_rate)
    if (
        not isinstance(checkpoint_dropout, (int, float))
        or isinstance(checkpoint_dropout, bool)
        or not 0 <= checkpoint_dropout < 1
    ):
        raise ValueError(
            f"Invalid checkpoint dropout_rate: expected a number in [0, 1), got {checkpoint_dropout!r}."
        )

    # Training saves DeepfakeResNet18 weights, whose keys are prefixed with "backbone.".
    model = get_model(num_classes=2, dropout_rate=float(checkpoint_dropout), pretrained=False)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint weights are incompatible with the trained DeepfakeResNet18 "
            "architecture (ResNet18 backbone, Dropout + 512-to-2 classifier). "
            "Ensure the deployed model_def.py matches the training code. "
            f"Details: {exc}"
        ) from exc

    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    print("[*] Testing DeepfakeResNet18 architecture definition...")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Target device: {dev}")
    
    test_model = get_model(num_classes=2, dropout_rate=0.5, pretrained=True)
    test_model.to(dev)
    test_model.eval()
    
    # Pass dummy batch of 2 RGB images (224x224)
    dummy_input = torch.randn(2, 3, 224, 224, device=dev)
    with torch.no_grad():
        outputs = test_model(dummy_input)
        probabilities = torch.softmax(outputs, dim=1)
        
    print(f"[*] Input shape:       {dummy_input.shape}")
    print(f"[*] Output logits:     {outputs.shape} (Expected: [2, 2])")
    print(f"[*] Softmax output:    {probabilities.shape}")
    print(f"[*] Class 0 is:        {LABEL_MAP[0]}")
    print(f"[*] Class 1 is:        {LABEL_MAP[1]}")
    assert outputs.shape == (2, 2), "Output shape mismatch!"
    print("[OK] Model architecture test PASSED successfully!")

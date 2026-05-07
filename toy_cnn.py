"""
Toy CNN for γ/hadron classification on ASTRI Mini-Array-like Cherenkov images.

Design choices (didactic, not research-grade):
- Input: single-channel 56x56 image (rough ASTRI camera grid surrogate)
- 3 conv blocks (16 → 32 → 64 channels), each with BatchNorm + ReLU + MaxPool
- Global average pooling → small MLP → 2-class logits

The architecture mirrors the canonical "small CNN" pattern: enough to be
recognisable as a real CNN, light enough to do CPU inference in seconds
on 1000s of events. Total params ~50k, file size <1 MB.

Why 56x56 and not the real ASTRI hex grid? Because the lecture is about
ML concepts, not pixel mapping. A square grid keeps the code one-line per layer
and removes the distraction of hexagonal convolutions.
"""

import torch
import torch.nn as nn


class ToyCherenkovCNN(nn.Module):
    def __init__(self, n_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 56 -> 28
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2: 28 -> 14
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3: 14 -> 7
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)


def build_and_save(out_path: str = "toy_cnn.pt", seed: int = 42):
    """Build the model with a deterministic seed and save state_dict + arch."""
    torch.manual_seed(seed)
    model = ToyCherenkovCNN(n_classes=2)
    n_params = sum(p.numel() for p in model.parameters())

    # Smoke-test with a synthetic batch
    x = torch.randn(4, 1, 56, 56)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (4, 2), f"Unexpected output shape {y.shape}"

    # Save in a self-contained format: state_dict + meta
    torch.save(
        {
            "state_dict": model.state_dict(),
            "arch": "ToyCherenkovCNN",
            "input_shape": [1, 56, 56],
            "n_classes": 2,
            "n_params": n_params,
        },
        out_path,
    )
    return n_params


if __name__ == "__main__":
    n = build_and_save("/home/claude/lecture/toy_cnn.pt")
    import os
    size_kb = os.path.getsize("/home/claude/lecture/toy_cnn.pt") / 1024
    print(f"saved · {n:,} parameters · {size_kb:.1f} KB on disk")

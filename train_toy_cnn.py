"""
Train the ToyCherenkovCNN on the synthetic dataset and resave with trained weights.
This way `toy_cnn.pt` distributed with the lecture contains a real, working classifier.
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from toy_cnn import ToyCherenkovCNN
from synth_camera import generate_dataset


def main():
    print("Generating training dataset...")
    X_train, y_train = generate_dataset(n_per_class=2000, seed=42)
    print("Generating validation dataset...")
    X_val, y_val = generate_dataset(n_per_class=500, seed=123)

    print(f"  train: X{X_train.shape}  y{y_train.shape}")
    print(f"  val:   X{X_val.shape}  y{y_val.shape}")

    # Datasets and loaders
    Xt = torch.from_numpy(X_train)
    yt = torch.from_numpy(y_train)
    Xv = torch.from_numpy(X_val)
    yv = torch.from_numpy(y_val)

    train_ds = TensorDataset(Xt, yt)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)

    # Model + optimisation
    torch.manual_seed(42)
    model = ToyCherenkovCNN(n_classes=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params:,} parameters")

    # Train enough epochs for BatchNorm running stats to converge.
    # Synthetic problem is easy by design — accuracy will be very high.
    # The lecture honestly tells students this is the "MC bubble":
    # great on simulation, but real-data performance is the open question (slide 44).
    n_epochs = 5
    print(f"\nTraining for {n_epochs} epochs (CPU)...")
    t0 = time.time()
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            running_loss += loss.item()
            n_batches += 1

        # Validation in batches (avoids holding the full val tensor on the graph)
        model.eval()
        val_correct = 0
        val_total = 0
        val_loader = DataLoader(TensorDataset(Xv, yv), batch_size=128, shuffle=False)
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb).argmax(dim=1)
                val_correct += (preds == yb).sum().item()
                val_total += len(yb)
        val_acc = val_correct / val_total
        print(f"  epoch {epoch+1}/{n_epochs}  loss={running_loss/n_batches:.4f}  val_acc={val_acc:.4f}")

    elapsed = time.time() - t0
    print(f"\nTraining time: {elapsed:.1f}s")

    # Save the trained checkpoint
    torch.save({
        "state_dict": model.state_dict(),
        "arch": "ToyCherenkovCNN",
        "input_shape": [1, 56, 56],
        "n_classes": 2,
        "n_params": n_params,
        "training_data": "synthetic Cherenkov-like images (synth_camera.py)",
        "val_accuracy": val_acc,
    }, "toy_cnn.pt")

    print(f"\nSaved trained model -> toy_cnn.pt  (val_acc={val_acc:.4f})")


if __name__ == "__main__":
    main()

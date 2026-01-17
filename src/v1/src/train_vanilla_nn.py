#!/usr/bin/env python
"""Train a simple vanilla neural network on the DIG 4 Bio Raman dataset.

What this script does
---------------------
1) Loads the 8 device training CSVs (anton_532, ..., tornado) and interpolates each spectrum
   onto a shared wavenumber grid (intersection across devices; default floor/ceil 300..1942).
2) Loads transfer_plate.csv (labels available) for validation.
3) Trains a vanilla MLP (no PINN, no domain adaptation) to predict 3 concentrations.
4) Generates a Kaggle-style submission for 96_samples.csv.

Run (example)
-------------
python train_vanilla_nn.py \
  --data_dir /path/to/dig-4-bio-raman-transfer-learning-challenge \
  --epochs 60 --batch_size 64 --lr 1e-3 \
  --out_dir ./outputs

Outputs
-------
- <out_dir>/model.pt
- <out_dir>/submission.csv
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from raman_light.data import (
    DEVICE_FILES,
    infer_shared_grid,
    load_device_training,
    load_test_plate,
    load_transfer_plate,
)
from raman_light.model import VanillaMLP


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


@torch.no_grad()
def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Compute R^2 per target column."""
    # y_true/y_pred: (N, 3)
    ss_res = torch.sum((y_true - y_pred) ** 2, dim=0)
    mean = torch.mean(y_true, dim=0, keepdim=True)
    ss_tot = torch.sum((y_true - mean) ** 2, dim=0)
    return 1.0 - (ss_res / (ss_tot + 1e-12))


def train_one_epoch(model, loader, opt, device):
    model.train()
    loss_fn = nn.MSELoss()
    total = 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        opt.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        opt.step()
        total += float(loss.item()) * xb.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, X, y, device):
    model.eval()
    X = X.to(device)
    y = y.to(device)
    pred = model(X)
    mse = torch.mean((pred - y) ** 2).item()
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y, pred).cpu().numpy()
    return rmse, r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Directory containing the CSV files")
    ap.add_argument("--out_dir", default="./outputs", help="Where to write outputs")

    ap.add_argument("--wn_floor", type=float, default=300.0, help="Min wavenumber for shared grid")
    ap.add_argument("--wn_ceil", type=float, default=1942.0, help="Max wavenumber for shared grid")
    ap.add_argument("--wn_step", type=float, default=1.0, help="Step size for shared grid")

    ap.add_argument("--plate_wn_start", type=float, default=65.0)
    ap.add_argument("--plate_wn_end", type=float, default=3350.0)

    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1) Grid + datasets
    grid_wns = infer_shared_grid(
        args.data_dir,
        device_files=DEVICE_FILES,
        wn_min_floor=args.wn_floor,
        wn_max_ceil=args.wn_ceil,
        step=args.wn_step,
    )
    print("Shared grid:", float(grid_wns.min()), "..", float(grid_wns.max()), "len=", len(grid_wns))

    train_ds = load_device_training(args.data_dir, grid_wns, device_files=DEVICE_FILES)
    val_ds = load_transfer_plate(
        args.data_dir,
        grid_wns,
        plate_filename="transfer_plate.csv",
        plate_wn_start=args.plate_wn_start,
        plate_wn_end=args.plate_wn_end,
    )
    test_ds = load_test_plate(
        args.data_dir,
        grid_wns,
        test_filename="96_samples.csv",
        plate_wn_start=args.plate_wn_start,
        plate_wn_end=args.plate_wn_end,
    )

    print("Train:", train_ds.X.shape, train_ds.y.shape)
    print("Val  :", val_ds.X.shape, val_ds.y.shape)
    print("Test :", test_ds.X.shape)

    # 2) Torch tensors
    X_train = torch.tensor(train_ds.X, dtype=torch.float32)
    y_train = torch.tensor(train_ds.y, dtype=torch.float32)
    X_val = torch.tensor(val_ds.X, dtype=torch.float32)
    y_val = torch.tensor(val_ds.y, dtype=torch.float32)
    X_test = torch.tensor(test_ds.X, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    # 3) Model
    model = VanillaMLP(
        input_dim=X_train.shape[1],
        hidden_dims=(512, 256, 128),
        dropout=args.dropout,
        non_negative=True,
    ).to(device)

    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_rmse = 1e9
    best_state = None

    # 4) Train
    for ep in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, opt, device)
        val_rmse, val_r2 = evaluate(model, X_val, y_val, device)

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if ep == 1 or ep % 10 == 0:
            print(
                f"epoch={ep:03d} train_mse={tr_loss:.6f} val_rmse={val_rmse:.4f} "
                f"val_r2=[{val_r2[0]:.3f}, {val_r2[1]:.3f}, {val_r2[2]:.3f}]"
            )

    print("Best val RMSE:", best_rmse)
    if best_state is not None:
        model.load_state_dict(best_state)

    # 5) Save model + metadata
    payload = {
        "model_state": model.state_dict(),
        "grid_wns": grid_wns,
        "args": vars(args),
    }
    model_path = os.path.join(args.out_dir, "model.pt")
    torch.save(payload, model_path)
    print("Saved model:", model_path)

    # 6) Predict test + write submission (use sample_submission.csv as schema)
    sub_path = os.path.join(args.data_dir, "sample_submission.csv")
    sub = pd.read_csv(sub_path)

    model.eval()
    with torch.no_grad():
        pred = model(X_test.to(device)).cpu().numpy()

    # Ensure ID column matches
    # Kaggle expects IDs 1..96 (from sample_submission) in the right order.
    # We'll keep it exactly as in sample_submission.csv.
    out = sub.copy()
    out.loc[:, out.columns[1:]] = pred[: len(out)]

    out_file = os.path.join(args.out_dir, "submission.csv")
    out.to_csv(out_file, index=False)
    print("Wrote:", out_file, "shape=", out.shape)


if __name__ == "__main__":
    main()

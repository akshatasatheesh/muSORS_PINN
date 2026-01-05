# ============================================================
# Kaggle DIG-4-BIO Raman Transfer Learning Challenge
# Improved v2: device-aware training, spectroscopy preprocessing,
# 1D CNN + physics (mixture) decoder, optional CORAL alignment,
# safer submission formatting.
# ============================================================

import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Optional (usually available on Kaggle). If not, we fall back gracefully.
try:
    from scipy.signal import savgol_filter
    HAS_SAVGOL = True
except Exception:
    HAS_SAVGOL = False

# ----------------------------
# 0) Paths / Config
# ----------------------------
DATA_PATH = "/kaggle/input/dig-4-bio-raman-transfer-learning-challenge"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Training devices (8)
DATASET_NAMES = [
    "anton_532", "anton_785", "kaiser", "mettler_toledo",
    "metrohm", "tec5", "timegate", "tornado"
]

# Bounds per device (from the public notebook)
LOWER_BOUNDS = {
    "anton_532": 200, "anton_785": 100, "kaiser": -37, "mettler_toledo": 300,
    "metrohm": 200, "tec5": 85, "timegate": 200, "tornado": 300,
}
UPPER_BOUNDS = {
    "anton_532": 3500, "anton_785": 2300, "kaiser": 1942, "mettler_toledo": 3350,
    "metrohm": 3350, "tec5": 3210, "timegate": 2000, "tornado": 3300,
}

# Global joint range (align all devices)
JOINT_LOWER_WN = 300
JOINT_UPPER_WN = 1942

# ----------------------------
# 1) Data loading (robust)
# ----------------------------
def get_csv_dataset(dataset_name, lower_wn=-1000, upper_wn=10000, dtype=np.float64):
    lower_wn = max(lower_wn, LOWER_BOUNDS[dataset_name])
    upper_wn = min(upper_wn, UPPER_BOUNDS[dataset_name])

    df = pd.read_csv(os.path.join(DATA_PATH, f"{dataset_name}.csv"))

    # Spectral columns are all except last 5 columns (labels + fold)
    spectral_headers = df.columns[:-5]
    spectral_vals = np.array([float(x) for x in spectral_headers])

    spectra_selection = np.logical_and(lower_wn <= spectral_vals, spectral_vals <= upper_wn)

    spectra = df.iloc[:, :-5].iloc[:, spectra_selection].values
    label = df.iloc[:, -5:-1].values         # last 5: [targets...], fold is last
    cv_indices = df.iloc[:, -1].values

    all_indices = np.arange(len(cv_indices))
    folds = sorted(list(set(cv_indices)))
    cv_folds = [
        (all_indices[cv_indices != fold_idx], all_indices[cv_indices == fold_idx])
        for fold_idx in folds
    ]

    wavenumbers = spectral_vals[spectra_selection]

    return (
        spectra.astype(dtype),
        label.astype(dtype),
        cv_folds,
        wavenumbers.astype(dtype),
    )

def load_joint_dataset(dataset_names, lower_wn=-1000, upper_wn=10000, dtype=np.float64):
    dtype = dtype or np.float64

    lower_wn = max(lower_wn, *[LOWER_BOUNDS[name] for name in dataset_names])
    upper_wn = min(upper_wn, *[UPPER_BOUNDS[name] for name in dataset_names])

    # Intersect with our desired joint range
    lower_wn = max(lower_wn, JOINT_LOWER_WN)
    upper_wn = min(upper_wn, JOINT_UPPER_WN)

    datasets = [
        get_csv_dataset(name, lower_wn=lower_wn, upper_wn=upper_wn, dtype=dtype)
        for name in dataset_names
    ]

    joint_wns = np.arange(lower_wn, upper_wn + 1)

    # Interpolate each device spectra onto the same wavenumber grid
    interpolated = []
    labels = []
    all_cv = []
    offsets = [0]
    for spectra, label, cv_folds, wns in datasets:
        interp_spectra = np.array([
            np.interp(joint_wns, xp=wns, fp=row) for row in spectra
        ])
        interpolated.append(interp_spectra)
        labels.append(label[:, :3])  # first 3 targets used in the public notebook
        all_cv.append(cv_folds)
        offsets.append(offsets[-1] + len(interp_spectra))

    X = np.concatenate(interpolated, axis=0)
    y = np.concatenate(labels, axis=0)

    # Build combined CV folds (device-provided folds, offset-adjusted)
    dataset_offsets = np.array(offsets[:-1], dtype=np.int64)
    fold_val_indices = []
    for (spectra, label, cv_folds, wns), offset in zip(datasets, dataset_offsets):
        for tr, va in cv_folds:
            fold_val_indices.append(va + offset)

    all_indices = set(range(len(X)))
    cv_folds_joint = [
        (np.array(list(all_indices - set(val_idx))), val_idx)
        for val_idx in fold_val_indices
    ]

    return X.astype(np.float64), y.astype(np.float64), cv_folds_joint, joint_wns.astype(np.float64)

def load_comp_data(filepath, is_train=True):
    """
    transfer_plate.csv (train=True) has headers and target columns.
    96_samples.csv (train=False) has no header and includes sample_id rows.
    """
    if is_train:
        df = pd.read_csv(filepath)
        target_cols = ['Glucose (g/L)', 'Sodium Acetate (g/L)', 'Magnesium Acetate (g/L)']
        y = df[target_cols].dropna().values
        X = df.iloc[:, :-4]  # remove last 4 cols (targets + analyte info)
    else:
        df = pd.read_csv(filepath, header=None)
        X = df
        y = None

    # Set column names: sample_id + spectral columns
    X.columns = ["sample_id"] + [str(i) for i in range(X.shape[1] - 1)]

    # Fill & clean sample_id
    X["sample_id"] = X["sample_id"].ffill()
    if is_train:
        X["sample_id"] = X["sample_id"].astype(str).str.strip()
    else:
        X["sample_id"] = X["sample_id"].astype(str).str.strip().str.replace("sample", "", regex=False).astype(int)

    # Clean spectral data (remove brackets; force numeric)
    spectral_cols = X.columns[1:]
    for col in spectral_cols:
        X[col] = (
            X[col].astype(str)
            .str.replace("[", "", regex=False)
            .str.replace("]", "", regex=False)
        )
        X[col] = pd.to_numeric(X[col], errors="coerce")

    return X, y

def fix_val_test_shape(X_2x2048):
    """
    transfer/test files come as 2 spectra per sample at 2048 points.
    We average the two spectra, then interpolate to joint_wns length (1643).
    """
    lower_wns = JOINT_LOWER_WN
    upper_wns = JOINT_UPPER_WN
    joint_wns = np.arange(lower_wns, upper_wns + 1)

    spectral_values = np.linspace(65, 3350, 2048)  # as in public notebook
    mask = np.logical_and(lower_wns <= spectral_values, spectral_values <= upper_wns)
    wns = spectral_values[mask]

    X = X_2x2048[:, mask]
    X = np.array([np.interp(joint_wns, xp=wns, fp=row) for row in X])
    return X

# ----------------------------
# 2) Spectroscopy preprocessing
# ----------------------------
def preprocess_spectra(X, use_savgol=True, deriv_order=1, snv=True):
    """
    X: (N, L) float array
    - SavGol smooth
    - Derivative (1st)
    - SNV per spectrum (mean 0, std 1)
    """
    Xp = X.copy()

    if use_savgol and HAS_SAVGOL:
        # window length must be odd and <= L
        L = Xp.shape[1]
        win = min(31, L - (1 - L % 2))  # keep odd
        if win < 7:
            win = 7 if L >= 7 else (L // 2 * 2 + 1)
        poly = 3 if win >= 7 else 2
        Xp = savgol_filter(Xp, window_length=win, polyorder=poly, axis=1)

    if deriv_order == 1:
        Xp = np.gradient(Xp, axis=1)
    elif deriv_order == 2:
        Xp = np.gradient(np.gradient(Xp, axis=1), axis=1)

    if snv:
        mu = Xp.mean(axis=1, keepdims=True)
        sd = Xp.std(axis=1, keepdims=True) + 1e-8
        Xp = (Xp - mu) / sd

    return Xp

# ----------------------------
# 3) Model: 1D CNN encoder + physics mixing decoder
# ----------------------------
class CNNMixPINN(nn.Module):
    """
    Encoder predicts concentrations (3).
    Decoder reconstructs spectra as: c @ fingerprints + smooth baseline
    """
    def __init__(self, L, wns, num_components=3, emb_dim=128, baseline_degree=3):
        super().__init__()
        self.L = L
        self.num_components = num_components

        # Normalize wavenumbers for baseline polynomial
        wns = torch.tensor(wns, dtype=torch.float32)
        wns_norm = (wns - wns.mean()) / (wns.std() + 1e-8)
        self.register_buffer("wns_norm", wns_norm)

        # 1D CNN encoder
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # compute conv output size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, L)
            out = self.conv(dummy)
            flat_dim = out.view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(flat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, emb_dim),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(emb_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_components),
            nn.Softplus(),  # nonnegative concentrations
        )

        # Fingerprints: parameterize as softplus(raw) to ensure nonnegative
        self.raw_fingerprints = nn.Parameter(torch.randn(num_components, L) * 0.01)

        # Smooth baseline polynomial coefficients
        # baseline(x) = sum_{k=0..d} a_k * (wn_norm^k)
        self.baseline_degree = baseline_degree
        self.baseline_coeff = nn.Parameter(torch.zeros(baseline_degree + 1))

    def encode(self, x):
        # x: (B, L)
        z = self.conv(x.unsqueeze(1))
        z = z.view(z.size(0), -1)
        emb = self.fc(z)
        c = self.head(emb)
        return c, emb

    def decode(self, c):
        # c: (B, C)
        fingerprints = F.softplus(self.raw_fingerprints)  # (C, L), nonnegative
        recon = c @ fingerprints  # (B, L)

        # baseline vector (L,)
        powers = torch.stack([self.wns_norm ** k for k in range(self.baseline_degree + 1)], dim=0)  # (d+1, L)
        baseline = (self.baseline_coeff.view(-1, 1) * powers).sum(dim=0)  # (L,)
        recon = recon + baseline.unsqueeze(0)
        return recon

    def forward(self, x):
        c, emb = self.encode(x)
        recon = self.decode(c)
        return c, recon, emb

# ----------------------------
# 4) Loss helpers (smoothness + CORAL)
# ----------------------------
def second_derivative_smoothness(fp):
    # fp: (C, L)
    d2 = fp[:, 2:] - 2 * fp[:, 1:-1] + fp[:, :-2]
    return torch.mean(d2 ** 2)

def coral_loss(source, target):
    """
    CORAL loss between feature covariances.
    source/target: (B, D)
    """
    def cov(x):
        x = x - x.mean(dim=0, keepdim=True)
        return (x.t() @ x) / (x.size(0) - 1 + 1e-8)

    Cs = cov(source)
    Ct = cov(target)
    return torch.mean((Cs - Ct) ** 2)

# ----------------------------
# 5) Train / Eval
# ----------------------------
@torch.no_grad()
def eval_rmse(model, X, y, batch_size=256):
    model.eval()
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    se = 0.0
    n = 0
    for xb, yb in dl:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        pred_c, _, _ = model(xb)
        se += torch.sum((pred_c - yb) ** 2).item()
        n += yb.numel()
    mse = se / max(n, 1)
    return float(np.sqrt(mse))

def train_one_fold(
    X_train, y_train,
    X_val, y_val,
    wns,
    epochs=30,
    batch_size=64,
    lr=2e-3,
    wd=1e-4,
    w_recon=0.3,
    w_smooth=0.05,
    w_fp_l1=1e-5,
    w_coral=0.0,
    X_target_for_coral=None,
    patience=6,
):
    L = X_train.shape[1]
    model = CNNMixPINN(L=L, wns=wns, num_components=3).to(DEVICE)

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_conc_fn = nn.SmoothL1Loss()  # robust
    loss_recon_fn = nn.MSELoss()

    train_dl = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True, drop_last=True)

    target_dl = None
    if w_coral > 0 and X_target_for_coral is not None:
        target_dl = DataLoader(TensorDataset(X_target_for_coral), batch_size=batch_size, shuffle=True, drop_last=True)
        target_it = iter(target_dl)

    best = {"rmse": 1e9, "state": None}
    bad = 0

    for ep in range(1, epochs + 1):
        model.train()
        for (xb, yb) in train_dl:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            opt.zero_grad()
            pred_c, recon, emb_s = model(xb)

            # Losses
            loss_conc = loss_conc_fn(pred_c, yb)
            loss_recon = loss_recon_fn(recon, xb)

            fp = F.softplus(model.raw_fingerprints)
            loss_smooth = second_derivative_smoothness(fp)
            loss_l1 = torch.mean(torch.abs(fp))

            loss = loss_conc + (w_recon * loss_recon) + (w_smooth * loss_smooth) + (w_fp_l1 * loss_l1)

            # Optional CORAL alignment to transfer device spectra
            if target_dl is not None:
                try:
                    (xt,) = next(target_it)
                except StopIteration:
                    target_it = iter(target_dl)
                    (xt,) = next(target_it)
                xt = xt.to(DEVICE)
                _, _, emb_t = model(xt)
                loss = loss + w_coral * coral_loss(emb_s, emb_t)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        val_rmse = eval_rmse(model, X_val, y_val)
        print(f"Epoch {ep:02d} | val RMSE: {val_rmse:.5f}")

        if val_rmse < best["rmse"] - 1e-5:
            best["rmse"] = val_rmse
            best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping at epoch {ep}. Best RMSE={best['rmse']:.5f}")
                break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return model, best["rmse"]

def fit_full_then_finetune(
    X_train, y_train, wns,
    X_transfer, y_transfer,
    epochs_pre=40,
    epochs_ft=15,
):
    # Pretrain using a simple split (small holdout from train for stability)
    n = X_train.shape[0]
    idx = np.arange(n)
    np.random.shuffle(idx)
    cut = int(0.9 * n)
    tr, va = idx[:cut], idx[cut:]

    model, rmse = train_one_fold(
        X_train[tr], y_train[tr],
        X_train[va], y_train[va],
        wns=wns,
        epochs=epochs_pre,
        batch_size=64,
        lr=2e-3,
        wd=1e-4,
        w_recon=0.3,
        w_smooth=0.05,
        w_fp_l1=1e-5,
        # Use CORAL to nudge embeddings toward transfer device
        w_coral=0.05,
        X_target_for_coral=X_transfer,
        patience=8,
    )
    print("Pretrain RMSE (holdout):", rmse)

    # Fine-tune on transfer_plate labels (lower LR)
    model.train()
    opt = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()
    dl = DataLoader(TensorDataset(X_transfer, y_transfer), batch_size=32, shuffle=True)

    for ep in range(1, epochs_ft + 1):
        for xb, yb in dl:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            opt.zero_grad()
            pred_c, recon, _ = model(xb)

            # fine-tune mainly on concentration accuracy; keep some recon regularization
            loss = loss_fn(pred_c, yb) + 0.15 * F.mse_loss(recon, xb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        ft_rmse = eval_rmse(model, X_transfer, y_transfer)
        print(f"FineTune Epoch {ep:02d} | transfer RMSE: {ft_rmse:.5f}")

    return model

# ----------------------------
# 6) Main: load, preprocess, train, predict, submit
# ----------------------------
def main():
    print("Using device:", DEVICE)

    # A) Load joint device training set
    X_train_raw, y_train_raw, cv_folds, joint_wns = load_joint_dataset(DATASET_NAMES)
    print("Train (raw):", X_train_raw.shape, y_train_raw.shape)

    # B) Load transfer_plate (labeled) + test (unlabeled)
    X_val_raw_df, y_val = load_comp_data(os.path.join(DATA_PATH, "transfer_plate.csv"), is_train=True)
    X_test_raw_df, _ = load_comp_data(os.path.join(DATA_PATH, "96_samples.csv"), is_train=False)

    # Average the two spectra per sample (reshape: -1, 2, 2048)
    X_val_2x2048 = X_val_raw_df.drop("sample_id", axis=1).values.reshape(-1, 2, 2048).mean(axis=1)
    X_test_2x2048 = X_test_raw_df.drop("sample_id", axis=1).values.reshape(-1, 2, 2048).mean(axis=1)

    # Fix shape to (N, 1643)
    X_val_raw = fix_val_test_shape(X_val_2x2048)
    X_test_raw = fix_val_test_shape(X_test_2x2048)

    print("Transfer/test (raw):", X_val_raw.shape, X_test_raw.shape)

    # C) Preprocess spectra (recommended)
    X_train = preprocess_spectra(X_train_raw, use_savgol=True, deriv_order=1, snv=True).astype(np.float32)
    X_val = preprocess_spectra(X_val_raw, use_savgol=True, deriv_order=1, snv=True).astype(np.float32)
    X_test = preprocess_spectra(X_test_raw, use_savgol=True, deriv_order=1, snv=True).astype(np.float32)

    # D) Scale targets (helps stability) using train stats
    y_train = y_train_raw.astype(np.float32)
    y_val = y_val.astype(np.float32)

    y_mu = y_train.mean(axis=0, keepdims=True)
    y_sd = y_train.std(axis=0, keepdims=True) + 1e-8
    y_train_s = (y_train - y_mu) / y_sd
    y_val_s = (y_val - y_mu) / y_sd

    # E) Torch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_s, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val_s, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    # F) Quick CV on the device-based folds (optional but recommended)
    # This can be slower; comment out if you only want full+finetune.
    print("\n=== Device-based CV (quick check) ===")
    cv_rmse = []
    for i, (tr_idx, va_idx) in enumerate(cv_folds[:5]):  # limit to first 5 folds for speed
        model, rmse = train_one_fold(
            X_train_t[tr_idx], y_train_t[tr_idx],
            X_train_t[va_idx], y_train_t[va_idx],
            wns=joint_wns,
            epochs=20,
            batch_size=64,
            lr=2e-3,
            wd=1e-4,
            w_recon=0.3,
            w_smooth=0.05,
            w_fp_l1=1e-5,
            w_coral=0.02,
            X_target_for_coral=X_val_t,  # align to transfer device
            patience=5,
        )
        print(f"Fold {i+1} RMSE (scaled targets): {rmse:.5f}")
        cv_rmse.append(rmse)

    if cv_rmse:
        print("Mean CV RMSE (scaled):", float(np.mean(cv_rmse)))

    # G) Train full model + fine-tune on transfer_plate
    print("\n=== Full train + fine-tune on transfer_plate ===")
    model = fit_full_then_finetune(
        X_train_t, y_train_t, joint_wns,
        X_val_t, y_val_t,
        epochs_pre=45,
        epochs_ft=15,
    )
    model.eval()

    # H) Predict on test, unscale targets
    with torch.no_grad():
        pred_s, _, _ = model(X_test_t.to(DEVICE))
        pred_s = pred_s.cpu().numpy()
    pred = pred_s * y_sd + y_mu  # back to original units

    # I) Build submission (try to match sample_submission if present)
    sample_sub_path = os.path.join(DATA_PATH, "sample_submission.csv")
    if os.path.exists(sample_sub_path):
        sub = pd.read_csv(sample_sub_path)
        # Attempt to fill by known target columns
        target_cols = ['Glucose (g/L)', 'Sodium Acetate (g/L)', 'Magnesium Acetate (g/L)']
        for j, col in enumerate(target_cols):
            if col in sub.columns:
                sub[col] = pred[:, j]
        sub.to_csv("submission.csv", index=False)
    else:
        # Fallback: just output the 3 target columns
        sub = pd.DataFrame(
            pred,
            columns=['Glucose (g/L)', 'Sodium Acetate (g/L)', 'Magnesium Acetate (g/L)']
        )
        sub.to_csv("submission.csv", index=False)

    print("\nWrote: submission.csv")

if __name__ == "__main__":
    main()

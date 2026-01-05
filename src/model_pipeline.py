# Updated end-to-end PINN-style pipeline for the DIG 4 Bio Raman Transfer Learning Challenge
# - Loads the 8 device training CSVs, interpolates to a shared wavenumber grid (300–1942 => 1643 features)
# - Loads transfer_plate (val) and 96_samples (test) in the expected format (2 replicates averaged)
# - Trains a physics-regularized model:
#     spectrum ≈ c @ fingerprints + polynomial_baseline
#   with non-negativity + smoothness constraints
# - Uses Leave-One-Device-Out (LODO) CV on the 8 training devices
# - Trains a final model on all training devices and writes submission.csv

import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# -------------------------
# 0) CONFIG
# -------------------------
DATA_PATH = "/kaggle/input/dig-4-bio-raman-transfer-learning-challenge"

TRAIN_DEVICE_NAMES = [
    "anton_532", "anton_785", "kaiser", "mettler_toledo",
    "metrohm", "tec5", "timegate", "tornado"
]

# Device-specific bounds from the public notebook (intersection will be used)
LOWER_BOUNDS = {
    "anton_532": 200, "anton_785": 100, "kaiser": -37, "mettler_toledo": 300,
    "metrohm": 200, "tec5": 85, "timegate": 200, "tornado": 300,
}
UPPER_BOUNDS = {
    "anton_532": 3500, "anton_785": 2300, "kaiser": 1942, "mettler_toledo": 3350,
    "metrohm": 3350, "tec5": 3210, "timegate": 2000, "tornado": 3300,
}

TARGET_COLS = ["Glucose (g/L)", "Sodium Acetate (g/L)", "Magnesium Acetate (g/L)"]

SEED = 42
BATCH_SIZE = 64
EPOCHS_CV = 40
EPOCHS_FINAL = 60
LR = 1e-3

# Loss weights (tune these)
W_RECON = 0.25
W_SMOOTH = 0.05
W_L1C = 1e-4

BASELINE_DEGREE = 3  # polynomial baseline degree

torch.manual_seed(SEED)
np.random.seed(SEED)


# -------------------------
# 1) TRAIN DEVICES LOADER (8 CSVs -> shared grid)
# -------------------------
def _safe_float_cols(cols):
    out = []
    for c in cols:
        try:
            out.append(float(c))
        except Exception:
            out.append(np.nan)
    return np.array(out, dtype=np.float64)

def get_device_dataset(device_name: str, lower_wn: int, upper_wn: int):
    """
    Reads <device_name>.csv.
    Assumes last 5 columns are metadata/labels:
      - last 5:-1 are labels (often 4 cols in source), last is fold index.
    We'll keep first 3 label dims for (glucose, sodium acetate, magnesium acetate).
    """
    path = os.path.join(DATA_PATH, f"{device_name}.csv")
    df = pd.read_csv(path)

    # spectral columns are everything except the last 5 columns (per the public notebook)
    spectral_df = df.iloc[:, :-5]
    label_df = df.iloc[:, -5:-1]
    cv_indices = df.iloc[:, -1].values

    wn_all = _safe_float_cols(spectral_df.columns)
    # Filter to valid wn and within bounds
    mask = np.isfinite(wn_all) & (wn_all >= lower_wn) & (wn_all <= upper_wn)

    spectra = spectral_df.loc[:, mask].values.astype(np.float64)
    wns = wn_all[mask].astype(np.float64)

    # keep first 3 targets (matches TARGET_COLS order in transfer_plate)
    labels = label_df.values.astype(np.float64)[:, :3]

    return spectra, labels, cv_indices, wns

def load_joint_train_dataset(device_names):
    # Shared intersection bounds so all devices can be interpolated to a common grid
    lower_wn = max(300, *[LOWER_BOUNDS[n] for n in device_names])
    upper_wn = min(1942, *[UPPER_BOUNDS[n] for n in device_names])

    joint_wns = np.arange(lower_wn, upper_wn + 1, dtype=np.float64)  # 300..1942 => 1643
    all_X = []
    all_y = []
    device_ids = []  # for LODO CV

    for did, name in enumerate(device_names):
        spectra, labels, _, wns = get_device_dataset(name, lower_wn, upper_wn)

        # Interpolate each spectrum to joint_wns
        X_interp = np.array([np.interp(joint_wns, xp=wns, fp=row) for row in spectra], dtype=np.float64)

        # Per-spectrum normalization (more stable across devices)
        denom = np.maximum(np.max(X_interp, axis=1, keepdims=True), 1e-12)
        X_norm = X_interp / denom

        all_X.append(X_norm.astype(np.float32))
        all_y.append(labels.astype(np.float32))
        device_ids.append(np.full((X_norm.shape[0],), did, dtype=np.int64))

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    device_ids = np.concatenate(device_ids, axis=0)

    return X, y, device_ids, joint_wns.astype(np.float32)


# -------------------------
# 2) TRANSFER/TEST LOADER (transfer_plate.csv, 96_samples.csv)
# -------------------------
def load_comp_data(filepath, is_train=True):
    """
    Robust loader for transfer_plate.csv (train-like) and 96_samples.csv (headerless test-like).

    transfer_plate.csv (is_train=True):
      - has targets in TARGET_COLS
      - spectral data is stored in many columns; sometimes includes sample_id and extra analyte columns
    96_samples.csv (is_train=False):
      - often header=None
      - first column is sample_id (repeated), needs ffill
      - spectral values may contain brackets
    """
    if is_train:
        df = pd.read_csv(filepath)
        # targets
        y = df[TARGET_COLS].values.astype(np.float32)

        # heuristic: spectral columns are everything except the last 4 columns in many versions
        # but we’ll do a more robust pick:
        # - exclude target cols
        # - exclude obvious non-spectral cols
        non_spec = set(TARGET_COLS)
        for c in df.columns:
            lc = str(c).lower()
            if "sample" in lc or "analyte" in lc or "device" in lc:
                non_spec.add(c)

        # attempt numeric-ish columns first
        numericish = [c for c in df.columns if c not in non_spec and str(c).replace(".", "", 1).isdigit()]
        if len(numericish) > 1000:
            X_df = df[numericish].copy()
            # add synthetic sample_id
            X_df.insert(0, "sample_id", np.arange(len(X_df)))
        else:
            # fallback: take everything except last 4 and targets, keep/force sample_id in first col
            X_df = df.drop(columns=[c for c in TARGET_COLS if c in df.columns]).copy()
            # if it already has sample_id keep it, else create it
            if "sample_id" not in [c.lower() for c in X_df.columns]:
                X_df.insert(0, "sample_id", np.arange(len(X_df)))
        X = X_df
    else:
        df = pd.read_csv(filepath, header=None)
        X = df.copy()
        y = None
        # name columns: sample_id + spectral columns
        X.columns = ["sample_id"] + [f"c{i}" for i in range(X.shape[1] - 1)]

    # Forward-fill sample_id and clean
    X["sample_id"] = X["sample_id"].ffill()
    X["sample_id"] = X["sample_id"].astype(str).str.strip()

    if not is_train:
        # test often has 'sample###' formatting
        X["sample_id"] = X["sample_id"].str.replace("sample", "", regex=False)
        X["sample_id"] = pd.to_numeric(X["sample_id"], errors="coerce").fillna(method="ffill").astype(int)

    # Clean spectral values (remove brackets and coerce to float)
    spectral_cols = [c for c in X.columns if c != "sample_id"]
    for c in spectral_cols:
        X[c] = (
            X[c].astype(str)
            .str.replace("[", "", regex=False)
            .str.replace("]", "", regex=False)
        )
        X[c] = pd.to_numeric(X[c], errors="coerce")

    return X, y

def fix_val_test_shape(X_2048: np.ndarray, joint_wns: np.ndarray):
    """
    transfer_plate and 96_samples are on a 2048-point grid from ~65..3350 (per public notebook).
    Select 300..1942 and interpolate onto joint_wns (300..1942).
    """
    lower_wns = int(joint_wns.min())
    upper_wns = int(joint_wns.max())

    # native grid used in the public notebook
    spectral_values = np.linspace(65, 3350, 2048).astype(np.float64)

    mask = (spectral_values >= lower_wns) & (spectral_values <= upper_wns)
    wns = spectral_values[mask]
    X_sel = X_2048[:, mask].astype(np.float64)

    X_interp = np.array([np.interp(joint_wns.astype(np.float64), xp=wns, fp=row) for row in X_sel], dtype=np.float64)

    # per-spectrum normalization (match train)
    denom = np.maximum(np.max(X_interp, axis=1, keepdims=True), 1e-12)
    X_norm = (X_interp / denom).astype(np.float32)
    return X_norm


# -------------------------
# 3) PINN-ish MODEL: c >= 0, fingerprints >= 0, smooth fingerprints, polynomial baseline
# -------------------------
class RamanPINN(nn.Module):
    def __init__(self, input_dim: int, num_components: int = 3, baseline_degree: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.num_components = num_components
        self.baseline_degree = baseline_degree

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_components),
        )
        self.softplus = nn.Softplus()

        # raw parameters -> softplus to enforce non-negativity
        self.fps_raw = nn.Parameter(torch.randn(num_components, input_dim) * 0.02)

        # Polynomial baseline basis (fixed), coefficients learned
        x = torch.linspace(-1.0, 1.0, input_dim).unsqueeze(1)  # (D,1)
        basis = [torch.ones_like(x)]
        for d in range(1, baseline_degree + 1):
            basis.append(x ** d)
        B = torch.cat(basis, dim=1)  # (D, deg+1)
        self.register_buffer("baseline_basis", B)  # not trainable
        self.baseline_coeff = nn.Parameter(torch.zeros(baseline_degree + 1))  # (deg+1,)

    def forward(self, x):
        # concentrations
        c = self.softplus(self.encoder(x))  # (N,3) non-negative

        # fingerprints
        fps = self.softplus(self.fps_raw)   # (3,D) non-negative

        # baseline
        baseline = (self.baseline_basis @ self.baseline_coeff).unsqueeze(0)  # (1,D)

        # reconstruction
        y_hat = c @ fps + baseline  # (N,D)
        return c, y_hat, fps


# -------------------------
# 4) TRAINING / EVAL
# -------------------------
def rmse(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2))

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    mse = nn.MSELoss()

    total = 0.0
    for Xb, yb in loader:
        Xb = Xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()

        c_pred, x_recon, fps = model(Xb)

        loss_conc = mse(c_pred, yb)
        loss_recon = mse(x_recon, Xb)

        # smoothness: second-difference penalty (encourages smooth fingerprints)
        d1 = fps[:, 1:] - fps[:, :-1]
        d2 = d1[:, 1:] - d1[:, :-1]
        loss_smooth = torch.mean(d2 ** 2)

        # sparse-ish concentrations
        loss_l1c = torch.mean(torch.abs(c_pred))

        loss = loss_conc + W_RECON * loss_recon + W_SMOOTH * loss_smooth + W_L1C * loss_l1c
        loss.backward()
        optimizer.step()

        total += loss.item() * Xb.size(0)

    return total / len(loader.dataset)

@torch.no_grad()
def eval_model(model, X, y, device):
    model.eval()
    X = X.to(device)
    y = y.to(device)
    c_pred, _, _ = model(X)
    return rmse(c_pred, y).item()

def lodo_splits(device_ids: np.ndarray):
    uniq = np.unique(device_ids)
    for did in uniq:
        val_idx = np.where(device_ids == did)[0]
        train_idx = np.where(device_ids != did)[0]
        yield did, train_idx, val_idx


# -------------------------
# 5) MAIN PIPELINE
# -------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load joint training set (8 devices)
    X_train_np, y_train_np, dev_ids, joint_wns = load_joint_train_dataset(TRAIN_DEVICE_NAMES)
    print("Train joint shape:", X_train_np.shape, y_train_np.shape, "features:", len(joint_wns))

    # Load transfer val + test
    X_val_raw, y_val_np = load_comp_data(os.path.join(DATA_PATH, "transfer_plate.csv"), is_train=True)
    X_test_raw, _ = load_comp_data(os.path.join(DATA_PATH, "96_samples.csv"), is_train=False)

    # Drop sample_id, reshape into (N,2,2048) and average replicates
    X_val_2048 = X_val_raw.drop(columns=["sample_id"]).values.reshape(-1, 2, 2048).mean(axis=1).astype(np.float32)
    X_test_2048 = X_test_raw.drop(columns=["sample_id"]).values.reshape(-1, 2, 2048).mean(axis=1).astype(np.float32)

    # Interpolate to joint_wns, normalize per spectrum
    X_val_np = fix_val_test_shape(X_val_2048, joint_wns.numpy())
    X_test_np = fix_val_test_shape(X_test_2048, joint_wns.numpy())

    print("Transfer val shape:", X_val_np.shape, y_val_np.shape)
    print("Test shape:", X_test_np.shape)

    # Torch tensors
    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.float32)
    X_val = torch.tensor(X_val_np, dtype=torch.float32)
    y_val = torch.tensor(y_val_np, dtype=torch.float32)
    X_test = torch.tensor(X_test_np, dtype=torch.float32)

    # -------------------------
    # LODO CV on train devices
    # -------------------------
    cv_scores = []
    for did, tr_idx, va_idx in lodo_splits(dev_ids):
        model = RamanPINN(input_dim=X_train.shape[1], num_components=3, baseline_degree=BASELINE_DEGREE).to(device)
        opt = optim.Adam(model.parameters(), lr=LR)

        train_loader = DataLoader(
            TensorDataset(X_train[tr_idx], y_train[tr_idx]),
            batch_size=BATCH_SIZE, shuffle=True, drop_last=False
        )

        for ep in range(EPOCHS_CV):
            loss = train_one_epoch(model, train_loader, opt, device)

        score = eval_model(model, X_train[va_idx], y_train[va_idx], device)
        cv_scores.append(score)
        print(f"LODO device={TRAIN_DEVICE_NAMES[did]} RMSE={score:.4f}")

    print("LODO CV mean RMSE:", float(np.mean(cv_scores)), "std:", float(np.std(cv_scores)))

    # -------------------------
    # Train FINAL model on all 8 devices
    # (You can early-stop using transfer_plate RMSE if desired)
    # -------------------------
    final_model = RamanPINN(input_dim=X_train.shape[1], num_components=3, baseline_degree=BASELINE_DEGREE).to(device)
    final_opt = optim.Adam(final_model.parameters(), lr=LR)

    full_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=False
    )

    best_val = 1e9
    best_state = None

    for ep in range(EPOCHS_FINAL):
        loss = train_one_epoch(final_model, full_loader, final_opt, device)
        val_score = eval_model(final_model, X_val, y_val, device)
        if val_score < best_val:
            best_val = val_score
            best_state = {k: v.detach().cpu().clone() for k, v in final_model.state_dict().items()}
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"[Final] epoch={ep+1:03d} loss={loss:.5f} transfer_RMSE={val_score:.4f}")

    print("Best transfer_plate RMSE:", best_val)
    if best_state is not None:
        final_model.load_state_dict(best_state)

    # -------------------------
    # Predict TEST and write submission
    # -------------------------
    final_model.eval()
    with torch.no_grad():
        c_test, _, _ = final_model(X_test.to(device))
        c_test = c_test.cpu().numpy()

    submission = pd.DataFrame(
        c_test,
        columns=TARGET_COLS  # Kaggle expects these 3 targets
    )
    submission.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with shape:", submission.shape)


if __name__ == "__main__":
    main()

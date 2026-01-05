import os, copy, random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

# Optional smoothing; if SciPy not present, SavGol is skipped.
try:
    from scipy.signal import savgol_filter
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# ---------------------------
# Config
# ---------------------------

# Kaggle default path. If you run locally, set DATA_PATH to your extracted dataset folder.
DATA_PATH = os.environ.get("DATA_PATH", "/kaggle/input/dig-4-bio-raman-transfer-learning-challenge")

DATASET_NAMES = [
    "anton_532", "anton_785", "kaiser", "mettler_toledo",
    "metrohm", "tec5", "timegate", "tornado"
]

JOINT_LOWER_WN = 300
JOINT_UPPER_WN = 1942
JOINT_WNS = np.arange(JOINT_LOWER_WN, JOINT_UPPER_WN + 1, dtype=np.float32)
L = len(JOINT_WNS)

# Choose meta-learning method: "reptile" (simple) or "fomaml" (stronger but heavier)
META_METHOD = "reptile"   # <- change to "fomaml" to use FOMAML

# What to adapt in the inner loop
# "all"    -> adapt encoder + head + fingerprints + baseline
# "encoder"-> adapt encoder+head+baseline; keep fingerprints fixed (often more stable)
# "head"   -> adapt head+baseline only
INNER_UPDATE_MODE = "encoder"

# Meta-learning hyperparams
META_ITERS = 1500
META_BATCH_TASKS = 4
INNER_STEPS = 5
SUPPORT_BS = 32
QUERY_BS = 64

INNER_LR = 1e-2
META_LR_REPTILE = 2e-2   # Reptile step size
META_LR_FOMAML = 1e-3    # Outer optimizer lr for FOMAML

# Loss weights
WEIGHT_RECON = 0.2       # reconstruction regularizer weight
WEIGHT_SMOOTH = 1e-3     # smoothness penalty on fingerprints
WEIGHT_L2 = 1e-5         # small L2

EVAL_EVERY = 100

# Fine-tune on transfer device
FINETUNE_EPOCHS = 80
FINETUNE_LR = 5e-4
FINETUNE_BATCH = 32
FINETUNE_UPDATE_MODE = "encoder"  # usually match INNER_UPDATE_MODE

SEED = 42


def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()


# ---------------------------
# Data loading / preprocessing
# ---------------------------

LOWER_BOUNDS = {
    "anton_532": 200, "anton_785": 100, "kaiser": -37, "mettler_toledo": 300,
    "metrohm": 200, "tec5": 85, "timegate": 200, "tornado": 300,
}
UPPER_BOUNDS = {
    "anton_532": 3500, "anton_785": 2300, "kaiser": 1942, "mettler_toledo": 3350,
    "metrohm": 3350, "tec5": 3210, "timegate": 2000, "tornado": 3300,
}


def read_device_csv(device_name: str, lower_wn: int, upper_wn: int):
    """
    Returns:
      spectra: (N, M) float32
      y:      (N, 3) float32
      wns:    (M,)   float32
    """
    lower_wn = max(lower_wn, LOWER_BOUNDS[device_name])
    upper_wn = min(upper_wn, UPPER_BOUNDS[device_name])

    fp = os.path.join(DATA_PATH, f"{device_name}.csv")
    df = pd.read_csv(fp)

    spec_cols = df.columns[:-5]
    wns_all = np.array([float(c) for c in spec_cols], dtype=np.float32)

    mask = (wns_all >= lower_wn) & (wns_all <= upper_wn)
    wns = wns_all[mask]
    spectra = df.iloc[:, :-5].iloc[:, mask].to_numpy(dtype=np.float32)

    # label in original notebook: df.iloc[:, -5:-1] and then take first 3 columns
    y = df.iloc[:, -5:-1].to_numpy(dtype=np.float32)[:, :3]

    return spectra, y, wns


def interp_to_joint(spectra: np.ndarray, wns: np.ndarray) -> np.ndarray:
    out = np.empty((spectra.shape[0], len(JOINT_WNS)), dtype=np.float32)
    for i in range(spectra.shape[0]):
        out[i] = np.interp(JOINT_WNS, wns, spectra[i]).astype(np.float32)
    return out


def preprocess_spectra(X: np.ndarray) -> np.ndarray:
    """
    Device-stable preprocessing:
    - optional SavGol smoothing
    - per-spectrum max norm (reduces intensity scale differences)
    - DOES NOT SNV-center per spectrum (keeps sign/shape consistent for reconstruction)
    """
    Xp = X.astype(np.float32, copy=True)

    if _HAS_SCIPY:
        win = 11 if Xp.shape[1] >= 11 else (Xp.shape[1] // 2 * 2 + 1)
        Xp = savgol_filter(Xp, window_length=win, polyorder=2, axis=1).astype(np.float32)

    denom = np.max(np.abs(Xp), axis=1, keepdims=True) + 1e-8
    Xp = Xp / denom
    return Xp.astype(np.float32)


def load_tasks():
    tasks = {}
    ys = []
    for name in DATASET_NAMES:
        spec, y, wns = read_device_csv(name, JOINT_LOWER_WN, JOINT_UPPER_WN)
        X = interp_to_joint(spec, wns)
        X = preprocess_spectra(X)
        tasks[name] = {"X": X, "y": y}
        ys.append(y)

    y_all = np.concatenate(ys, axis=0)

    # Scale targets WITHOUT centering, so non-negativity constraints remain compatible.
    y_scale = np.std(y_all, axis=0).astype(np.float32)
    y_scale[y_scale < 1e-6] = 1.0

    for name in tasks:
        tasks[name]["y_scaled"] = tasks[name]["y"] / y_scale

    return tasks, y_scale


def load_transfer_plate(y_scale: np.ndarray):
    fp = os.path.join(DATA_PATH, "transfer_plate.csv")
    df = pd.read_csv(fp)

    target_cols = ["Glucose (g/L)", "Sodium Acetate (g/L)", "Magnesium Acetate (g/L)"]
    mask = ~df[target_cols].isna().any(axis=1)

    df = df.loc[mask].reset_index(drop=True)

    y = df[target_cols].to_numpy(dtype=np.float32)

    # Take everything except the last 4 columns (same as the Kaggle notebook pattern)
    X = df.iloc[:, :-4].copy()
    X.columns = ["sample_id"] + [str(i) for i in range(X.shape[1] - 1)]
    X["sample_id"] = X["sample_id"].ffill().astype(str).str.strip()

    spec_cols = X.columns[1:]
    for col in spec_cols:
        X[col] = X[col].astype(str).str.replace("[", "", regex=False).str.replace("]", "", regex=False)
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X_vals = X.drop(columns=["sample_id"]).to_numpy(dtype=np.float32)
    X_vals = X_vals.reshape(-1, 2, 2048).mean(axis=1)

    # align to JOINT_WNS (300..1942) like the reference notebook
    spectral_values = np.linspace(65, 3350, 2048).astype(np.float32)
    m = (spectral_values >= JOINT_LOWER_WN) & (spectral_values <= JOINT_UPPER_WN)
    wns = spectral_values[m]
    X_sel = X_vals[:, m]
    X_joint = np.array([np.interp(JOINT_WNS, wns, row) for row in X_sel], dtype=np.float32)

    X_joint = preprocess_spectra(X_joint)
    y_scaled = y / y_scale
    return X_joint, y, y_scaled


def load_test_96():
    fp = os.path.join(DATA_PATH, "96_samples.csv")
    df = pd.read_csv(fp, header=None)

    X = df.copy()
    X.columns = ["sample_id"] + [str(i) for i in range(X.shape[1] - 1)]
    X["sample_id"] = X["sample_id"].ffill().astype(str).str.strip()

    spec_cols = X.columns[1:]
    for col in spec_cols:
        X[col] = X[col].astype(str).str.replace("[", "", regex=False).str.replace("]", "", regex=False)
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X_vals = X.drop(columns=["sample_id"]).to_numpy(dtype=np.float32)
    X_vals = X_vals.reshape(-1, 2, 2048).mean(axis=1)

    spectral_values = np.linspace(65, 3350, 2048).astype(np.float32)
    m = (spectral_values >= JOINT_LOWER_WN) & (spectral_values <= JOINT_UPPER_WN)
    wns = spectral_values[m]
    X_sel = X_vals[:, m]
    X_joint = np.array([np.interp(JOINT_WNS, wns, row) for row in X_sel], dtype=np.float32)

    X_joint = preprocess_spectra(X_joint)
    return X_joint


# ---------------------------
# Model
# ---------------------------

class RamanMetaPINN(nn.Module):
    """
    Encoder: 1D CNN -> latent -> concentrations (3)
    Decoder: reconstruct spectrum via linear mixing: recon = c @ fingerprints + baseline
    """
    def __init__(self, L: int):
        super().__init__()
        self.L = L

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        dummy = torch.zeros(1, 1, L)
        with torch.no_grad():
            h = self.encoder(dummy)
        flat = h.shape[1] * h.shape[2]

        self.head = nn.Sequential(
            nn.Linear(flat, 256),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Softplus()  # concentrations >= 0
        )

        self.fingerprints = nn.Parameter(torch.randn(3, L) * 0.05)
        self.baseline = nn.Parameter(torch.zeros(L))

    def forward(self, x):
        z = self.encoder(x.unsqueeze(1))
        z = z.flatten(1)
        c = self.head(z)                           # (B,3)
        recon = c @ self.fingerprints + self.baseline  # (B,L)
        return c, recon


def fingerprints_smoothness(fp: torch.Tensor) -> torch.Tensor:
    d1 = fp[:, 1:] - fp[:, :-1]
    d2 = d1[:, 1:] - d1[:, :-1]
    return (d2 ** 2).mean()


def compute_loss(model: nn.Module, xb: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
    pred, recon = model(xb)
    loss_conc = nn.SmoothL1Loss()(pred, yb)
    loss_recon = nn.MSELoss()(recon, xb)
    loss_smooth = fingerprints_smoothness(model.fingerprints)

    loss_l2 = 0.0
    for p in model.parameters():
        loss_l2 = loss_l2 + (p ** 2).mean()

    return loss_conc + WEIGHT_RECON * loss_recon + WEIGHT_SMOOTH * loss_smooth + WEIGHT_L2 * loss_l2


def get_inner_params(model: RamanMetaPINN, mode: str):
    if mode == "all":
        return list(model.parameters())
    if mode == "encoder":
        params = []
        params += list(model.encoder.parameters())
        params += list(model.head.parameters())
        params += [model.baseline]          # allow per-task baseline shift
        return params
    if mode == "head":
        return list(model.head.parameters()) + [model.baseline]
    raise ValueError(f"Unknown mode: {mode}")


def clone_model(base_model: RamanMetaPINN) -> RamanMetaPINN:
    m = RamanMetaPINN(L=base_model.L).to(DEVICE)
    m.load_state_dict(base_model.state_dict())
    return m


def sample_batch(X: torch.Tensor, y: torch.Tensor, batch_size: int):
    n = X.shape[0]
    idx = torch.randint(0, n, (batch_size,), device=X.device)
    return X[idx], y[idx]


# ---------------------------
# Meta-training
# ---------------------------

def meta_train_reptile(model: RamanMetaPINN, tasks: dict):
    model.train()
    names = list(tasks.keys())

    for it in range(1, META_ITERS + 1):
        deltas = [torch.zeros_like(p.data) for p in model.parameters()]
        chosen = random.sample(names, k=min(META_BATCH_TASKS, len(names)))
        avg_q = 0.0

        for tname in chosen:
            X = tasks[tname]["X_t"]
            y = tasks[tname]["y_t"]

            fast = clone_model(model)
            inner_params = get_inner_params(fast, INNER_UPDATE_MODE)
            inner_opt = optim.SGD(inner_params, lr=INNER_LR)

            for _ in range(INNER_STEPS):
                xb, yb = sample_batch(X, y, SUPPORT_BS)
                inner_opt.zero_grad(set_to_none=True)
                loss = compute_loss(fast, xb, yb)
                loss.backward()
                inner_opt.step()

            with torch.no_grad():
                xbq, ybq = sample_batch(X, y, QUERY_BS)
                avg_q += compute_loss(fast, xbq, ybq).item()

            for i, (p_meta, p_fast) in enumerate(zip(model.parameters(), fast.parameters())):
                deltas[i] += (p_fast.data - p_meta.data)

        scale = META_LR_REPTILE / len(chosen)
        for p, d in zip(model.parameters(), deltas):
            p.data.add_(d, alpha=scale)

        if it % EVAL_EVERY == 0:
            print(f"[Reptile] iter={it}/{META_ITERS} avg_query_loss={avg_q/len(chosen):.5f}")

    return model


def meta_train_fomaml(model: RamanMetaPINN, tasks: dict):
    model.train()
    names = list(tasks.keys())
    meta_opt = optim.Adam(model.parameters(), lr=META_LR_FOMAML)

    for it in range(1, META_ITERS + 1):
        meta_opt.zero_grad(set_to_none=True)
        chosen = random.sample(names, k=min(META_BATCH_TASKS, len(names)))
        avg_q = 0.0

        for tname in chosen:
            X = tasks[tname]["X_t"]
            y = tasks[tname]["y_t"]

            fast = clone_model(model)
            inner_params = get_inner_params(fast, INNER_UPDATE_MODE)
            inner_opt = optim.SGD(inner_params, lr=INNER_LR)

            for _ in range(INNER_STEPS):
                xb, yb = sample_batch(X, y, SUPPORT_BS)
                inner_opt.zero_grad(set_to_none=True)
                loss = compute_loss(fast, xb, yb)
                loss.backward()
                inner_opt.step()

            xbq, ybq = sample_batch(X, y, QUERY_BS)
            qloss = compute_loss(fast, xbq, ybq)
            avg_q += float(qloss.detach().cpu())

            grads = torch.autograd.grad(qloss, list(fast.parameters()), create_graph=False, retain_graph=False)

            # FOMAML: copy query grads from fast -> meta (ignore 2nd order)
            for p_meta, g in zip(model.parameters(), grads):
                if p_meta.grad is None:
                    p_meta.grad = g.detach().clone()
                else:
                    p_meta.grad.add_(g.detach())

        for p in model.parameters():
            if p.grad is not None:
                p.grad.div_(len(chosen))

        meta_opt.step()

        if it % EVAL_EVERY == 0:
            print(f"[FOMAML] iter={it}/{META_ITERS} avg_query_loss={avg_q/len(chosen):.5f}")

    return model


# ---------------------------
# Fine-tune + submission
# ---------------------------

def finetune_on_transfer(model: RamanMetaPINN, X_tr: np.ndarray, y_scaled: np.ndarray):
    model.train()
    X_t = torch.from_numpy(X_tr).to(DEVICE)
    y_t = torch.from_numpy(y_scaled).to(DEVICE)

    params = get_inner_params(model, FINETUNE_UPDATE_MODE)
    opt = optim.Adam(params, lr=FINETUNE_LR)

    n = X_t.shape[0]
    for ep in range(1, FINETUNE_EPOCHS + 1):
        perm = torch.randperm(n, device=DEVICE)
        total = 0.0
        for i in range(0, n, FINETUNE_BATCH):
            idx = perm[i:i + FINETUNE_BATCH]
            xb, yb = X_t[idx], y_t[idx]
            opt.zero_grad(set_to_none=True)
            loss = compute_loss(model, xb, yb)
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu()) * len(idx)

        if ep % 10 == 0 or ep == 1:
            print(f"[finetune] epoch={ep}/{FINETUNE_EPOCHS} loss={total/n:.5f}")

    return model


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def main():
    seed_all(SEED)
    print("DEVICE:", DEVICE)
    print("META_METHOD:", META_METHOD)
    print("INNER_UPDATE_MODE:", INNER_UPDATE_MODE)

    tasks, y_scale = load_tasks()
    print("y_scale:", y_scale)

    # Move device tasks to torch once for fast sampling
    for name in tasks:
        tasks[name]["X_t"] = torch.from_numpy(tasks[name]["X"]).to(DEVICE)
        tasks[name]["y_t"] = torch.from_numpy(tasks[name]["y_scaled"]).to(DEVICE)

    model = RamanMetaPINN(L=L).to(DEVICE)
    print("Model params:", sum(p.numel() for p in model.parameters()))

    if META_METHOD.lower() == "reptile":
        model = meta_train_reptile(model, tasks)
    elif META_METHOD.lower() == "fomaml":
        model = meta_train_fomaml(model, tasks)
    else:
        raise ValueError("META_METHOD must be 'reptile' or 'fomaml'")

    # Fine-tune on transfer device
    X_val, y_val, y_val_scaled = load_transfer_plate(y_scale)
    print("transfer_plate shapes:", X_val.shape, y_val.shape)
    model = finetune_on_transfer(model, X_val, y_val_scaled)

    # Evaluate on transfer_plate
    model.eval()
    with torch.no_grad():
        preds_scaled, _ = model(torch.from_numpy(X_val).to(DEVICE))
        preds = preds_scaled.detach().cpu().numpy() * y_scale
        preds = np.clip(preds, 0.0, None)

    print("Transfer plate RMSE (all targets):", rmse(y_val, preds))

    # Predict test
    X_test = load_test_96()
    with torch.no_grad():
        test_scaled, _ = model(torch.from_numpy(X_test).to(DEVICE))
        test_pred = test_scaled.detach().cpu().numpy() * y_scale
        test_pred = np.clip(test_pred, 0.0, None)

    # Submission
    sub_fp = os.path.join(DATA_PATH, "sample_submission.csv")
    if os.path.exists(sub_fp):
        sub = pd.read_csv(sub_fp)
        cols = [c for c in sub.columns if c.lower() not in ("sample_id", "id")]
        if len(cols) >= 3:
            sub[cols[0]] = test_pred[:, 0]
            sub[cols[1]] = test_pred[:, 1]
            sub[cols[2]] = test_pred[:, 2]
        else:
            sub = pd.DataFrame(test_pred, columns=["Glucose (g/L)", "Sodium Acetate (g/L)", "Magnesium Acetate (g/L)"])
    else:
        sub = pd.DataFrame(test_pred, columns=["Glucose (g/L)", "Sodium Acetate (g/L)", "Magnesium Acetate (g/L)"])

    sub.to_csv("submission.csv", index=False)
    print("Wrote submission.csv")


if __name__ == "__main__":
    main()

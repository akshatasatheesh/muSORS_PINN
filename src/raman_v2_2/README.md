# raman_v2_2 (v2 + invariance loss)

This is a **new, standalone** version of your Raman `v2` baseline that adds an **invariance loss** (robustness to nuisance spectral variations) without touching your existing code.

## What’s new

- Same interpolation + preprocessing pipeline you already have (MSC / baseline / Savitzky–Golay / scaling)
- **Invariance loss**: enforces that predictions stay stable under physics-inspired nuisance transforms:
  - intensity scaling
  - baseline drift (low-order polynomial)
  - small wavenumber shifts
  - (optional) additive noise

## Folder layout

- `train_vanilla_nn_v2.py` — training + submission writer
- `raman_v2_2/...` — data/preprocess/model helpers

## Install

From inside this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install numpy pandas scikit-learn torch scipy
```

> If you already have a working venv with these deps, you can reuse it.

## Run (no invariance loss)

```bash
python train_vanilla_nn_v2.py --data_dir "/Users/satheeshjoseph/Akshata/raman_light/data/dig-4-bio-raman-transfer-learning-challenge" --out_dir ./outputs_v2_2 --epochs 150 --batch_size 64 --lr 3e-4 --use_msc --baseline poly --baseline_degree 3 --use_savgol --savgol_window 11 --savgol_polyorder 3 --scaling global_max --scale_y --non_negative 0
```

## Run (with invariance loss)

Start with a small value (e.g., `0.05`–`0.2`).

```bash
python train_vanilla_nn_v2.py --data_dir "/Users/satheeshjoseph/Akshata/raman_light/data/dig-4-bio-raman-transfer-learning-challenge" --out_dir ./outputs_v2_2_inv --epochs 150 --batch_size 64 --lr 3e-4 --use_msc --baseline poly --baseline_degree 3 --use_savgol --savgol_window 11 --savgol_polyorder 3 --scaling global_max --scale_y --non_negative 0 --lambda_inv 0.1 --inv_scale_lo 0.9 --inv_scale_hi 1.1 --inv_baseline_amp 0.02 --inv_baseline_degree 2 --inv_max_shift 1.0 --inv_noise_std 0.0
```

### Notes / tips

- If you enable `--scale_y`, keep `--non_negative 0` (because standardized targets can be negative).
- If training diverges, reduce `--lambda_inv` and/or reduce `--inv_baseline_amp`.

Outputs:
- `./<out_dir>/model.pt`
- `./<out_dir>/submission.csv`

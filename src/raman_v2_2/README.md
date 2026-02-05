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

SAMPLE RUN
```bash
(.venv) satheeshjoseph@satheeshs-mbp raman_v2_2 % python train_vanilla_nn_v2.py --data_dir "/Users/satheeshjoseph/Akshata/raman_light/data/dig-4-bio-raman-transfer-learning-challenge" --out_dir ./outputs_v2_2 --epochs 150 --batch_size 64 --lr 3e-4 --use_msc --baseline poly --baseline_degree 3 --use_savgol --savgol_window 11 --savgol_polyorder 3 --scaling standard --non_negative 0 
Using device: cpu
Shared grid: 300.0 .. 1942.0 len= 1643
Train: (2261, 1643) (2261, 3) devices: 8
Val: (96, 1643) (96, 3)
Test: (96, 1643)
Preprocess: PreprocessConfig(use_msc=True, baseline='poly', baseline_degree=3, snip_max_half_window=20, snip_smooth_half_window=3, use_savgol=True, savgol_window=11, savgol_polyorder=3, savgol_deriv=0, scaling='standard')
Target weights: [0.05443924 2.4882488  0.45731193]
epoch=001 train_mse=0.841532 inv_mse=0.000000 val_rmse=2.9932 val_r2=[-2.084, -0.529, -0.208]
epoch=010 train_mse=0.263276 inv_mse=0.000000 val_rmse=1.8076 val_r2=[0.002, -1.393, -0.600]
epoch=020 train_mse=0.147127 inv_mse=0.000000 val_rmse=1.8746 val_r2=[-0.108, -1.483, -0.120]
epoch=030 train_mse=0.113853 inv_mse=0.000000 val_rmse=2.1918 val_r2=[-0.520, -2.401, -0.435]
Early stopping at epoch 35 (patience=20).
Best val RMSE: 1.788697717783597
Saved model: ./outputs_v2_2/model.pt
Wrote: ./outputs_v2_2/submission.csv shape= (96, 4)
              ID    Glucose  Sodium Acetate  Magnesium Sulfate
count  96.000000  96.000000       96.000000          96.000000
mean   48.500000   6.495628        0.614819           1.063587
std    27.856777   0.111445        0.158401           0.087874
min     1.000000   6.202198        0.381955           0.912840
25%    24.750000   6.415316        0.490281           0.994892
50%    48.500000   6.504153        0.571689           1.039753
75%    72.250000   6.591155        0.737336           1.138701
max    96.000000   6.679750        0.996660           1.290842
```

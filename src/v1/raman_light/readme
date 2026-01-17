# Raman Light: Vanilla Neural Net baseline

This is a minimal baseline to train a simple MLP on the **DIG 4 Bio Raman Transfer Learning Challenge** dataset.

## What it does
- Loads the 8 *device* training CSVs and interpolates spectra onto a shared wavenumber grid.
- Uses `transfer_plate.csv` as validation (closer to the test domain).
- Trains a vanilla MLP to predict 3 concentrations.
- Writes `submission.csv` for `96_samples.csv` in Kaggle format.

## Run
From this folder (so `raman_light/` is importable):

```bash
python train_vanilla_nn.py \
  --data_dir /path/to/dig-4-bio-raman-transfer-learning-challenge \
  --out_dir ./outputs \
  --epochs 60 --batch_size 64 --lr 1e-3
```

Outputs:
- `outputs/model.pt`
- `outputs/submission.csv`

## Notes
- `transfer_plate.csv` uses a column name `Magnesium Acetate (g/L)` even though the competition targets are
  `Magnesium Sulfate`. This baseline treats the 3rd label as the third target.
- For transfer/test plates, the wavenumber grid is not explicitly provided; we assume a linear grid 65..3350 with
  2048 points. If you have a definitive grid, pass `--plate_wn_start/--plate_wn_end`.

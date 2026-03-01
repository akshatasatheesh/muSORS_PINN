# Raman Transfer Learning Project — State Save
# Date: 2026-03-01
# Session: v12→v15 development

## VERSION HISTORY & SCORES

| Version | Public | Private | Notes |
|---------|--------|---------|-------|
| v8_ensemble | 0.80215 | — | 15% ensemble + 85% winning CSV |
| v8_win_15 | 0.89499 | — | Best known submission |
| v10_stack | 0.41489 | 0.58153 | Base ML models (no offset) |
| v10_hgb | 0.40785 | 0.57857 | HGB only |
| v10_nn | 0.14215 | 0.00868 | CATASTROPHIC — physics NN overfit |
| v10_ensemble | 0.40209 | 0.58505 | NN dragged it below stack |
| v10_stack+bias | 0.72365 | 0.82774 | Manual offset [+1.3, +0.1, +0.5] |
| v10_hgb+bias | 0.68106 | 0.74479 | Manual offset applied to HGB |
| v12_stack | 0.41489 | 0.58153 | Same base code, frozen NMF decoder |
| v12_nn | 0.14181 | 0.00430 | Still catastrophic — problem is 430K params/96 samples |
| v12_hgb | 0.40785 | 0.57857 | Same HGB |
| v12_ensemble | 0.39874 | 0.57945 | NN still hurting |

## KEY INSIGHTS DISCOVERED

1. **NN is fundamentally broken at n=96**: 430K params vs 96 samples = 4500:1 ratio.
   No regularization fixes this (frozen decoder, physics loss, etc all failed).

2. **Domain-shift bias is the #1 lever**: Manual [+1.3, +0.1, +0.5] offset jumped
   private from 0.58 → 0.83 (+0.25). The models predict correct ranking but are
   systematically shifted due to train→test instrument differences.

3. **Winning CSV (~0.90 private) is the anchor**: All improvements come from
   getting closer to winning CSV or blending with it effectively.

4. **Physics NN scores 0.00-0.01 private consistently** across v10, v12.
   NN auto-gate (exclude if OOF R² < 0.05) is essential.

## VERSION PROGRESSION (this session)

### v12: Frozen NMF Decoder (FAILED)
- Hypothesis: learnable decoder params (6500) caused overfitting
- Fix: pre-compute basis via NMF, freeze decoder, zero learnable params
- Result: NN still 0.00 private → problem is NN itself, not decoder
- File: /mnt/user-data/outputs/raman_transfer_v12.py

### v13: Physics as Features + Domain Calibration (no NN)
- Dropped NN entirely
- Physics NMF coefficients → input features for HGB (not NN loss)
- DomainShiftCalibrator: learns affine y=a*pred+b from winning CSV
- Added PLS regression (classical chemometrics)
- NO manual offset hardcoded
- File: /mnt/user-data/outputs/raman_transfer_v13.py
- Status: NOT YET SUBMITTED

### v14: v13 + v10 NN (merged)
- User requested NN back from v10 (exact physics loss code)
- Added auto-gate: NN excluded from ensemble if OOF R² < 0.05
- Removed manual offset
- Has both vanilla NN and PINN comparison (user added)
- File: /mnt/user-data/outputs/raman_transfer_v14.py
- Status: NOT YET SUBMITTED

### v15: 2×2 Factorial Comparison (CURRENT)
- Core experiment: Vanilla vs Physics at BOTH meta-learning AND fine-tuning
- Model A: Vanilla Reptile → Vanilla FT (pure baseline)
- Model B: Vanilla Reptile → PINN FT (physics only at fine-tune)
- Model C: Physics Reptile → Vanilla FT (physics only at meta-learn)
- Model D: Physics Reptile → PINN FT (full physics pipeline)
- NEW: `reptile_inner_loop_physics` — inner loop with data+physics loss
- NEW: `meta_train_physics` — physics-guided Reptile meta-learning
- NEW: `compute_physics_loss_meta` — gentler physics weights for meta stability
- 7 auto-generated comparison plots (bar chart, training curves, scatter,
  heatmap, residuals, meta convergence, summary table)
- File: /mnt/user-data/outputs/raman_transfer_v15.py
- Status: NOT YET RUN / NOT YET SUBMITTED

## FILES ON DISK

### Outputs (user-accessible)
- /mnt/user-data/outputs/raman_transfer_v12.py
- /mnt/user-data/outputs/raman_transfer_v13.py
- /mnt/user-data/outputs/raman_transfer_v14.py
- /mnt/user-data/outputs/raman_transfer_v15.py  ← LATEST

### User uploads (reference)
- /mnt/user-data/uploads/raman_transfer_v10.py (v10 physics NN code)
- /mnt/user-data/uploads/raman_transfer_v14_pinn_vs_vanilla.py (user's v14 with vanilla NN)
- /mnt/user-data/uploads/pinn_raman_v8.py (winning v8 approach)

## PENDING / NEXT STEPS

1. **Run v15** — the 2×2 factorial experiment hasn't been executed yet
2. **Submit v15 outputs** — particularly:
   - v15_nn_D (full physics pipeline NN)
   - v15_ensemble (auto-gated stack + best NN)
   - v15_ensemble_affine (domain-calibrated)
   - v15_ensemble_win_15 (blended with winning CSV)
3. **Analyze plots** — do the 7 comparison graphs show physics improving things?
4. **If NN still fails**: The user's project seems to be for a science fair /
   showcase comparing PINN vs vanilla. Even if physics doesn't help on this
   96-sample problem, the comparison framework and plots are the deliverable.
5. **Consider**: the manual bias experiment (+0.25 gain) suggests domain
   calibration is more impactful than any model change. v15 has the affine
   calibrator built in.

## TECHNICAL NOTES

- Winning CSV: submission_pp_hgb_7_2_0.csv (searched in data/ and ./)
- Train: 96 samples, 3 targets (Glucose, NaAc, MgSO4)
- Spectra: 300-1942 cm⁻¹, 1643 bins
- Device datasets: 8 auxiliary devices for meta-learning
- Physics: Raman linear mixing model X = c @ basis + baseline
- Physics losses: reconstruction, smoothness (TV+Laplacian), sparsity, peak locality

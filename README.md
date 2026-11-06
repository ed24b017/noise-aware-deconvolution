# Noise-Aware Deconvolution

Frequency-domain Wiener deconvolution whose regularizer λ(ω) is estimated from the
noisy image alone, with no ground truth and no hand-tuned noise-to-signal ratio.
It is compared against classical baselines on synthetic and real microscopy / OCT data.
The full write-up is in `Report.pdf`.

## Method in one paragraph

For the model `y = h * x + n`, the restored image is

    X̂(ω) = conj(H(ω)) Y(ω) / (|H(ω)|² + λ(ω))

The proposed method (`lib/proposed.py`, `adaptive_wiener`) sets
`λ(ω) = σ̂² / (A |ω|^-β)` in three steps:

1. **Noise level σ̂²:** Estimated from the frequencies where `|H|² < eps_band`. The blurred signal is
   gone there, so the spectrum is pure noise. If the PSF is too narrow to leave such a band,
   it falls back to a wavelet MAD estimate.
2. **Signal prior:** A power law `S_x(ω) = A |ω|^-β`, fitted to the noise-corrected, de-blurred
   radial spectrum using only bins where the signal clearly beats the noise.
3. **λ(ω):** The noise-to-signal ratio `σ̂² / S_x(ω)`, which is used as the Wiener regularizer.

## Repository layout

```
00_setup_data.ipynb          download + stage datasets
01_forward_baselines_new.ipynb   forward model sanity checks, baselines, tuning on dev set
02_proposed_new.ipynb        proposed method demo + ablation study
03_experiments_new.ipynb     main synthetic experiments + statistics
04_real_data_new.ipynb       real FMD and OCT data
05_figures_new.ipynb         all paper figures and tables from saved results
lib/                         importable code used by every notebook
  forward.py                 PSFs, blur, noise models, edge phantom, frequency grid
  data.py                    FMD loaders, normalization, crops
  metrics.py                 PSNR, SSIM, log-spectral distance, edge metrics, noise gain
  baselines.py               inverse, constant Wiener, Tikhonov+GCV, Richardson-Lucy, unsupervised Wiener, oracle Wiener, mirror padding
  proposed.py                the proposed adaptive Wiener filter
configs/
  params.yaml                experiment grid (seeds, PSF widths, noise levels, crop, eps_band)
  tuned.yaml                 baseline hyperparameters tuned on the dev set (written by 01)
  split.yaml                 dev / test split record (written by 04)
data/raw/FMD, data/raw/OCT   datasets (not in git; produced by 00)
results/                     CSVs written by the notebooks; results/tables/ holds the report tables
figures/                     PNGs written by the notebooks
Report.pdf                   the report
```

## Setup

Python 3.13 was used (see `results/versions.txt`: numpy 2.1.3, scipy 1.16.3, scikit-image 0.25.2).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install jupyter ipykernel      # not in requirements.txt
jupyter lab
```

Start Jupyter from the repository root. Each notebook sets `ROOT` to the
`DSP_DECONV_ROOT` environment variable, or to the current working directory if that is
unset, and adds `ROOT/lib` to `sys.path`. To run from elsewhere:

```bash
export DSP_DECONV_ROOT=/path/to/noise-aware-deconvolution
```

## Data

`00_setup_data.ipynb` stages two datasets. You can skip it if `data/raw/` is already populated.

| Dataset | Used for | Source | Size |
|---|---|---|---|
| FMD (Fluorescence Microscopy Denoising): `WideField_BPAE_B`, `Confocal_BPAE_B`, `TwoPhoton_MICE` | dev set, synthetic tests, real-noise tests | Google Drive IDs parsed from the `yinhaoz/denoising-fluorescence` repo; needs `gdown` and `git` | 3 tarballs |
| Kermany 2018 OCT (`paultimothymooney/kermany2018`) | qualitative real-data test | Kaggle; needs `~/.kaggle/kaggle.json` (chmod 600) or `KAGGLE_USERNAME`/`KAGGLE_KEY` | about 5 GB |

Expected layout after setup:

```
data/raw/FMD/<subset>/raw/<fov>/*.png     noisy frames
data/raw/FMD/<subset>/gt/<fov>/*.png      clean 50-frame average
data/raw/OCT/*.jpeg                       20 randomly chosen test B-scans
```

Notes for re-running 00:
- The OCT extraction step uses `z.extractall`, which can overwrite files. Pre-existing files can trigger an
  interactive "replace?" prompt when the cell is re-run.
- The FMD tarballs are cached in `data/tmp/`. Delete `data/tmp/` to reclaim disk once `data/raw/` is built.

## Running order

Run the notebooks in this order. Later notebooks read files written by earlier ones.

| # | Notebook | Reads | Writes |
|---|---|---|---|
| 0 | `00_setup_data` | internet | `data/raw/*`, `configs/params.yaml` (if missing) |
| 1 | `01_forward_baselines_new` | `data/raw/FMD` | `configs/tuned.yaml` (best constant NSR and RL iteration count, picked on the dev set) |
| 2 | `03_experiments_new` | `configs/tuned.yaml` | `results/synthetic_gauss_test.csv`, `synthetic_pg_test.csv`, `edge_test.csv`, `wilcoxon.csv`, `psf_mismatch.csv`, `boundary.csv`, `figures/boundary_error_map.png` |
| 3 | `02_proposed_new` | `tuned.yaml`, **`synthetic_gauss_test.csv`** | `results/ablation.csv` |
| 4 | `04_real_data_new` | `tuned.yaml`, `data/raw/OCT` | `results/real_fmd.csv`, `real_oct.csv`, `versions.txt`, `configs/split.yaml`, `figures/real_*.png` |
| 5 | `05_figures_new` | all of the above | `figures/fig1`–`fig7*.png`, `results/tables/table1`–`table9*.csv` |

Important: **02 must run after 03.** The last cells of 02 compare against `synthetic_gauss_test.csv`,
which 03 produces. Notebook numbering does not match dependency order here.

## What the experiments cover

- **Synthetic Gaussian noise (03):** 8 test images (4 quadrants × 2 subsets) × 3 PSF widths (σ = 1.5, 2.5, 4.0 px) × 4 noise levels
  (σ_n = 0.005–0.05) × 5 seeds, for 7 methods.
- **Poisson-Gaussian noise (03):** peak photon counts 30, 100, 300, with read noise 0.01.
- **Edge phantom (03):** edge width (FWHM), MTF50, overshoot, and noise amplification.
- **Wilcoxon signed-rank tests (03):** proposed vs. each tuned or unsupervised baseline.
- **PSF mismatch (03):** assumed PSF width off by ±10, 20, 30%.
- **Boundary handling (03):** periodic assumption vs. mirror padding.
- **Ablation (02):** oracle σ, oracle signal spectrum, and a constant λ (no frequency shaping).
- **Real FMD (04):** noise gain vs. resolution (MTF50) trade-off on the raw frame, with the 50-frame average as reference.
- **Real OCT (04):** background noise reduction on 20 B-scans (qualitative; the lateral PSF is assumed).

Methods compared: `inverse`, `wiener_const`, `tikhonov_gcv`, `rl`, `unsup_wiener`, `proposed`,
and `oracle_wiener`, which is an upper bound because it uses the true signal spectrum.

## Data split

Recorded in `configs/split.yaml`:
- **Dev:** `WideField_BPAE_B` fields 1–18. Fields 1–3 are used to tune the baselines.
- **Test:** field 19 of `Confocal_BPAE_B` and `TwoPhoton_MICE`. No test data is used for tuning.

## Known caveats and things to check when running

- **Tuned RL iterations are at the grid edge.** `configs/tuned.yaml` currently has `rl_iter: 1`,
  which is the lowest value in the search grid. Notebook 01 prints a warning for this. The
  RL baseline may therefore be handicapped; widen the grid and re-run before trusting RL comparisons.
- **`imp` module shim.** Each notebook injects a fake `imp` module into `sys.modules` so the
  `autoreload` extension works on Python 3.12+, where `imp` was removed. Notebooks 00–03 use
  `sys.modules["imp"] = importlib`, which has no `.reload`; 04 and 05 use a version that does.
  If `%autoreload` fails, use the 04/05 variant.
- **`lib/data.py` ignores `DSP_DECONV_ROOT`.** It finds the data from its own file location
  (`lib/..`), so the data must live inside the repository even if `ROOT` points elsewhere.
- **Private helpers are imported across modules.** 02 and 04 import `baselines._apply`
  directly, so renaming it breaks both notebooks. Autoreload is enabled so edits to `lib/` take
  effect without restarting the kernel.
- **Stale comments in `lib/`.** "Owner: P1/P2/P3" headers and references to "Cell 6" / "S6" / "S7"
  are leftovers from an earlier layout.
- **Unpinned dependencies.** `requirements.txt` lists no versions. The tested versions are in
  `results/versions.txt`. `jupyter` and `ipykernel` are missing from the file.
- **Runtime.** 03 and 02 loop over the full grid (thousands of deconvolutions, including
  `unsupervised_wiener`, which is slow). Expect these to take the longest. For a quick smoke test,
  reduce `seeds` in `configs/params.yaml` to `[0]`, and restore it afterwards.

## Quick smoke test (needs no dataset)

```python
import sys; sys.path.insert(0, 'lib')
import numpy as np
from scipy.ndimage import gaussian_filter
from forward import *; from proposed import adaptive_wiener; from metrics import psnr
x = gaussian_filter(np.random.default_rng(0).random((256, 256)), 1.0)
x = (x - x.min()) / (x.max() - x.min())
psf = gaussian_psf(psf_size(2.5), 2.5)
y = add_gaussian(blur(x, psf), 0.02, np.random.default_rng(1))
xh, info = adaptive_wiener(y, psf, return_info=True)
print(np.sqrt(info['sigma2']), info['beta'])   # sigma_hat should be close to 0.02
```

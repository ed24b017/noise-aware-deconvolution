# Dataset loaders.  Owner: P1
from pathlib import Path
import numpy as np, skimage.io as io
ROOT = Path(__file__).resolve().parents[1]

def _read(p):
    im = io.imread(p).astype(np.float64)
    return im.mean(-1) if im.ndim == 3 else im

def fmd_pair(subset, fov, frame=0):
    """Clean 50-frame average (gt) and one raw noisy frame. Change '*.png' if Cell 6 shows .tif files."""
    base = ROOT / 'data/raw/FMD' / subset
    gt = _read(sorted((base / 'gt' / str(fov)).glob('*.png'))[0])
    raw = _read(sorted((base / 'raw' / str(fov)).glob('*.png'))[frame])
    return gt, raw

def norm_pair(gt, *others):
    """Scale to 0-1 with the clean image's 0.5/99.5 percentiles; the same scale for every image of the set."""
    lo, hi = np.percentile(gt, [0.5, 99.5])
    f = lambda im: (im - lo) / (hi - lo)
    return (np.clip(f(gt), 0, 1),) + tuple(f(o) for o in others)

def center_crop(im, n):
    r, c = (im.shape[0] - n) // 2, (im.shape[1] - n) // 2
    return im[r:r + n, c:c + n]

def quadrants(im, n=256):
    """Four non-overlapping n x n crops: more test images from one field of view."""
    return [im[:n, :n], im[:n, n:2*n], im[n:2*n, :n], im[n:2*n, n:2*n]]

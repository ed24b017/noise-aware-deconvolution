# Baseline deconvolution methods.  Owner: P2
import numpy as np
from numpy.fft import fft2, ifft2
from forward import psf2otf
from skimage.restoration import richardson_lucy, unsupervised_wiener

def _apply(y, H, lam):
    """X_hat = conj(H) Y / (|H|^2 + lambda(w)); lam may be a number or a per-frequency array."""
    return np.real(ifft2(np.conj(H) * fft2(y) / (np.abs(H) ** 2 + lam)))

def inverse(y, psf, eps=1e-30):                  # eps only prevents division by exactly 0
    return _apply(y, psf2otf(psf, y.shape), eps)

def wiener_const(y, psf, nsr):                   # one noise-to-signal ratio for all frequencies
    return _apply(y, psf2otf(psf, y.shape), nsr)

def wiener_oracle(y, psf, x_true, sigma):
    """Upper bound: uses the TRUE signal spectrum. Only possible in simulation."""
    Sx = np.abs(fft2(x_true)) ** 2 / y.size      # periodogram, same scale as sigma^2
    return _apply(y, psf2otf(psf, y.shape), sigma ** 2 / (Sx + 1e-12))

def laplacian_otf(shape):
    L = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], float)
    return psf2otf(L, shape)

def tikhonov(y, psf, lam):
    return _apply(y, psf2otf(psf, y.shape), lam * np.abs(laplacian_otf(y.shape)) ** 2)

def gcv_lambda(y, psf, lams=np.logspace(-6, 0, 61)):
    """Generalized cross-validation: picks lambda from y alone (no ground truth)."""
    H2 = np.abs(psf2otf(psf, y.shape)) ** 2
    L2 = np.abs(laplacian_otf(y.shape)) ** 2
    Y2 = np.abs(fft2(y)) ** 2
    scores = []
    for lam in lams:
        F = lam * L2 / (H2 + lam * L2)           # 1 - filter gain, per frequency
        scores.append(np.sum(F ** 2 * Y2) / np.sum(F) ** 2)
    return lams[int(np.argmin(scores))], np.array(scores)

def tikhonov_gcv(y, psf):
    return tikhonov(y, psf, gcv_lambda(y, psf)[0])

def rl(y, psf, n_iter=30):
    return richardson_lucy(np.clip(y, 1e-6, None), psf, num_iter=n_iter, clip=False)

def unsup_wiener(y, psf):
    return unsupervised_wiener(y, psf, clip=False, rng=0)[0]

def padded(method, y, psf, pad=32, **kw):
    """Mirror-pad before FFT deconvolution, crop after: removes border ringing (used in S7)."""
    yp = np.pad(y, pad, mode='reflect')
    return method(yp, psf, **kw)[pad:-pad, pad:-pad]

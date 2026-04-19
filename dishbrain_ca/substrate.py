"""Lenia-inspired substrate with local parameter fields.

Activation A[y,x] in [0,1] on a toroidal grid. Growth is a rectified
threshold nonlinearity parameterised by (mu, sigma), stored as per-cell
fields so different regions of the dish can obey different rules. The
coupling kernel is a compact Gaussian, and convolution is done via FFT on
the torus.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LeniaParams:
    """Per-cell growth parameters. mu and sigma are (H, W) arrays."""

    mu: np.ndarray
    sigma: np.ndarray

    @classmethod
    def uniform(
        cls,
        H: int,
        W: int,
        mu: float = 0.15,
        sigma: float = 0.015,
        jitter: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> "LeniaParams":
        rng = rng or np.random.default_rng()
        m = np.full((H, W), mu, dtype=np.float32)
        s = np.full((H, W), sigma, dtype=np.float32)
        if jitter > 0:
            m = m + (jitter * rng.standard_normal((H, W))).astype(np.float32)
            s = s + (0.1 * jitter * rng.standard_normal((H, W))).astype(np.float32)
        s = np.maximum(s, 1e-3)
        return cls(mu=m, sigma=s)

    def copy(self) -> "LeniaParams":
        return LeniaParams(mu=self.mu.copy(), sigma=self.sigma.copy())


def make_kernel(R: int = 7, width: float = 0.45) -> np.ndarray:
    """Compact Gaussian smoothing kernel, normalised to sum=1.

    Small radius + modest width keeps local coupling tight, so signals
    propagate gradually rather than smearing across the dish in one tick.
    """
    yy, xx = np.mgrid[-R : R + 1, -R : R + 1]
    d = np.sqrt(xx * xx + yy * yy) / R
    k = np.exp(-0.5 * (d / width) ** 2)
    k[d > 1.0] = 0.0
    k = k / k.sum()
    return k.astype(np.float32)


def _kernel_to_fft(K: np.ndarray, H: int, W: int) -> np.ndarray:
    """Embed and center the kernel for circular FFT convolution."""
    KH, KW = K.shape
    Kp = np.zeros((H, W), dtype=np.float32)
    Kp[:KH, :KW] = K
    Kp = np.roll(Kp, -(KH // 2), axis=0)
    Kp = np.roll(Kp, -(KW // 2), axis=1)
    return np.fft.rfft2(Kp)


def conv_torus(A: np.ndarray, K_fft: np.ndarray) -> np.ndarray:
    """Circular convolution via precomputed kernel FFT."""
    return np.fft.irfft2(np.fft.rfft2(A) * K_fft, s=A.shape).astype(np.float32)


def growth(u: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Rectified sigmoid growth: zero below threshold, saturating above.

    g(u; mu, sigma) = max(0, tanh((u - mu) / sigma))

    Paired with a leak term in the integrator, this gives a monostable
    substrate: resting state A=0, stim drives local activity, per-cell mu
    controls excitability (lower = easier to fire), per-cell sigma controls
    the sharpness of the threshold. No NN, no backprop -- still a local
    rule with two scalar parameters per cell.
    """
    return np.maximum(0.0, np.tanh((u - mu) / sigma)).astype(np.float32)


@dataclass
class LeniaState:
    A: np.ndarray
    params: LeniaParams


def lenia_step(
    A: np.ndarray,
    params: LeniaParams,
    K_fft: np.ndarray,
    dt: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """One Lenia tick. Returns (new_A, u) where u is the neighborhood sum."""
    u = conv_torus(A, K_fft)
    G = growth(u, params.mu, params.sigma)
    A_new = np.clip(A + dt * G, 0.0, 1.0).astype(np.float32)
    return A_new, u

"""
Synthetic Cherenkov image generator for the CNN section of the lecture.

Design:
- 56x56 single-channel images (compatible with ToyCherenkovCNN)
- γ events: narrow elongated Gaussian blobs, low scatter, oriented toward centre
- proton events: wider, more spherical, sometimes multiple blobs, random orientation

These are NOT physically simulated showers — they are statistical surrogates
designed to give the CNN a learnable γ/hadron distinction in a fully reproducible
way for the lecture. The design choices echo the gross visual differences
explained in slide 5 (compact-elongated vs broad-irregular).
"""

from __future__ import annotations
import numpy as np


IMG_SIZE = 56


def _gaussian_blob(grid_x, grid_y, cx, cy, sx, sy, theta, amplitude=1.0):
    """Anisotropic 2D Gaussian, rotated by `theta` rad."""
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    a = (cos_t**2) / (2 * sx**2) + (sin_t**2) / (2 * sy**2)
    b = -np.sin(2 * theta) / (4 * sx**2) + np.sin(2 * theta) / (4 * sy**2)
    c = (sin_t**2) / (2 * sx**2) + (cos_t**2) / (2 * sy**2)
    z = a * (grid_x - cx)**2 + 2 * b * (grid_x - cx) * (grid_y - cy) + c * (grid_y - cy)**2
    return amplitude * np.exp(-z)


def _add_pixel_noise(img, level=0.05, rng=None):
    """Add Poisson-like + Gaussian noise floor."""
    rng = rng or np.random.default_rng()
    out = img + rng.normal(0, level, img.shape)
    out = np.clip(out, 0, None)
    return out


def synthesize_gamma(rng=None) -> np.ndarray:
    """
    γ-like image: elongated blob, mostly pointing radially toward camera centre.
    """
    rng = rng or np.random.default_rng()
    grid_x, grid_y = np.meshgrid(np.arange(IMG_SIZE), np.arange(IMG_SIZE))
    centre = IMG_SIZE / 2

    r = rng.uniform(3, 16)
    phi = rng.uniform(0, 2 * np.pi)
    cx = centre + r * np.cos(phi)
    cy = centre + r * np.sin(phi)

    # 90% radial-pointing, 10% random — small label-noise
    if rng.random() < 0.10:
        theta = rng.uniform(0, 2 * np.pi)
    else:
        radial_angle = np.arctan2(cy - centre, cx - centre)
        alpha_jitter = rng.normal(0, np.deg2rad(20))
        theta = radial_angle + alpha_jitter

    # Elongation: typically narrow
    sx = rng.uniform(3.5, 6.0)
    sy = rng.uniform(1.4, 2.6)
    amp = rng.uniform(0.6, 1.0)

    img = _gaussian_blob(grid_x, grid_y, cx, cy, sx, sy, theta, amp)

    if rng.random() < 0.10:
        cx2 = centre + rng.normal(0, 14)
        cy2 = centre + rng.normal(0, 14)
        amp2 = rng.uniform(0.15, 0.30)
        img2 = _gaussian_blob(grid_x, grid_y, cx2, cy2,
                              rng.uniform(1.5, 2.5), rng.uniform(1.5, 2.5),
                              rng.uniform(0, 2 * np.pi), amp2)
        img = img + img2

    return _add_pixel_noise(img, 0.08, rng)


def synthesize_proton(rng=None) -> np.ndarray:
    """
    Proton-like image: broader on average, randomly oriented (only 10% pointing radially).
    """
    rng = rng or np.random.default_rng()
    grid_x, grid_y = np.meshgrid(np.arange(IMG_SIZE), np.arange(IMG_SIZE))
    centre = IMG_SIZE / 2

    cx = centre + rng.normal(0, 11)
    cy = centre + rng.normal(0, 11)

    # 10% pointing radially (label noise for the hard cases)
    if rng.random() < 0.10:
        radial_angle = np.arctan2(cy - centre, cx - centre)
        theta = radial_angle + rng.normal(0, np.deg2rad(20))
    else:
        theta = rng.uniform(0, 2 * np.pi)

    # Slightly broader on average
    sx = rng.uniform(2.5, 5.0)
    sy = rng.uniform(2.0, 4.0)
    amp = rng.uniform(0.4, 0.9)

    img = _gaussian_blob(grid_x, grid_y, cx, cy, sx, sy, theta, amp)

    if rng.random() < 0.40:
        cx2 = centre + rng.normal(0, 13)
        cy2 = centre + rng.normal(0, 13)
        sx2 = rng.uniform(1.5, 3.0)
        sy2 = rng.uniform(1.2, 2.5)
        amp2 = rng.uniform(0.25, 0.50)
        img2 = _gaussian_blob(grid_x, grid_y, cx2, cy2, sx2, sy2,
                              rng.uniform(0, 2 * np.pi), amp2)
        img = img + img2

    return _add_pixel_noise(img, 0.10, rng)


def generate_dataset(n_per_class: int = 5000, seed: int = 42):
    """Build a paired (X, y) dataset of synthetic Cherenkov-like images.

    Returns
    -------
    X : np.ndarray of shape (2 * n_per_class, 1, 56, 56), float32
    y : np.ndarray of shape (2 * n_per_class,), int64
        1 = γ, 0 = proton
    """
    rng = np.random.default_rng(seed)
    X = np.empty((2 * n_per_class, 1, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    y = np.empty(2 * n_per_class, dtype=np.int64)
    for i in range(n_per_class):
        X[i, 0] = synthesize_gamma(rng)
        y[i] = 1
    for i in range(n_per_class):
        X[n_per_class + i, 0] = synthesize_proton(rng)
        y[n_per_class + i] = 0
    # Shuffle so γ and proton are interleaved
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


if __name__ == "__main__":
    X, y = generate_dataset(n_per_class=100)
    print(f"X shape: {X.shape}  dtype={X.dtype}")
    print(f"y shape: {y.shape}  dtype={y.dtype}")
    print(f"γ: {(y == 1).sum()}  proton: {(y == 0).sum()}")
    print(f"intensity range: [{X.min():.3f}, {X.max():.3f}]")
    print(f"mean intensity (γ):     {X[y == 1].mean():.3f}")
    print(f"mean intensity (proton): {X[y == 0].mean():.3f}")

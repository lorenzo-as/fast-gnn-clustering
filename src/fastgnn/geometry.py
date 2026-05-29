"""Geometry coordinate helpers."""

from __future__ import annotations

import numpy as np


def etaphi_to_xy_at_z(
    eta: np.ndarray,
    phi: np.ndarray,
    z: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project eta/phi coordinates to cartesian coordinates at a fixed z plane."""
    theta = 2.0 * np.arctan(np.exp(-eta))
    r = z * np.tan(theta)
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return x, y, np.full_like(x, z)


def xyz_to_eta_phi(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert cartesian coordinates to eta (pseudorapidity) and phi (azimuth)."""
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(r, np.abs(z))
    eta = -np.log(np.tan(theta / 2.0 + 1e-9))
    eta = np.where(z < 0, -eta, eta)
    phi = np.arctan2(y, x)
    return eta, phi


def xyz_to_r_phi_z(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert cartesian coordinates to cylindrical coordinates."""
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return r, phi, z

"""
CMSSW coordinate utility functions for HGCAL geometry.

Used by preprocessing and dataset code.
"""

from __future__ import annotations

import numpy as np

HGCAL_Z = 318.5  # cm, z position of HGCAL face for projection


def etaphi_to_xy_at_z(
    eta: np.ndarray,
    phi: np.ndarray,
    z: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert (eta, phi) impact coordinates to (x, y, z) at a given z plane.
    Used to project SimCluster impact points onto the detector face.

    Args:
        eta:  pseudorapidity array
        phi:  azimuthal angle array (radians)
        z:    z position of the projection plane in cm (e.g. 318.5 for HGCAL face)

    Returns:
        (x, y, z_array) in cm
    """
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
    """
    Convert cartesian (x, y, z) hit positions to (eta, phi).

    Args:
        x, y, z: hit positions in cm

    Returns:
        (eta, phi) arrays
    """
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(r, np.abs(z))
    eta = -np.log(np.tan(theta / 2.0 + 1e-9))
    # Preserve sign of z in eta
    eta = np.where(z < 0, -eta, eta)
    phi = np.arctan2(y, x)
    return eta, phi


def xyz_to_r_phi_z(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert cartesian (x, y, z) to cylindrical (r, phi, z).

    Returns:
        (r, phi, z)
    """
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return r, phi, z

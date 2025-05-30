# File: odf_coefficients.py

import numpy as np
from wigner_d import wigner_D_matrix

def compute_odf_coefficient_matrices(euler_angles, weights, L_max):
    """
    Computes the ODF coefficient matrices C^l for l in [0, L_max].

    Parameters:
        euler_angles: (N,3) array of Euler angles (radians)
        weights: (N,) array of volume fractions per grain
        L_max: int, maximum degree of harmonics

    Returns:
        coeffs: dict mapping l -> complex ndarray of shape (2l+1, 2l+1)
    """
    coeffs = {}
    for l in range(L_max + 1):
        size = 2 * l + 1
        C_l = np.zeros((size, size), dtype=complex)
        for angle, w in zip(euler_angles, weights):
            D = wigner_D_matrix(l, *angle)
            C_l += w * D
        coeffs[l] = C_l
    return coeffs

import numpy as np
import scipy
from scipy.special import sph_harm
from scipy.spatial.transform import Rotation as R
from scipy.special import iv

from sympy import symbols, factorial, sqrt, cos, sin, pi, I, S, simplify
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.physics.quantum.spin import Rotation as Wigner_d

from tqdm import tqdm
import matplotlib.pyplot as plt

def generate_symmetry_quats(group_name="O"):
    """
    Generate the symmetry group elements as quaternions [w, x, y, z]
    suitable for symmetrized Wigner D calculations.

    Args:
        group_name (str):
            - 'O' for cubic m-3m (full octahedral, 24 elements)
            - 'T' for tetrahedral
            - 'I' for icosahedral
            - 'D3', 'D4', etc. for dihedral groups
            - 'C4', 'C3', etc. for cyclic groups

    Returns:
        List of numpy arrays: each quaternion [w, x, y, z]
    """
    group = R.create_group(group_name)  # generates RotationGroup object
    quats_xyzw = group.as_quat()        # returns quats in [x, y, z, w] order
    quats_wxyz = np.column_stack((quats_xyzw[:, 3], quats_xyzw[:, 0], quats_xyzw[:, 1], quats_xyzw[:, 2]))
    return [q for q in quats_wxyz]

def quat_mult(q1, q2):
    """
    Multiply two quaternions: q = q1 * q2
    Each quaternion is [w, x, y, z]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z])

def quaternion_to_euler_zyz(q):
    """
    Converts quaternion [w, x, y, z] to ZYZ Euler angles (alpha, beta, gamma)
    """
    r = R.from_quat([q[1], q[2], q[3], q[0]])  # [x, y, z, w]
    return r.as_euler('ZYZ', degrees=False)

def compute_wigner_D_element(l, m, n, q):
    """
    Compute a single Wigner D^l_{mn} element at quaternion q
    """
    from sympy import N, exp

    alpha, beta, gamma = quaternion_to_euler_zyz(q)

    # Convert to sympy floats
    alpha = N(alpha)
    beta = N(beta)
    gamma = N(gamma)

    d_val = Wigner_d(l).doit().doit()[m, n].subs('beta', beta)
    D_val = exp(-I * m * alpha) * d_val * exp(-I * n * gamma)
    return complex(D_val.evalf())

def symmetrized_wigner_D(l, m, n, q, symmetry_quats):
    """
    Computes symmetrized D^l_{mn}(q) using symmetry operations G
    """
    total = 0
    for s in symmetry_quats:
        q_sym = quat_mult(q, s)
        total += compute_wigner_D_element(l, m, n, q_sym)
    return total / len(symmetry_quats)

def k_hat_vp(l, m, n, kappa):
    """
    Compute K_hat^l_{mn} for de la Vallée-Poussin kernel (zonal kernel).
    Returns 0 if m ≠ n due to zonal symmetry.
    """
    if m != n:
        return 0.0

    coeff = (2 * l + 1) / (4 * np.pi)
    bessel_ratio = iv(l, kappa) / iv(0, kappa)
    return coeff * bessel_ratio

def compute_clmn_symmetrized(l, m, n, ori_quats, symmetry_quats):
    """
    Compute symmetrized ODF Fourier coefficient c^l_{mn}

    Args:
        l, m, n (int): Wigner D function indices
        ori_quats (List[List[float]]): List of quaternions [w, x, y, z]
        symmetry_quats (List[List[float]]): List of symmetry quaternions [w, x, y, z]

    Returns:
        complex: c^l_{mn} coefficient
    """
    N = len(ori_quats)
    total = 0.0 + 0.0j

    for q in ori_quats:
        D_sym = symmetrized_wigner_D(l, m, n, q, symmetry_quats)
        total += D_sym

    return total / N

def compute_flmn(l, m, n, ori_quats, symmetry_quats, kappa):
    """
    Compute f^l_{mn} = c^l_{mn} * K_hat^l_{mn} for the de la Vallée-Poussin kernel

    Args:
        l, m, n (int): Wigner D indices
        ori_quats (List[List[float]]): Orientation quaternions
        symmetry_quats (List[List[float]]): Symmetry group quaternions
        kappa (float): Concentration parameter of the VP kernel

    Returns:
        complex: f^l_{mn} coefficient
    """
    c_lmn = compute_clmn_symmetrized(l, m, n, ori_quats, symmetry_quats)
    K_hat = k_hat_vp(l, m, n, kappa)
    return c_lmn * K_hat

def compute_odf_coefficients_symmetrized(L_max, ori_quats, symmetry_quats, kappa):
    """
    Computes ODF coefficients f^l_{mn} using symmetrized Wigner D functions.

    Returns:
        coeff_dict: dictionary of {(l, m, n): complex}
        coeff_vector: np.ndarray flattened for PCA
    """
    coeff_dict = {}
    coeff_vector = []

    for l in range(L_max + 1):
        for m in range(-l, l + 1):
            for n in range(-l, l + 1):
                f_lmn = compute_flmn(l, m, n, ori_quats, symmetry_quats, kappa)
                coeff_dict[(l, m, n)] = f_lmn
                coeff_vector.append(f_lmn.real)
                coeff_vector.append(f_lmn.imag)

    coeff_vector = np.array(coeff_vector)
    return coeff_dict, coeff_vector

def evaluate_odf_symmetrized(q, coeff_dict, symmetry_quats, L_max):
    """
    Evaluate the ODF at a quaternion q using symmetrized Wigner D and coefficients.

    Args:
        q (List[float]): Quaternion [w, x, y, z]
        coeff_dict (dict): {(l, m, n): complex}
        symmetry_quats (List[List[float]]): Symmetry group elements
        L_max (int): Maximum harmonic degree

    Returns:
        float: ODF value at orientation q
    """
    f_val = 0.0 + 0.0j

    for l in range(L_max + 1):
        for m in range(-l, l + 1):
            for n in range(-l, l + 1):
                f_lmn = coeff_dict[(l, m, n)]
                D_sym = symmetrized_wigner_D(l, m, n, q, symmetry_quats)
                f_val += f_lmn * D_sym

    return f_val.real

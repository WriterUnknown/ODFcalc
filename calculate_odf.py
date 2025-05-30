# File: calculate_odf.py

import numpy as np
from wigner_d import wigner_D_matrix
from odf_coefficients import compute_odf_coefficients

def compute_odf(odf_coeffs, L_max, phi1, Phi, phi2):
    """
    Reconstructs the ODF value at a given Euler angle (phi1, Phi, phi2) in radians.

    Parameters:
        odf_coeffs: dict of ODF coefficient matrices {l: matrix}
        L_max: max degree of harmonics
        phi1, Phi, phi2: Euler angles (in radians)

    Returns:
        odf_value: float (real part of the complex spherical harmonic sum)
    """
    odf_value = 0.0
    for l in range(L_max + 1):
        D = wigner_D_matrix(l, phi1, Phi, phi2)
        coeff_matrix = odf_coeffs[l]
        odf_value += np.real(np.sum(coeff_matrix * D))
    return odf_value

def evaluate_odf_on_grid(odf_coeffs, L_max, n_points=50):
    """
    Evaluate the ODF on a regular grid of Euler angles.

    Returns:
        odf_grid: (n, n, n) array of ODF values
        angles_grid: tuple of phi1, Phi, phi2 arrays
    """
    phi1_vals = np.linspace(0, 2*np.pi, n_points)
    Phi_vals = np.linspace(0, 0.5*np.pi, n_points)  # Restricting range due to symmetry
    phi2_vals = np.linspace(0, 2*np.pi, n_points)

    odf_grid = np.zeros((n_points, n_points, n_points))

    for i, phi1 in enumerate(phi1_vals):
        for j, Phi in enumerate(Phi_vals):
            for k, phi2 in enumerate(phi2_vals):
                odf_grid[i, j, k] = compute_odf(odf_coeffs, L_max, phi1, Phi, phi2)

    return odf_grid, (phi1_vals, Phi_vals, phi2_vals)

def read_ori_file(filepath):
    """
    Reads Euler angles from a Neper .ori file.

    Parameters:
        filepath: str, path to the .ori file

    Returns:
        euler_angles: (N, 3) ndarray of Euler angles in radians
    """
    data = np.loadtxt(filepath)
    # If Neper outputs degrees, convert to radians
    euler_angles = np.radians(data[:, :3])
    return euler_angles

def read_volumes_from_tess(filepath):
    """
    Extracts grain volumes (weights) from a Neper .tess file.

    Parameters:
        filepath: str, path to the .tess file

    Returns:
        weights: (N,) ndarray of volume fractions
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Look for the line that starts with "*Cells"
    for i, line in enumerate(lines):
        if line.strip().startswith("*Cells"):
            start_idx = i + 1
            break
    else:
        raise ValueError("No *Cells section found in .tess file.")

    # Read number of grains
    n_cells = int(lines[start_idx].strip())
    volumes = []

    for line in lines[start_idx + 1: start_idx + 1 + n_cells]:
        parts = line.strip().split()
        volume = float(parts[1])  # Neper .tess format: id volume ...
        volumes.append(volume)

    volumes = np.array(volumes)
    weights = volumes / np.sum(volumes)
    return weights


def main(ori_file, tess_file, L_max):
    # Read Neper orientation and tessellation data
    euler_angles = read_ori_file(ori_file)
    weights = read_volumes_from_tess(tess_file)

    # Compute coefficient matrices with custom grain-based sampling
    odf_coeffs = compute_odf_coefficient_matrices(euler_angles, weights, L_max)

    # Evaluate ODF on grid (for visualization or analysis)
    odf_grid, angle_grid = evaluate_odf_on_grid(odf_coeffs, L_max)

    # Output or visualize
    print("ODF grid shape:", odf_grid.shape)
    return odf_grid, angle_grid

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute ODF from Neper orientation and tessellation files.")
    parser.add_argument("--ori", type=str, required=True, help="Path to .ori file")
    parser.add_argument("--tess", type=str, required=True, help="Path to .tess file")
    parser.add_argument("--L_max", type=int, default=6, help="Maximum degree of spherical harmonics")
    args = parser.parse_args()

    odf_grid, angle_grid = main(args.ori, args.tess, args.L_max)
    # You can add code here to visualize slices or projections of the ODF.

import numpy as np
from math import factorial

def wigner_d(l, m, n, beta):

    d = 0.0
    k_min = max(0, m, n)
    k_max = min(l+m, l-n)

    for k in range(k_min, k_max + 1):

        num = ((-1)**(k + n - m)) * np.sqrt(factorial(l + m) * factorial(l - m) * factorial(l + n) * factorial(l - n))
        denom = (factorial(k) * factorial(l + m - k) * factorial(l - n - k) * factorial(n - m + k))
        angle = (np.cos(beta / 2)**((2 * l) + m - n - (2 * k)) * np.sin(beta / 2)**(2 * k + n - m))

        d += num / denom * angle

    return d


def wigner_D(l, m, n, alpha, beta, gamma):

    d = wigner_d_small(l, m, n, beta)
    D = np.exp(-1j * m * alpha) * d * np.exp(-1j * n * gamma)

    return D

def wigner_D_matrix(l, alpha, beta, gamma):

    size = 2 * l + 1
    D = np.zeros((size, size), dtype=complex)

    m_vals = np.arange(-l, l + 1)
    n_vals = np.arange(-l, l + 1)

    for i, m in enumerate(m_vals):
        for j, n in enumerate(n_vals):
            D[i, j] = wigner_D(l, m, n, alpha, beta, gamma)

    return D

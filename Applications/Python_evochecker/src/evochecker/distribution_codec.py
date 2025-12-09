from __future__ import annotations

import numpy as np

_EPS = 1e-12


def stick_break(u: np.ndarray) -> np.ndarray:
    """
    u: shape (k-1,), values in [0,1]
    returns p: shape (k,), sum(p)=1
    """
    u = np.asarray(u, dtype=float)
    u = np.clip(u, _EPS, 1.0 - _EPS)

    k_minus_1 = u.shape[0]
    k = k_minus_1 + 1

    p = np.empty(k, dtype=float)
    rem = 1.0
    for i in range(k_minus_1):
        p[i] = rem * u[i]
        rem *= (1.0 - u[i])
    p[-1] = rem
    return p


def inv_stick_break(p: np.ndarray) -> np.ndarray:
    """
    Inverse mapping: p (k,) -> u (k-1,), for loading seed files.
    p: probabilities, sum to 1.
    """
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 0.0, 1.0)
    s = float(p.sum())
    if s <= 0:
        raise ValueError("Distribution has zero sum.")
    p = p / s

    k = p.shape[0]
    u = np.empty(k - 1, dtype=float)

    rem = 1.0
    for i in range(k - 1):
        if rem <= _EPS:
            u[i] = 0.0
        else:
            u[i] = np.clip(p[i] / rem, _EPS, 1.0 - _EPS)
        rem -= p[i]
        rem = max(rem, 0.0)
    return u

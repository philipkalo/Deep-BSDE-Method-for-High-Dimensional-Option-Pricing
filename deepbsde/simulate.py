"""Geometric-Brownian-motion simulators.

Two kinds are needed:

* ``simulate_terminal`` jumps straight to maturity (European payoffs, MC).
* ``simulate_paths_multi`` returns the whole time-discretised path plus the
  Brownian increments, which the Deep BSDE recursion needs.

Correlation is introduced by a Cholesky factor of the correlation matrix.
Log-Euler stepping is exact for GBM, so the asset paths carry no
discretisation bias (only the backward BSDE recursion does).
"""

import numpy as np


def simulate_terminal(S0, sigma, rho, r, T, N, seed=12345):
    """Simulate terminal asset prices S_T only.

    Parameters
    ----------
    S0, sigma : (d,) arrays
    rho : (d, d) array
    N : int             number of paths

    Returns
    -------
    (N, d) array of terminal prices
    """
    L = np.linalg.cholesky(rho)
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((N, len(S0)))
    return S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * (Z @ L.T))


def simulate_paths_multi(S0, sigma, rho, r, T, N_steps, N_paths, seed=12345):
    """Simulate full correlated GBM paths and their Brownian increments.

    Returns
    -------
    X  : (N_paths, N_steps + 1, d)   asset prices at every time node
    dW : (N_paths, N_steps, d)       Brownian increments (variance dt)

    The correlation acts on the trailing (asset) axis via ``Z @ L.T``.
    """
    d = len(S0)
    dt = T / N_steps
    L = np.linalg.cholesky(rho)
    rng = np.random.default_rng(seed)

    Z = rng.standard_normal((N_paths, N_steps, d))
    dW = (Z @ L.T) * np.sqrt(dt)                    # correlated, variance dt

    drift = (r - 0.5 * sigma**2) * dt              # (d,), broadcast over paths/steps
    increments = drift + sigma * dW
    log_incr = np.cumsum(increments, axis=1)
    log_S0 = np.broadcast_to(np.log(S0), (N_paths, 1, d))
    log_path = np.concatenate([log_S0, np.log(S0) + log_incr], axis=1)
    return np.exp(log_path), dW

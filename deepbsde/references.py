"""Reference prices: closed-form and Monte Carlo.

These are the "known answers" the Deep BSDE solver is validated against.
The geometric basket has a closed form (a product of lognormals is
lognormal), so it collapses to a single effective asset; the arithmetic
basket does not and must be priced numerically.
"""

import numpy as np
from scipy.stats import norm


# --------------------------------------------------------------------------
# Single-asset Black-Scholes
# --------------------------------------------------------------------------
def black_scholes_call(S0, K, r, T, sigma):
    """Black-Scholes price of a European call.

    Parameters
    ----------
    S0, K : float   spot and strike
    r : float       risk-free rate
    T : float       time to maturity (years)
    sigma : float   volatility

    Returns
    -------
    float           the call price
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def black_scholes_call_delta(S0, K, r, T, sigma):
    """Black-Scholes call delta (dPrice/dS0 = N(d1))."""
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)


# --------------------------------------------------------------------------
# Effective-asset parameters for the geometric basket
# --------------------------------------------------------------------------
def effective_params(sigma, rho, r):
    """Effective volatility and drift of the geometric basket.

    The geometric average of correlated lognormals is itself lognormal,
    with these effective parameters, so basket pricing reduces to a
    single-asset Black-Scholes problem.

    Returns
    -------
    (sigma_hat, mu_hat) : floats
    """
    d = len(sigma)
    sigma_hat = np.sqrt(np.sum(sigma[:, None] * sigma[None, :] * rho) / d**2)
    mu_hat = (1.0 / d) * np.sum(r - 0.5 * sigma**2) + 0.5 * sigma_hat**2
    return sigma_hat, mu_hat


def exact_geometric(S0, K, r, T, sigma, rho):
    """Exact price of a geometric-average basket call.

    Parameters
    ----------
    S0, sigma : (d,) arrays   per-asset spots and volatilities
    rho : (d, d) array        correlation matrix
    K, r, T : floats

    Returns
    -------
    float
    """
    sigma_hat, mu_hat = effective_params(sigma, rho, r)
    S0_hat = np.exp(np.mean(np.log(S0)))          # geometric mean (overflow-safe)
    q = r - mu_hat                                # effective dividend yield
    d1 = (np.log(S0_hat / K) + (r - q + 0.5 * sigma_hat**2) * T) / (sigma_hat * np.sqrt(T))
    d2 = d1 - sigma_hat * np.sqrt(T)
    return np.exp(-q * T) * S0_hat * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def exact_digital_geometric(S0, K, r, T, sigma, rho):
    """Exact price of a cash-or-nothing digital call on the geometric basket.

    Pays 1 if the geometric average exceeds K at maturity, else 0.
    """
    sigma_hat, mu_hat = effective_params(sigma, rho, r)
    S0_hat = np.exp(np.mean(np.log(S0)))
    q = r - mu_hat
    d2 = (np.log(S0_hat / K) + (r - q - 0.5 * sigma_hat**2) * T) / (sigma_hat * np.sqrt(T))
    return np.exp(-r * T) * norm.cdf(d2)


# --------------------------------------------------------------------------
# Monte Carlo (the honest baseline)
# --------------------------------------------------------------------------
def basket_mc(ST, K, r, T, kind="arithmetic"):
    """Monte Carlo price of a basket call from terminal asset values.

    Parameters
    ----------
    ST : (N, d) array   simulated terminal prices
    kind : str          "arithmetic" or "geometric"

    Returns
    -------
    (price, standard_error)
    """
    if kind == "arithmetic":
        avg = ST.mean(axis=1)
    else:
        avg = np.exp(np.mean(np.log(ST), axis=1))
    disc = np.exp(-r * T) * np.maximum(avg - K, 0.0)
    return disc.mean(), disc.std(ddof=1) / np.sqrt(len(disc))


def control_variate_benchmark(S0, K, r, T, sigma, rho, N_paths=1_000_000, seed=999):
    """Variance-reduced MC price of the arithmetic basket.

    Uses the geometric basket (exact price known) as a control variate,
    which cancels most of the Monte Carlo variance.

    Returns
    -------
    (cv_price, cv_se, plain_price, plain_se)
    """
    from .simulate import simulate_terminal
    ST = simulate_terminal(S0, sigma, rho, r, T, N_paths, seed=seed)
    disc = np.exp(-r * T)
    arith = disc * np.maximum(ST.mean(axis=1) - K, 0.0)
    geo = disc * np.maximum(np.exp(np.mean(np.log(ST), axis=1)) - K, 0.0)
    geo_exact = exact_geometric(S0, K, r, T, sigma, rho)
    c = np.cov(arith, geo, ddof=1)[0, 1] / np.var(geo, ddof=1)
    cv = arith - c * (geo - geo_exact)
    return (cv.mean(), cv.std(ddof=1) / np.sqrt(N_paths),
            arith.mean(), arith.std(ddof=1) / np.sqrt(N_paths))


def fd_grid_points(d, M=100):
    """Number of grid points a finite-difference scheme would need: M**d.

    Illustrates the curse of dimensionality (10**200 at d=100, M=100).
    """
    return M ** d

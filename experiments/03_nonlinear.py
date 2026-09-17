"""Nonlinear pricing: the headline result, with verification.

Prices the differing-rates (Bergman) problem up to d=100 -- a nonlinear
problem with no direct Monte Carlo estimator -- and verifies each price
with the comparison-theorem sandwich: the true price must lie between the
two linear prices at the lending and borrowing rates.
"""

import numpy as np
from deepbsde.references import exact_geometric
from deepbsde.solver import train_bsde

if __name__ == "__main__":
    Rl, Rb, mu = 0.04, 0.06, 0.06
    for d in [20, 50, 100]:
        S0 = np.full(d, 100.0); sigma = np.full(d, 0.2)
        rho = np.full((d, d), 0.3); np.fill_diagonal(rho, 1.0)
        K, T = 100.0, 1.0

        low = exact_geometric(S0, K, Rl, T, sigma, rho)   # bracket at lending rate
        high = exact_geometric(S0, K, Rb, T, sigma, rho)  # bracket at borrowing rate

        prices = [train_bsde(d, generator="nonlinear", Rl=Rl, Rb=Rb, mu=mu,
                             seed=s, verbose=False)["Y0"]
                  for s in [12345, 23456, 34567]]
        mean_p = np.mean(prices)
        inside = low <= mean_p <= high
        print(f"d={d:>3}  nonlinear={mean_p:.4f}  "
              f"bracket=[{low:.4f}, {high:.4f}]  inside={inside}")

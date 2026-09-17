"""Scaling and the honest baseline: dimensionality is not the obstacle.

Times Deep BSDE and plain Monte Carlo to reach a target accuracy across
dimensions up to d=100, and states the finite-difference grid size for
comparison. Reproduces the Phase 2 reframing: MC is orders of magnitude
faster and its cost does not grow with d.
"""

import time
import numpy as np
from deepbsde.simulate import simulate_terminal
from deepbsde.references import exact_geometric, fd_grid_points
from deepbsde.solver import train_bsde


def mc_time_to_accuracy(d, eps=1e-2, pilot_N=100_000, seed=999):
    """Time plain MC to reach relative error eps on the geometric basket."""
    S0 = np.full(d, 100.0); sigma = np.full(d, 0.2)
    rho = np.full((d, d), 0.3); np.fill_diagonal(rho, 1.0)
    K, r, T = 100.0, 0.05, 1.0
    exact = exact_geometric(S0, K, r, T, sigma, rho)

    ST = simulate_terminal(S0, sigma, rho, r, T, pilot_N, seed=seed)
    payoff = np.exp(-r * T) * np.maximum(np.exp(np.mean(np.log(ST), axis=1)) - K, 0.0)
    N_req = int(np.ceil((payoff.std(ddof=1) / (eps * exact)) ** 2))

    start = time.time()
    ST2 = simulate_terminal(S0, sigma, rho, r, T, N_req, seed=seed + 1)
    p2 = np.exp(-r * T) * np.maximum(np.exp(np.mean(np.log(ST2), axis=1)) - K, 0.0)
    return {"d": d, "N_required": N_req, "time_s": time.time() - start,
            "price": p2.mean()}


if __name__ == "__main__":
    print(f"{'d':>4} {'FD points':>12} {'MC N':>10} {'MC time':>10} {'DBSDE time':>12}")
    for d in [5, 20, 50, 100]:
        mc = mc_time_to_accuracy(d)
        db = train_bsde(d, payoff="geometric", verbose=False)
        print(f"{d:>4} {'10^%d' % (2 * d):>12} {mc['N_required']:>10,} "
              f"{mc['time_s']:>9.3f}s {db['time_s']:>11.1f}s")

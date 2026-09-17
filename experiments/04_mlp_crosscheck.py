"""MLP cross-check: an independent method confirms the nonlinear price.

Validates the MLP estimator against the Black-Scholes price and delta
(linear case), then runs it on the Bergman generator at d=1 and compares
to the Deep BSDE nonlinear price. Agreement between two methods sharing no
machinery is the strongest available verification.
"""

import numpy as np
from deepbsde.mlp import mlp
from deepbsde.solver import train_bsde

if __name__ == "__main__":
    # --- validate MLP against known price and delta (linear) ---
    vals, grads = [], []
    for s in range(5):
        rng = np.random.default_rng(s)
        v, g = mlp(3, 0.0, 100.0, M=20, K=100.0, r=0.05, T=1.0, sigma=0.2,
                   rng=rng, generator="linear")
        vals.append(v); grads.append(g)
    print(f"MLP value: {np.mean(vals):.4f} +/- {np.std(vals):.4f}   (BS = 10.4506)")
    print(f"MLP delta: {np.mean(grads):.4f} +/- {np.std(grads):.4f}   (BS delta = 0.6368)")

    # --- nonlinear cross-check at d=1 ---
    prices = []
    for s in range(10):
        rng = np.random.default_rng(s)
        v, _ = mlp(3, 0.0, 100.0, M=20, K=100.0, r=0.05, T=1.0, sigma=0.2,
                   rng=rng, generator="bergman", Rl=0.04, Rb=0.06, mu=0.06)
        prices.append(v)
    mlp_price = np.mean(prices)

    db = [train_bsde(1, generator="nonlinear", Rl=0.04, Rb=0.06, mu=0.06,
                     seed=s, verbose=False)["Y0"] for s in [12345, 23456, 34567]]
    db_price = np.mean(db)

    print(f"\nMLP       d=1 Bergman: {mlp_price:.4f}")
    print(f"Deep BSDE d=1 Bergman: {db_price:.4f}")
    print(f"relative difference: {abs(mlp_price - db_price) / db_price:.2%}")

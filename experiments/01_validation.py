"""Validation: the solver recovers known prices from d=1 to d=10.

Prices the geometric basket (which has a closed form) across dimensions
and reports the relative error against the exact price, for three seeds.
Reproduces the Phase 1 validation table.
"""

import numpy as np
from deepbsde.solver import train_bsde

if __name__ == "__main__":
    for d in [1, 3, 5, 10]:
        errs = []
        for seed in [12345, 23456, 34567]:
            res = train_bsde(d, payoff="geometric", seed=seed, verbose=False)
            errs.append(res["rel_err"])
        print(f"d={d:>3}: mean rel_err = {np.mean(errs):.4%}  "
              f"(range {min(errs):.4%}-{max(errs):.4%})")

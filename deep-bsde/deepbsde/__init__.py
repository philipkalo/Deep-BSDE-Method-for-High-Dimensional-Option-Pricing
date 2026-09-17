"""Deep BSDE for high-dimensional option pricing.

A neural-network solver for the semilinear parabolic PDEs that arise in
option pricing, together with the reference methods used to validate it
(closed-form prices, Monte Carlo, and a multilevel-Picard estimator).

Submodules
----------
references   : closed-form and Monte Carlo reference prices.
simulate     : geometric-Brownian-motion path simulators.
solver       : the Deep BSDE solver (forward passes + training loop).
mlp          : an independent multilevel-Picard estimator (cross-check).
"""

from . import references, simulate, solver, mlp  # noqa: F401

__all__ = ["references", "simulate", "solver", "mlp"]

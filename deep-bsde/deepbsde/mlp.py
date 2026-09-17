"""Multilevel-Picard (MLP) estimator -- an independent cross-check.

MLP solves the same semilinear PDE by a recursive Monte Carlo scheme
(E, Hutzenthaler, Jentzen & Kruse). It shares no machinery with the
Deep BSDE solver -- no network, no training -- so agreement between the
two is strong, independent evidence that the nonlinear price is correct.

The estimator returns *both* the value u(t, x) and its gradient du/dx.
The gradient is obtained via a Malliavin weight Z / (x * sigma * sqrt(tau)),
which lets the same samples estimate the derivative -- needed because the
Bergman generator depends on the hedge Z = sigma * x * du/dx.

Cost grows like (2M)**n, so MLP is used only in low dimension (d = 1-2),
where it is cheap and where its purpose (an independent check) is served.
"""

import numpy as np


def generator_f(u, Z, generator, r, Rl, Rb, mu, sigma):
    """Pointwise driver f(u, Z) for the MLP recursion."""
    if generator == "zero":
        return 0.0
    if generator == "linear":
        return -r * u
    if generator == "bergman":
        return (-Rl * u
                - ((mu - Rl) / sigma) * Z
                + (Rb - Rl) * max(Z / sigma - u, 0.0))
    raise ValueError(f"unknown generator: {generator}")


def mlp(n, t, x, M, K, r, T, sigma, rng,
        generator="linear", Rl=0.04, Rb=0.06, mu=0.06):
    """MLP estimate of (u(t, x), du/dx(t, x)) at level n, in one dimension.

    Parameters
    ----------
    n : int             level (higher = more accurate; cost ~ (2M)**n)
    t, x : float        evaluate the solution here
    M : int             base sample count
    rng : np.random.Generator
    generator : str     "zero", "linear", or "bergman"

    Returns
    -------
    (value, gradient) : floats
    """
    if n == 0:
        return 0.0, 0.0

    # ---- terminal term: value and Malliavin gradient ----
    tau = T - t
    Z = rng.standard_normal(M ** n)
    X_T = x * np.exp((r - 0.5 * sigma**2) * tau + sigma * np.sqrt(tau) * Z)
    payoff = np.maximum(X_T - K, 0.0)
    value_term = np.mean(payoff)
    grad_term = np.mean(payoff * Z / (x * sigma * np.sqrt(tau)))

    # ---- telescoping correction sum over levels 0 .. n-1 ----
    val_corr = grad_corr = 0.0
    for l in range(n):
        num = M ** (n - l)
        v_acc = g_acc = 0.0
        for _ in range(num):
            theta = rng.uniform(t, T)
            h = theta - t
            Zi = rng.standard_normal()
            X_theta = x * np.exp((r - 0.5 * sigma**2) * h + sigma * np.sqrt(h) * Zi)

            u_l, du_l = mlp(l, theta, X_theta, M, K, r, T, sigma, rng,
                            generator, Rl, Rb, mu)
            if l > 0:
                u_lm1, du_lm1 = mlp(l - 1, theta, X_theta, M, K, r, T, sigma, rng,
                                    generator, Rl, Rb, mu)
            else:
                u_lm1, du_lm1 = 0.0, 0.0

            Z_l = sigma * X_theta * du_l
            Z_lm1 = sigma * X_theta * du_lm1
            diff = (generator_f(u_l, Z_l, generator, r, Rl, Rb, mu, sigma)
                    - generator_f(u_lm1, Z_lm1, generator, r, Rl, Rb, mu, sigma))

            w = Zi / (x * sigma * np.sqrt(h))       # Malliavin weight for the gradient
            v_acc += diff
            g_acc += diff * w

        val_corr += (T - t) * v_acc / num           # (T - t): MC integral over time
        grad_corr += (T - t) * g_acc / num

    return value_term + val_corr, grad_term + grad_corr

"""The Deep BSDE solver.

The pricing PDE is recast (nonlinear Feynman-Kac) as a backward SDE

    dY_t = -f(t, X_t, Y_t, Z_t) dt + Z_t^T dW_t,   Y_T = g(X_T),

where Y is the option price and Z the hedging strategy. Following
E, Han & Jentzen (2017), Y0 and the function Z(t, x) are treated as
unknowns: the recursion is run *forward* and the terminal mismatch

    L(theta) = E | Y_N - g(X_N) |^2

is minimised. A neural network represents Z.

Two network architectures are provided (a single merged network taking
(t, x), and one sub-network per timestep) and two generators (linear
discounting, and the Bergman differing-rates nonlinearity).
"""

import time
import numpy as np
import tensorflow as tf
from tensorflow import keras

from .references import exact_geometric, exact_digital_geometric
from .simulate import simulate_paths_multi


# --------------------------------------------------------------------------
# Networks
# --------------------------------------------------------------------------
def build_z_network(d, width=None):
    """A single merged network Z(t, x): input (1 + d), output d.

    Width defaults to d + 10 (the E-Han-Jentzen heuristic).
    """
    width = width if width is not None else d + 10
    return keras.Sequential([
        keras.layers.Input(shape=(1 + d,)),
        keras.layers.Dense(width, activation="tanh"),
        keras.layers.Dense(width, activation="tanh"),
        keras.layers.Dense(d),
    ])


def build_z_networks_per_step(d, N_steps, width=None):
    """A list of N_steps independent networks, one per timestep.

    Each maps x -> Z (input d, output d); time is implicit in the index.
    """
    width = width if width is not None else d + 10
    return [
        keras.Sequential([
            keras.layers.Input(shape=(d,)),
            keras.layers.Dense(width, activation="tanh"),
            keras.layers.Dense(width, activation="tanh"),
            keras.layers.Dense(d),
        ])
        for _ in range(N_steps)
    ]


# --------------------------------------------------------------------------
# Forward passes (one per architecture x generator combination)
# --------------------------------------------------------------------------
def bsde_forward_multi(Y0, z_net, X, dW, r, T, N_steps):
    """Merged-network forward pass, linear generator f = -r Y."""
    dt = T / N_steps
    Y = tf.fill([tf.shape(X)[0]], Y0)
    for n in range(N_steps):
        t_n = tf.fill([tf.shape(X)[0], 1], n * dt)
        inp = tf.concat([t_n, X[:, n, :]], axis=1)
        Z_n = z_net(inp)
        noise = tf.reduce_sum(Z_n * dW[:, n, :], axis=1)
        Y = Y + r * Y * dt + noise
    return Y


def bsde_forward_per_step(Y0, z_nets, X, dW, r, T, N_steps):
    """Per-timestep-network forward pass, linear generator."""
    dt = T / N_steps
    Y = tf.fill([tf.shape(X)[0]], Y0)
    for n in range(N_steps):
        Z_n = z_nets[n](X[:, n, :])
        noise = tf.reduce_sum(Z_n * dW[:, n, :], axis=1)
        Y = Y + r * Y * dt + noise
    return Y


def bsde_forward_nonlinear(Y0, z_net, X, dW, Rl, Rb, mu, sigma, T, N_steps):
    """Merged-network forward pass, Bergman differing-rates generator.

    f = -Rl Y - ((mu - Rl)/sigma) sum(Z)
          + (Rb - Rl) max( sum(Z)/sigma - Y, 0 )

    The (.)+ term charges the borrowing spread only on funds actually
    borrowed -- a nonlinearity in Y and Z that plain Monte Carlo cannot
    price without nested simulation. It costs one line here.
    """
    dt = T / N_steps
    Y = tf.fill([tf.shape(X)[0]], Y0)
    for n in range(N_steps):
        t_n = tf.fill([tf.shape(X)[0], 1], n * dt)
        inp = tf.concat([t_n, X[:, n, :]], axis=1)
        Z_n = z_net(inp)
        noise = tf.reduce_sum(Z_n * dW[:, n, :], axis=1)
        z_sum = tf.reduce_sum(Z_n, axis=1)
        f = (-Rl * Y
             - ((mu - Rl) / sigma) * z_sum
             + (Rb - Rl) * tf.maximum(z_sum / sigma - Y, 0.0))
        Y = Y - f * dt + noise
    return Y


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def train_bsde(d, architecture="merged", payoff="geometric", generator="linear",
               Rl=0.04, Rb=0.06, mu=0.06, K=100.0, r=0.05, T=1.0, N_steps=50,
               N_paths=20000, epochs=2000, lr=1e-2, tol=1e-3, seed=12345,
               verbose=True):
    """Train a Deep BSDE solver and return a results dict.

    Parameters
    ----------
    d : int                     number of assets
    architecture : str          "merged" or "per_timestep"
    payoff : str                "geometric", "arithmetic", or "digital"
    generator : str             "linear" or "nonlinear" (Bergman rates)
    Rl, Rb, mu : float          lending rate, borrowing rate, drift (nonlinear only)
    ... standard market / training hyperparameters ...

    Returns
    -------
    dict with keys: d, seed, architecture, payoff, generator, Y0, exact,
    rel_err, best_error, final_loss, time_to_tol, time_s.

    ``exact`` is None when no closed form exists (arithmetic payoff, or the
    nonlinear generator); in those cases rel_err/best_error are None and the
    result is verified externally (comparison bounds, MLP, control variate).
    """
    tf.random.set_seed(seed)

    S0 = np.full(d, 100.0)
    sigma = np.full(d, 0.2)
    rho = np.full((d, d), 0.3)
    np.fill_diagonal(rho, 1.0)

    X_np, dW_np = simulate_paths_multi(S0, sigma, rho, r, T, N_steps, N_paths, seed=seed)
    X = tf.constant(X_np, dtype=tf.float32)
    dW = tf.constant(dW_np, dtype=tf.float32)

    # ---- payoff and (if available) exact reference ----
    if payoff == "geometric":
        avg_T = tf.exp(tf.reduce_mean(tf.math.log(X[:, -1, :]), axis=1))
        g_XN = tf.maximum(avg_T - K, 0.0)
        exact = exact_geometric(S0, K, r, T, sigma, rho)
    elif payoff == "arithmetic":
        avg_T = tf.reduce_mean(X[:, -1, :], axis=1)
        g_XN = tf.maximum(avg_T - K, 0.0)
        exact = None
    elif payoff == "digital":
        avg_T = tf.exp(tf.reduce_mean(tf.math.log(X[:, -1, :]), axis=1))
        g_XN = tf.cast(avg_T > K, tf.float32)
        exact = exact_digital_geometric(S0, K, r, T, sigma, rho)
    else:
        raise ValueError(f"unknown payoff: {payoff}")

    # the nonlinear price has no closed form -> verified externally
    if generator == "nonlinear":
        exact = None

    # ---- trainable price and optimiser ----
    if payoff == "digital":
        y0_init = float(exact_digital_geometric(S0, K, r, T, sigma, rho))
    else:
        y0_init = float(exact) if exact is not None else \
            float(exact_geometric(S0, K, r, T, sigma, rho))
    Y0 = tf.Variable(y0_init, trainable=True, dtype=tf.float32)
    optimizer = keras.optimizers.Adam(learning_rate=lr)

    # ---- build network and select the forward pass ----
    if architecture == "merged":
        z_net = build_z_network(d)
        net_call = lambda: bsde_forward_multi(Y0, z_net, X, dW, r=r, T=T, N_steps=N_steps)
        nl_call = lambda: bsde_forward_nonlinear(
            Y0, z_net, X, dW, Rl=Rl, Rb=Rb, mu=mu,
            sigma=float(sigma[0]), T=T, N_steps=N_steps)
        get_vars = lambda: [Y0] + z_net.trainable_variables
    elif architecture == "per_timestep":
        z_nets = build_z_networks_per_step(d, N_steps)
        net_call = lambda: bsde_forward_per_step(Y0, z_nets, X, dW, r=r, T=T, N_steps=N_steps)
        nl_call = None  # nonlinear per-timestep not used in this study
        get_vars = lambda: [Y0] + [v for net in z_nets for v in net.trainable_variables]
    else:
        raise ValueError(f"unknown architecture: {architecture}")

    forward = nl_call if generator == "nonlinear" else net_call

    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean((forward() - g_XN) ** 2)
        variables = get_vars()
        optimizer.apply_gradients(zip(tape.gradient(loss, variables), variables))
        return loss

    # ---- train, tracking time-to-tolerance robustly ----
    start = time.time()
    time_to_tol, best_error, consecutive = None, np.inf, 0
    for epoch in range(epochs):
        loss = train_step()
        if exact is not None and epoch % 50 == 0:
            rel_error = abs(float(Y0.numpy()) - exact) / exact
            best_error = min(best_error, rel_error)
            if rel_error < tol:
                consecutive += 1
                if consecutive >= 3 and time_to_tol is None:
                    time_to_tol = time.time() - start
            else:
                consecutive = 0
    elapsed = time.time() - start

    Y0_val = float(Y0.numpy())
    result = {
        "d": d, "seed": seed, "architecture": architecture,
        "payoff": payoff, "generator": generator,
        "Y0": Y0_val,
        "exact": float(exact) if exact is not None else None,
        "rel_err": abs(Y0_val - exact) / exact if exact is not None else None,
        "best_error": best_error if exact is not None else None,
        "final_loss": float(loss.numpy()),
        "time_to_tol": time_to_tol,
        "time_s": elapsed,
    }
    if verbose:
        ex = f"{result['exact']:.4f}" if result["exact"] is not None else "  n/a"
        re = f"{result['rel_err']:.4%}" if result["rel_err"] is not None else "   n/a"
        print(f"d={d:>3} {architecture:>12} {payoff:>10} {generator:>9}  "
              f"Y0={Y0_val:.4f}  exact={ex}  rel_err={re}  "
              f"loss={result['final_loss']:.2f}  time={elapsed:.1f}s")
    return result

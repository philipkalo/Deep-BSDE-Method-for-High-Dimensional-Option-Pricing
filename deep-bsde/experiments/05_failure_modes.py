"""Failure modes: reliability is governed by smoothness, not size.

Two stress tests:

* Digital (discontinuous) payoff -- the price survives but the hedge
  breaks near maturity, where the true hedge is singular.
* Long maturity (T=3 vs T=1, fixed dt) -- accuracy is unaffected.

Together: the method's fragility depends on payoff smoothness, not on
problem size (dimension or horizon).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from deepbsde.references import exact_digital_geometric
from deepbsde.simulate import simulate_paths_multi
from deepbsde.solver import build_z_network, bsde_forward_multi, train_bsde

import tensorflow as tf
from tensorflow import keras


def analytic_Z_digital(X_n, t_n, K, r, T, sigma):
    """Analytic hedge of a digital call; singular as t -> T (pdf / sqrt(tau))."""
    tau = T - t_n
    d2 = (np.log(X_n / K) + (r - 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    delta = np.exp(-r * tau) * norm.pdf(d2) / (X_n * sigma * np.sqrt(tau))
    return sigma * X_n * delta


def digital_hedge_error_curve(seed=12345):
    """Train a d=1 digital solver and plot hedge error vs time."""
    K, r, T, N_steps, N_paths = 100.0, 0.05, 1.0, 50, 20000
    sigma_1 = 0.2
    S0 = np.full(1, 100.0); sigma = np.full(1, 0.2); rho = np.full((1, 1), 1.0)

    tf.random.set_seed(seed)
    X_np, dW_np = simulate_paths_multi(S0, sigma, rho, r, T, N_steps, N_paths, seed=seed)
    X = tf.constant(X_np, dtype=tf.float32); dW = tf.constant(dW_np, dtype=tf.float32)
    avg_T = tf.exp(tf.reduce_mean(tf.math.log(X[:, -1, :]), axis=1))
    g_XN = tf.cast(avg_T > K, tf.float32)
    exact = exact_digital_geometric(S0, K, r, T, sigma, rho)

    z_net = build_z_network(1)
    Y0 = tf.Variable(float(exact), trainable=True, dtype=tf.float32)
    opt = keras.optimizers.Adam(1e-2)
    for _ in range(2000):
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean((bsde_forward_multi(Y0, z_net, X, dW,
                                   r=r, T=T, N_steps=N_steps) - g_XN) ** 2)
        v = [Y0] + z_net.trainable_variables
        opt.apply_gradients(zip(tape.gradient(loss, v), v))

    dt = T / N_steps
    errors, times = [], []
    for n in range(1, N_steps):
        t_n = n * dt
        X_n = X_np[:, n, 0]
        inp = np.column_stack([np.full(len(X_n), t_n), X_n]).astype(np.float32)
        Z_net = z_net(inp).numpy()[:, 0]
        Z_ana = analytic_Z_digital(X_n, t_n, K, r, T, sigma_1)
        errors.append(np.mean((Z_net - Z_ana) ** 2) / np.mean(Z_ana ** 2))
        times.append(t_n)

    plt.figure(figsize=(6, 4.5))
    plt.plot(times, errors, marker="o", ms=3)
    plt.xlabel("time"); plt.ylabel(r"relative $L^2$ hedge error")
    plt.title("Digital payoff: hedge error vs time (d=1)")
    plt.tight_layout()
    plt.savefig("digital_hedge_error.png", dpi=200)
    print(f"digital price: {float(Y0.numpy()):.4f}  exact: {exact:.4f}")
    print(f"max hedge error: {max(errors):.3f} at t={times[int(np.argmax(errors))]:.2f}")


def long_maturity_test():
    """Compare accuracy at T=1 vs T=3 (fixed dt) on the geometric basket."""
    for T_val, N in [(1.0, 50), (3.0, 150)]:
        errs = [train_bsde(5, payoff="geometric", T=T_val, N_steps=N,
                           seed=s, verbose=False)["rel_err"]
                for s in [12345, 23456, 34567]]
        print(f"T={T_val}: mean rel_err = {np.mean(errs):.4%}")


if __name__ == "__main__":
    print("=== digital payoff ==="); digital_hedge_error_curve()
    print("\n=== long maturity ==="); long_maturity_test()

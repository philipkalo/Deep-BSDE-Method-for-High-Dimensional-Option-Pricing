# Deep BSDE for High-Dimensional Option Pricing

A neural-network solver for the high-dimensional PDEs that arise in option
pricing, with a full validation and failure-mode study.

The central finding: **dimensionality is not what motivates neural PDE
solvers — nonlinearity is.** On linear problems, plain Monte Carlo already
prices options in 100 dimensions faster than Deep BSDE and with cost that
*falls* as dimension grows. The method earns its place on *nonlinear*
problems, where no direct Monte Carlo estimator exists — and there it is
verified here, in 100 dimensions, two independent ways.

## What's here

| Result | Script | Finding |
|---|---|---|
| Validation | `experiments/01_validation.py` | Recovers known prices to <0.5% up to d=10 |
| Scaling baseline | `experiments/02_scaling_baseline.py` | Monte Carlo beats Deep BSDE by ~10⁴× on linear problems |
| Nonlinear pricing | `experiments/03_nonlinear.py` | Prices differing-rates options at d=100, verified by comparison bounds |
| MLP cross-check | `experiments/04_mlp_crosscheck.py` | Independent estimator agrees to ~0.8% at d=1 |
| Failure modes | `experiments/05_failure_modes.py` | Fragility tracks payoff smoothness, not problem size |

## The method in one paragraph

Nonlinear Feynman–Kac recasts the pricing PDE as a backward stochastic
differential equation, `dY = -f(t,X,Y,Z) dt + Zᵀ dW`, where `Y` is the
option price and `Z` the hedge. The solver treats `Y₀` and the function
`Z(t,x)` as unknowns, simulates the recursion forward along Monte Carlo
paths, and trains a neural network to minimise the terminal mismatch
`E|Y_N − g(X_N)|²`. Because the state space is sampled rather than gridded,
cost grows polynomially — not exponentially — with dimension. Crucially, the
driver `f` may be nonlinear at no extra cost, since it is evaluated
pointwise; that is where the method beats Monte Carlo.

## Package layout

```
deepbsde/
  references.py   closed-form (Black–Scholes, geometric basket) and Monte Carlo prices
  simulate.py     correlated GBM path simulators
  solver.py       the Deep BSDE solver: networks, forward passes, training loop
  mlp.py          independent multilevel-Picard estimator (cross-check)
experiments/      runnable scripts, one per result above
figures/          figures used in the accompanying poster
```

## Installation

```bash
git clone https://github.com/<your-username>/deep-bsde.git
cd deep-bsde
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Running

Each experiment is self-contained. Run from the repository root so the
`deepbsde` package is importable:

```bash
python -m experiments.01_validation
python -m experiments.03_nonlinear
```

Low-dimensional experiments run in minutes on a CPU; d=100 runs take
noticeably longer and benefit from a GPU.

## Key results

- Deep BSDE recovers the Black–Scholes price and the geometric-basket price
  to within 0.5% from d=1 to d=100.
- On these linear problems Monte Carlo is ~4 orders of magnitude faster at
  every dimension, and its required sample count *decreases* with dimension
  (diversification narrows the payoff). Dimensionality alone does not
  justify a neural method.
- Under asymmetric borrowing/lending rates (Bergman), where Monte Carlo
  would need nested simulation whose cost grows doubly-exponentially, Deep
  BSDE prices the option at d=100. Every price lies inside its
  comparison-theorem bracket, and an independent multilevel-Picard estimator
  agrees to ~0.8% at d=1.
- Failure modes track payoff *smoothness*, not problem *size*: a
  discontinuous digital payoff breaks the hedge near maturity (~47× the
  error of a smooth call) while long maturities are tolerated. The training
  loss does not certify accuracy — it measures the hedge, which is largely
  decoupled from the price.

## References

- E, Han & Jentzen (2017), *Deep learning-based numerical methods for
  high-dimensional parabolic PDEs and BSDEs.*
- E, Hutzenthaler, Jentzen & Kruse (2019), *Multilevel Picard iterations.*
- Bergman (1995), *Option pricing with differential borrowing and lending
  rates.*

## Author

Filippos Akylas Kaloudis — Physics with Theoretical Physics, Imperial College London.

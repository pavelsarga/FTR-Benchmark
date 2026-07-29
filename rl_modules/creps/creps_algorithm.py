"""Constrained REPS (Pecka et al. 2016, "Autonomous Flipper Control with Safety
Constraints") core math: dual optimization, sample weights, and the weighted
maximum-likelihood upper-level Gaussian refit. Pure numpy/scipy — no TorchRL,
gymnasium, or Isaac imports, so this module is importable and unit-testable without a
simulator or GPU.

Context-free formulation (matching the paper's own reported experiment, which never
uses context — s is fixed to 0 throughout): the upper-level policy is a single global
Gaussian q(omega) = N(mu, Sigma), so the paper's theta/phi(s) dual terms (Contextual
REPS's context-preservation constraint) do not appear here at all.

The dual formula, gamma sign convention, and solver robustness scheme below were
rewritten to match the authors' own reference implementation of Constrained REPS
(`constrained_reps/constrained_reps.py` and `constrained_reps/constraint.py` in
paper_repos/tradr-simulation/safe_exploration — the ST flipper task itself isn't in
that checkout, see train_creps.py's module docstring, but the core CREPS algorithm code
is a faithful, more complete reference for the method itself). In particular:

Constraint sign convention: each column of `constraints` is a COMPLIANCE indicator in
[0, 1] (1 = safe / within mechanical limits, 0 = violated), and each entry of `delta` is
a LOWER bound on that column's expected value under the new distribution:
E_p[constraints[:, k]] >= delta[k] -- this is exactly `CREPSConstraint(is_min_bound=True)`
in the reference repo (their default, and the only mode used by every constraint they
actually construct), which is checked there via `value >= self.bound`
(constraint.py:60-61). This differs from the paper's literal equation (1)/(2) (written
as an upper bound on expected UNSAFETY, sum p(1-C) <= delta) but matches both the
reference repo's actual `is_min_bound` semantics and what the paper's own Section IV
toy-example sanity check verifies numerically ("the average safety ... is 0.6064,
indeed above the required safety bound" of delta=0.6 -- compliance compared directly
against delta, not against 1-delta).

Dual (rederived to match the reference repo's exact sign convention -- their
`constrained_reps.py:dual_function`, with the context/theta/phi terms dropped since we
never use context, matching their own `context=None` codepath):
    Z_i(eta, gamma) = exp((R_i - gamma . C_i) / eta)
    g(eta, gamma) = eta * log(mean_i Z_i) + eta*epsilon + gamma . delta
    dg/deta        = epsilon + log(mean(Z)) - sum(Z*(R - gamma.C)) / (eta * sum(Z))
    dg/dgamma_k     = delta_k - sum(Z * C_k) / sum(Z)
    p_i (closed form, Eq. 11 analogue) = Z_i / sum(Z)
gamma is bounded <= 0 (their `constraint.py`'s `is_min_bound=True` -> optimizer bound
`(None, 0.0)`, `constrained_reps.py:441-446`), the mirror image of a naive "gamma>=0"
convention -- both are the same convex program, just with gamma negated; this file uses
their sign so it lines up 1:1 with their code if ever cross-referenced.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

# Reference repo's CREPSParameters defaults (constrained_reps.py:682-683).
DEFAULT_ETA_MIN = 1e-2
DEFAULT_DUAL_VAR_MAX = 1e6
DEFAULT_MAX_OPTIMIZATION_TRIALS = 10


def _logsumexp(x: np.ndarray) -> tuple[float, np.ndarray]:
    """Returns (log(mean(exp(x))), the shifted-but-unnormalized exp(x - max(x)))."""
    m = np.max(x)
    shifted = np.exp(x - m)
    log_mean = m + np.log(np.mean(shifted))
    return log_mean, shifted


def _dual_value_and_grad(
    params: np.ndarray, rewards: np.ndarray, constraints: np.ndarray, kl_epsilon: float, delta: np.ndarray,
) -> tuple[float, np.ndarray]:
    eta = params[0]
    gamma = params[1:]
    inner = rewards - constraints @ gamma  # (N,)
    exponent = inner / eta
    log_mean_z, z_shifted = _logsumexp(exponent)  # z_shifted_i = exp(exponent_i - max)
    sum_z = np.sum(z_shifted)  # shares the same max-shift as z_shifted; cancels in ratios below

    g = eta * log_mean_z + eta * kl_epsilon + gamma @ delta

    weighted_inner = np.sum(z_shifted * inner) / sum_z
    d_eta = kl_epsilon + log_mean_z - weighted_inner / eta

    weighted_c = (z_shifted @ constraints) / sum_z  # (K,)
    d_gamma = delta - weighted_c

    grad = np.concatenate([[d_eta], d_gamma])
    return g, grad


def solve_creps_dual(
    rewards: np.ndarray,
    constraints: np.ndarray,
    kl_epsilon: float,
    delta: np.ndarray,
    eta_init: float = 1.0,
    gamma_init: np.ndarray | None = None,
    eta_min: float = DEFAULT_ETA_MIN,
    dual_var_max: float = DEFAULT_DUAL_VAR_MAX,
    max_optimization_trials: int = DEFAULT_MAX_OPTIMIZATION_TRIALS,
    rng: np.random.Generator | None = None,
) -> tuple[float, np.ndarray, dict]:
    """Solves min_{eta>eta_min, gamma<=0} g(eta, gamma) (see module docstring).

    rewards: (N,) float64. constraints: (N, K) float64, compliance indicators in [0,1]
    (see module docstring sign convention). delta: (K,) float64, lower bound per
    constraint column.

    Matches the reference repo's `ConstrainedREPS.solve_dual`/`optimize_dual`
    (constrained_reps.py:378-497) robustness scheme: the first attempt is warm-started
    from (eta_init, gamma_init) -- a project-specific addition on top of their scheme,
    since our outer loop calls this every iteration and warm-starting from the previous
    iteration's solution converges faster in the common case; every subsequent attempt
    (up to `max_optimization_trials` total) uses a fresh random restart exactly like
    theirs (`eta0 ~ U(0.5, 1.5)`, `gamma0 ~ -U(0, 1)` per constraint,
    constrained_reps.py:389-395). A solve only counts as successful if scipy reports
    `success`, every variable is finite, AND every variable's magnitude is below
    `dual_var_max` (constrained_reps.py:473-476) -- a solution pinned at a huge value
    usually means the requested (kl_epsilon, delta) combination has no finite-temperature
    solution (see e.g. the KL-infeasible corner documented in test_creps_algorithm.py),
    not a real optimum.

    Returns (eta, gamma, info) where info carries `success`/`message`/`attempts` for the
    caller to log/warn on non-convergence.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    constraints = np.asarray(constraints, dtype=np.float64)
    n, k = constraints.shape
    assert rewards.shape == (n,), f"rewards shape {rewards.shape} != ({n},)"
    delta = np.asarray(delta, dtype=np.float64)
    assert delta.shape == (k,), f"delta shape {delta.shape} != ({k},)"

    if rng is None:
        rng = np.random.default_rng()
    bounds = [(eta_min, None)] + [(None, 0.0)] * k

    last_message = ""
    for attempt in range(max_optimization_trials):
        if attempt == 0:
            gamma0 = np.zeros(k) if gamma_init is None else np.clip(gamma_init, None, 0.0)
            x0 = np.concatenate([[max(eta_init, eta_min)], gamma0])
        else:
            # Reference repo's random-restart formula (constrained_reps.py:389-395),
            # specialized to our always-is_min_bound (gamma<=0) constraints.
            x0 = np.concatenate([[0.5 + rng.random()], -rng.random(k)])

        result = minimize(
            _dual_value_and_grad,
            x0,
            args=(rewards, constraints, kl_epsilon, delta),
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
        )
        last_message = str(result.message)

        valid = (
            result.success
            and np.isfinite(result.x).all()
            and bool(np.all(np.abs(result.x) < dual_var_max))
        )
        if valid:
            eta = float(result.x[0])
            gamma = result.x[1:].copy()
            info = {"success": True, "message": last_message, "attempts": attempt + 1}
            return eta, gamma, info

    info = {"success": False, "message": last_message, "attempts": max_optimization_trials}
    return eta_init, (np.zeros(k) if gamma_init is None else np.asarray(gamma_init, dtype=np.float64)), info


def compute_sample_weights(rewards: np.ndarray, constraints: np.ndarray, eta: float, gamma: np.ndarray) -> np.ndarray:
    """p_i = Z_i / sum(Z), Z_i = exp((R_i - gamma.C_i)/eta), computed via a max-shifted
    logsumexp to avoid overflow. Returns weights (N,) summing to 1."""
    rewards = np.asarray(rewards, dtype=np.float64)
    constraints = np.asarray(constraints, dtype=np.float64)
    exponent = (rewards - constraints @ gamma) / eta
    _, z_shifted = _logsumexp(exponent)
    return z_shifted / np.sum(z_shifted)


def weighted_gaussian_fit(
    omega_samples: np.ndarray, weights: np.ndarray, min_std: float | np.ndarray = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form weighted maximum-likelihood Gaussian refit (no gradient descent):
        new_mean = sum_i w_i * omega_i
        new_cov  = sum_i w_i * (omega_i - new_mean)(omega_i - new_mean)^T
    `weights` must already sum to 1 (as returned by compute_sample_weights).

    Both moments use the SAME (standard, single-power) weighting. The reference repo's
    `WeightedMLGaussianUpdater` (policy_updaters.py:35-60) instead estimates the mean via
    `scipy`-equivalent `lstsq(diag(w)@S, diag(w)@B)` (estimators.py:34-67), which for a
    constant-only regression (our context-free case, S=ones) works out to a
    w^2-weighted mean (`sum(w_i^2 x_i)/sum(w_i^2)`), while its covariance step still
    weights by plain w around that w^2-weighted mean -- an internal inconsistency in
    their code between the two moments. That behavior was deliberately NOT reproduced
    here (kept as standard, single-power weighted ML throughout, the textbook-correct
    REPS update) since it looks like an unintentional quirk of their least-squares
    formulation rather than a considered design choice.

    The diagonal of new_cov is floored at min_std**2 per-dimension (not a uniform
    eps*I ridge) so no single dimension can collapse to near-zero variance while
    well-conditioned dimensions/off-diagonals are left untouched. The result is then
    symmetrized and defensively projected to the nearest PSD matrix (clip negative
    eigenvalues to 0) -- a literal weighted outer-product sum can end up not-quite-PSD
    in floating point once weights become very peaked (low effective sample size).
    """
    omega_samples = np.asarray(omega_samples, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    n, d = omega_samples.shape
    assert weights.shape == (n,), f"weights shape {weights.shape} != ({n},)"

    mean = weights @ omega_samples  # (D,)
    centered = omega_samples - mean
    cov = (centered * weights[:, None]).T @ centered  # (D, D)
    cov = 0.5 * (cov + cov.T)

    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 0.0, None)
    cov = (eigvecs * eigvals) @ eigvecs.T
    cov = 0.5 * (cov + cov.T)

    min_std_arr = np.broadcast_to(np.asarray(min_std, dtype=np.float64), (d,))
    diag = np.diag(cov).copy()
    floor = min_std_arr ** 2
    np.fill_diagonal(cov, np.maximum(diag, floor))

    return mean, cov


def effective_sample_size(weights: np.ndarray) -> float:
    """ESS = 1 / sum(w_i^2). weights must sum to 1. Log this every outer iteration --
    a collapsing ESS (approaching 1) signals the upper-level distribution's weighting
    is degenerating onto a single sample."""
    weights = np.asarray(weights, dtype=np.float64)
    return float(1.0 / np.sum(weights ** 2))

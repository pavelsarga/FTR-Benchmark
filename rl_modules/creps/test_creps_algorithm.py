"""Pure-Python tests for creps_algorithm.py -- no Isaac Sim, no GPU, no conda env
needed (just numpy/scipy). Run with: pytest rl_modules/creps/test_creps_algorithm.py
"""

import numpy as np
from scipy.optimize import check_grad

from creps_algorithm import (
    _dual_value_and_grad,
    compute_sample_weights,
    effective_sample_size,
    solve_creps_dual,
    weighted_gaussian_fit,
)


def test_paper_figure2_toy_example():
    """Mirrors the paper's own Figure 2 / Section II.C sanity check: 1-D omega, reward
    peaked at an UNSAFE omega=0, safety=1 for omega>0.5, q(omega) has mean=0.3 and
    VARIANCE=0.3 (std=sqrt(0.3)) exactly as the paper states ("both mean and variance
    equal to 0.3"). The paper's own literal (epsilon=0.1, delta=0.6) is a KL-infeasible
    corner for this q in a single reweight: the marginal-safety KL lower bound (by the
    data-processing inequality, independent of sample count) for shifting P(safe) from
    ~0.36 (this q's true safe mass) to 0.6 is ~0.12 nats, already exceeding an epsilon of
    0.1 -- solve_creps_dual correctly reports non-convergence (eta/gamma diverge) in that
    exact corner, which is itself a useful correctness signal (an infeasible target
    genuinely has no finite-temperature solution) but not something to assert on here.
    This test instead uses a comfortably feasible (epsilon, delta) pair and checks the
    qualitative claim the paper's toy example illustrates: the reweighted distribution's
    expected safety hits the requested delta (the paper's "prefers safe, though
    suboptimal choices" behavior), even though the unconstrained reward optimum (omega=0)
    is unsafe.
    """
    rng = np.random.default_rng(0)
    n = 20000
    std = np.sqrt(0.3)  # paper: "both mean and variance equal to 0.3"
    omega = rng.normal(loc=0.3, scale=std, size=(n, 1))
    reward = -(omega[:, 0] ** 2)              # peaked (unsafe) at omega=0
    safety_ok = (omega[:, 0] > 0.5).astype(np.float64)
    constraints = safety_ok[:, None]           # (N, 1) single constraint column

    kl_epsilon, delta = 0.5, 0.5
    eta, gamma, info = solve_creps_dual(reward, constraints, kl_epsilon=kl_epsilon, delta=np.array([delta]))
    assert info["success"], info["message"]
    assert eta > 0
    # gamma <= 0 for an is_min_bound (E_p[C] >= delta) constraint -- matches the
    # reference repo's sign convention (constraint.py/constrained_reps.py), the mirror
    # image of a naive gamma>=0 convention.
    assert gamma[0] <= 0

    weights = compute_sample_weights(reward, constraints, eta, gamma)
    assert np.isclose(weights.sum(), 1.0, atol=1e-9)

    achieved_safety = weights @ constraints[:, 0]
    assert achieved_safety >= delta - 1e-2, f"achieved_safety={achieved_safety} should be >= ~{delta}"

    new_mean, _ = weighted_gaussian_fit(omega, weights)
    assert new_mean[0] > 0.3, "reweighted distribution should shift toward the safe region relative to q's mean"


def test_gradients_match_finite_differences():
    rng = np.random.default_rng(1)
    n, k = 500, 2
    rewards = rng.normal(size=n)
    constraints = rng.integers(0, 2, size=(n, k)).astype(np.float64)
    kl_epsilon, delta = 0.3, np.array([0.7, 0.85])

    def f(x):
        val, _ = _dual_value_and_grad(x, rewards, constraints, kl_epsilon, delta)
        return val

    def grad(x):
        _, g = _dual_value_and_grad(x, rewards, constraints, kl_epsilon, delta)
        return g

    x0 = np.array([1.5, 0.2, 0.3])
    err = check_grad(f, grad, x0)
    assert err < 1e-4, f"analytic gradient mismatch vs finite differences: {err}"


def test_solve_creps_dual_converges_on_random_problem():
    rng = np.random.default_rng(2)
    n, k = 300, 2
    rewards = rng.normal(size=n)
    constraints = rng.uniform(0, 1, size=(n, k))
    eta, gamma, info = solve_creps_dual(rewards, constraints, kl_epsilon=0.5, delta=np.array([0.5, 0.5]))
    assert info["success"]
    assert eta > 0
    assert np.all(gamma <= 0)


def test_weighted_gaussian_fit_variance_floor_and_psd():
    rng = np.random.default_rng(3)
    n, d = 200, 6
    omega = rng.normal(size=(n, d))
    # near-degenerate weights: one sample dominates
    weights = np.full(n, 1e-9)
    weights[0] = 1.0 - 1e-9 * (n - 1)
    weights /= weights.sum()

    mean, cov = weighted_gaussian_fit(omega, weights, min_std=1e-2)
    assert mean.shape == (d,)
    assert cov.shape == (d, d)
    assert np.allclose(cov, cov.T)
    eigvals = np.linalg.eigvalsh(cov)
    assert eigvals.min() >= -1e-8, f"cov not PSD, min eigval={eigvals.min()}"
    assert np.all(np.diag(cov) >= (1e-2) ** 2 - 1e-12)


def test_weighted_gaussian_fit_matches_plain_mean_cov_for_uniform_weights():
    rng = np.random.default_rng(4)
    n, d = 5000, 6
    omega = rng.normal(loc=1.0, scale=2.0, size=(n, d))
    weights = np.full(n, 1.0 / n)
    mean, cov = weighted_gaussian_fit(omega, weights, min_std=1e-6)
    assert np.allclose(mean, omega.mean(axis=0), atol=0.1)
    assert np.allclose(np.diag(cov), omega.var(axis=0), atol=0.5)


def test_solver_reports_failure_and_falls_back_on_infeasible_problem():
    """The KL-infeasible corner documented in test_paper_figure2_toy_example's docstring
    (paper's literal epsilon=0.1, delta=0.6 with q's true safe mass ~0.36 -- shifting to
    0.6 needs ~0.12 nats of KL, exceeding epsilon=0.1). solve_creps_dual should exhaust
    its retries (random restarts, matching the reference repo) and correctly report
    non-convergence rather than silently returning a runaway (huge eta/gamma) solution --
    falling back to the caller-supplied (eta_init, gamma_init) instead."""
    rng = np.random.default_rng(5)
    n = 5000
    std = np.sqrt(0.3)
    omega = rng.normal(loc=0.3, scale=std, size=(n, 1))
    reward = -(omega[:, 0] ** 2)
    safety_ok = (omega[:, 0] > 0.5).astype(np.float64)
    constraints = safety_ok[:, None]

    eta, gamma, info = solve_creps_dual(
        reward, constraints, kl_epsilon=0.1, delta=np.array([0.6]),
        eta_init=1.23, gamma_init=np.array([-0.45]), max_optimization_trials=5,
        rng=np.random.default_rng(6),
    )
    assert not info["success"]
    assert info["attempts"] == 5
    assert eta == 1.23
    assert np.allclose(gamma, [-0.45])


def test_solver_random_restart_recovers_from_bad_warm_start():
    """A warm start far from the optimum (but not infeasible) should still converge once
    the solver falls through to a random restart on a later attempt."""
    rng = np.random.default_rng(7)
    n, k = 400, 1
    rewards = rng.normal(size=n)
    constraints = rng.uniform(0, 1, size=(n, k))

    eta, gamma, info = solve_creps_dual(
        rewards, constraints, kl_epsilon=0.5, delta=np.array([0.5]),
        eta_init=1e5, gamma_init=np.array([-1e5]),  # deliberately bad/extreme warm start
        rng=np.random.default_rng(8),
    )
    assert info["success"]
    assert info["attempts"] >= 1
    assert eta > 0
    assert np.all(np.abs([eta, *gamma]) < 1e4)


def test_effective_sample_size():
    n = 1000
    uniform_weights = np.full(n, 1.0 / n)
    assert np.isclose(effective_sample_size(uniform_weights), n, rtol=1e-6)

    one_hot = np.zeros(n)
    one_hot[0] = 1.0
    assert np.isclose(effective_sample_size(one_hot), 1.0)


if __name__ == "__main__":
    test_paper_figure2_toy_example()
    test_gradients_match_finite_differences()
    test_solve_creps_dual_converges_on_random_problem()
    test_weighted_gaussian_fit_variance_floor_and_psd()
    test_weighted_gaussian_fit_matches_plain_mean_cov_for_uniform_weights()
    test_solver_reports_failure_and_falls_back_on_infeasible_problem()
    test_solver_random_restart_recovers_from_bad_warm_start()
    test_effective_sample_size()
    print("All CREPS algorithm tests passed.")

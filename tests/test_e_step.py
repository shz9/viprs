import numpy as np
import pytest

from viprs.model.vi.e_step_cpp import (
    cpp_e_step,
    cpp_e_step_grid,
    cpp_e_step_mixture,
)


def test_integer_ld_matches_preconverted_floating_ld():
    ld_integer = np.array(
        [100, 20, -10, 20, 100, 30, -10, 30, 100], dtype=np.int8
    )
    ld_floating = ld_integer.astype(np.float64) * 0.01

    def run_e_step(ld_data, dq_scale):
        var_gamma = np.zeros(3, dtype=np.float64)
        var_mu = np.zeros(3, dtype=np.float64)
        eta = np.zeros(3, dtype=np.float64)
        q = np.zeros(3, dtype=np.float64)
        eta_diff = np.zeros(3, dtype=np.float64)

        cpp_e_step(
            np.zeros(3, dtype=np.intc),
            np.array([0, 3, 6, 9], dtype=np.int32),
            ld_data,
            np.array([0.4, -0.25, 0.15], dtype=np.float64),
            var_gamma,
            var_mu,
            eta,
            q,
            eta_diff,
            np.array([-0.2, -0.1, -0.3], dtype=np.float64),
            np.ones(3, dtype=np.float64),
            np.full(3, 0.8, dtype=np.float64),
            np.float64(dq_scale),
            1,
            False,
        )

        return var_gamma, var_mu, eta, q, eta_diff

    integer_result = run_e_step(ld_integer, 0.01)
    floating_result = run_e_step(ld_floating, 1.0)

    for integer_value, floating_value in zip(integer_result, floating_result):
        np.testing.assert_allclose(integer_value, floating_value, rtol=1e-12, atol=1e-12)


def test_global_variance_scaling_preserves_e_step():
    variance_scale = 100.0
    mean_scale = np.sqrt(variance_scale)

    def run_e_step(scaled):
        scale = variance_scale if scaled else 1.0
        sqrt_scale = mean_scale if scaled else 1.0
        var_tau = np.array([120.0, 150.0]) / scale
        var_gamma = np.array([0.2, 0.3])
        var_mu = np.array([0.01, -0.02]) * sqrt_scale
        eta = var_gamma * var_mu
        q = np.array([-0.004, 0.006]) * sqrt_scale
        eta_diff = np.zeros(2)

        cpp_e_step(
            np.zeros(2, dtype=np.intc),
            np.array([0, 2, 4], dtype=np.int32),
            np.array([1.0, 0.2, 0.2, 1.0]),
            np.array([0.04, -0.03]) * sqrt_scale,
            var_gamma,
            var_mu,
            eta,
            q,
            eta_diff,
            np.array([-1.0, -0.8]),
            np.sqrt(0.5 * var_tau),
            np.array([0.9, 0.7]),
            np.float64(1.0),
            1,
            False,
        )
        return var_gamma, var_mu / sqrt_scale, eta / sqrt_scale, q / sqrt_scale

    for unscaled, scaled in zip(run_e_step(False), run_e_step(True)):
        np.testing.assert_allclose(scaled, unscaled, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("low_memory", [False, True])
def test_mixture_skips_negligible_ld_update(low_memory):
    expected_gamma = 1.0 / (1.0 + np.exp(-1.0))
    eta = np.array([expected_gamma + 1e-10], dtype=np.float64)
    old_eta = eta.copy()
    q = np.zeros(1, dtype=np.float64)
    eta_diff = np.ones(1, dtype=np.float64)
    var_gamma = np.zeros((1, 1), dtype=np.float64)
    var_mu = np.zeros((1, 1), dtype=np.float64)

    cpp_e_step_mixture(
        np.zeros(1, dtype=np.intc),
        np.array([0, 1], dtype=np.int32),
        np.ones(1, dtype=np.int8),
        np.ones(1, dtype=np.float64),
        var_gamma,
        var_mu,
        eta,
        q,
        eta_diff,
        np.zeros(1, dtype=np.float64),
        np.zeros((1, 1), dtype=np.float64),
        np.ones((1, 1), dtype=np.float64),
        np.ones((1, 1), dtype=np.float64),
        np.float64(1.0),
        1,
        low_memory,
    )

    np.testing.assert_array_equal(eta_diff, 0.0)
    np.testing.assert_array_equal(q, 0.0)
    np.testing.assert_array_equal(eta, old_eta)
    np.testing.assert_allclose(var_gamma, expected_gamma, rtol=0.0, atol=1e-15)
    np.testing.assert_array_equal(var_mu, 1.0)


@pytest.mark.parametrize("low_memory", [False, True])
def test_grid_skips_negligible_ld_update(low_memory):
    expected_gamma = 1.0 / (1.0 + np.exp(-1.0))
    eta = np.asfortranarray([[expected_gamma + 1e-10]], dtype=np.float64)
    old_eta = eta.copy()
    q = np.asfortranarray([[0.0]], dtype=np.float64)
    eta_diff = np.asfortranarray([[1.0]], dtype=np.float64)
    var_gamma = np.asfortranarray([[0.0]], dtype=np.float64)
    var_mu = np.asfortranarray([[0.0]], dtype=np.float64)

    cpp_e_step_grid(
        np.zeros(1, dtype=np.intc),
        np.array([0, 1], dtype=np.int32),
        np.ones(1, dtype=np.int8),
        np.ones(1, dtype=np.float64),
        var_gamma,
        var_mu,
        eta,
        q,
        eta_diff,
        np.asfortranarray([[0.0]], dtype=np.float64),
        np.asfortranarray([[1.0]], dtype=np.float64),
        np.asfortranarray([[1.0]], dtype=np.float64),
        np.float64(1.0),
        np.zeros(1, dtype=np.intc),
        1,
        low_memory,
    )

    np.testing.assert_array_equal(eta_diff, 0.0)
    np.testing.assert_array_equal(q, 0.0)
    np.testing.assert_array_equal(eta, old_eta)
    np.testing.assert_allclose(var_gamma, expected_gamma, rtol=0.0, atol=1e-15)
    np.testing.assert_array_equal(var_mu, 1.0)

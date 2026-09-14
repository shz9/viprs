from types import SimpleNamespace

import numpy as np

from magenpy.LDLinearOperator import LDLinearOperator
from viprs.model.VIPRS import VIPRS
from viprs.model.VIPRSMix import VIPRSMix
from viprs.model.vi.e_step_cpp import cpp_e_step


def make_mix_model(m=2, k=2):
    model = VIPRSMix.__new__(VIPRSMix)
    model.K = k
    model.d = np.linspace(0.5, 1.0, k, dtype=np.float64)
    model.float_precision = "float64"
    model.order = "C"
    model.fix_params = {}
    model.shapes = {1: (m, k)}
    model.gdl = SimpleNamespace(m=m)
    model.lambda_min = 0.0
    return model


def test_k1_e_step_matches_viprs_and_refreshes_log_tau():
    model = make_mix_model(m=2, k=1)
    model.d = np.ones(1)
    model.pi = np.array([0.1])
    model.tau_beta = np.array([4.0])
    model.sigma_epsilon = 0.8
    model.lambda_min = np.array([0.0, 0.2])
    model.n_per_snp = {1: np.array([[100.0], [80.0]])}
    model.std_beta = {1: np.array([0.1, -0.05])}
    model.ld_left_bound = {1: np.zeros(2, dtype=np.intc)}
    model.ld_indptr = {1: np.array([0, 2, 4], dtype=np.int32)}
    model.ld_data = {1: np.array([1.0, 0.25, 0.25, 1.0])}
    model.var_gamma = {1: np.full((2, 1), 0.1)}
    model.var_mu = {1: np.zeros((2, 1))}
    model.var_tau = {1: np.ones((2, 1))}
    model._log_var_tau = {1: np.zeros((2, 1))}
    model.eta = {1: np.zeros(2)}
    model.q = {1: np.zeros(2)}
    model.eta_diff = {1: np.zeros(2)}
    model.dequantize_scale = 1.0
    model.threads = 1
    model.low_memory = False

    expected_tau = (
        model.n_per_snp[1] * (1.0 + model.lambda_min[:, None])
        / model.sigma_epsilon
        + model.tau_beta
    )
    expected_gamma = np.zeros(2)
    expected_mu = np.zeros(2)
    expected_eta = np.zeros(2)
    expected_q = np.zeros(2)
    expected_eta_diff = np.zeros(2)
    expected_u_logs = (
        np.log(model.pi[0])
        - np.log(1.0 - model.pi[0])
        + 0.5 * (np.log(model.tau_beta[0]) - np.log(expected_tau[:, 0]))
    )

    cpp_e_step(
        model.ld_left_bound[1],
        model.ld_indptr[1],
        model.ld_data[1],
        model.std_beta[1],
        expected_gamma,
        expected_mu,
        expected_eta,
        expected_q,
        expected_eta_diff,
        expected_u_logs,
        np.sqrt(0.5 * expected_tau[:, 0]),
        model.n_per_snp[1][:, 0] / (expected_tau[:, 0] * model.sigma_epsilon),
        1.0,
        1,
        False,
    )

    model.e_step()

    np.testing.assert_allclose(model.var_tau[1], expected_tau)
    np.testing.assert_allclose(model._log_var_tau[1], np.log(expected_tau))
    np.testing.assert_allclose(model.var_gamma[1][:, 0], expected_gamma)
    np.testing.assert_allclose(model.var_mu[1][:, 0], expected_mu)
    np.testing.assert_allclose(model.eta[1], expected_eta)
    np.testing.assert_allclose(model.q[1], expected_q)
    np.testing.assert_allclose(model.eta_diff[1], expected_eta_diff)


def test_tau_betas_initialization_accounts_for_all_snps():
    model = make_mix_model(m=10, k=2)
    model.initialize_theta(
        {
            "pis": np.array([0.1, 0.2]),
            "tau_betas": np.array([100.0, 200.0]),
        }
    )

    expected_h2 = 10 * (0.1 / 100.0 + 0.2 / 200.0)
    np.testing.assert_allclose(model.sigma_epsilon, 1.0 - expected_h2)


def test_precision_multipliers_produce_expected_component_widths():
    model = make_mix_model(m=10, k=2)
    model.d = np.array([0.25, 1.0])
    model.initialize_theta(
        {
            "pis": np.array([0.1, 0.2]),
            "sigma_epsilon": 0.8,
        }
    )

    np.testing.assert_allclose(model.tau_beta[0] / model.tau_beta[1], 0.25)
    np.testing.assert_allclose(
        (1.0 / model.tau_beta[0]) / (1.0 / model.tau_beta[1]), 4.0
    )
    np.testing.assert_allclose(
        model._prior_variance_sum(model.tau_beta), 0.2
    )


def test_elbo_matches_likelihood_prior_and_entropy_decomposition():
    model = make_mix_model(m=2, k=2)
    model._sample_size = 100
    model.sigma_epsilon = 0.7
    model.fix_params = {"sigma_epsilon": 0.7}
    model.pi = np.array([0.1, 0.2])
    model.tau_beta = np.array([4.0, 10.0])
    model.var_gamma = {1: np.array([[0.2, 0.3], [0.1, 0.4]])}
    model.var_mu = {1: np.array([[0.2, -0.1], [0.05, 0.3]])}
    model.var_tau = {1: np.array([[8.0, 12.0], [9.0, 11.0]])}
    model._log_var_tau = {1: np.log(model.var_tau[1])}
    model.eta = model.compute_eta()
    model.zeta = model.compute_zeta()
    model.std_beta = {1: np.array([0.08, -0.03])}
    model._sigma_g = 0.15

    np.testing.assert_allclose(
        model.elbo(),
        model.loglikelihood() + model.log_prior() + model.entropy(),
        rtol=1e-12,
        atol=1e-12,
    )


def test_global_variance_scaling_preserves_objectives_and_public_summaries():
    model = make_mix_model(m=2, k=2)
    model._sample_size = 100
    model._parameter_scale = 1.0
    model.sigma_epsilon = 0.7
    model.fix_params = {"sigma_epsilon": 0.7}
    model.pi = np.array([0.1, 0.2])
    model.tau_beta = np.array([4.0, 10.0])
    model.var_gamma = {1: np.array([[0.2, 0.3], [0.1, 0.4]])}
    model.var_mu = {1: np.array([[0.2, -0.1], [0.05, 0.3]])}
    model.var_tau = {1: np.array([[8.0, 12.0], [9.0, 11.0]])}
    model._log_var_tau = {1: np.log(model.var_tau[1])}
    model.eta = model.compute_eta()
    model.zeta = model.compute_zeta()
    model.q = {1: np.array([0.01, -0.02])}
    model.eta_diff = {1: np.array([0.001, -0.002])}
    model.std_beta = {1: np.array([0.08, -0.03])}
    model._sigma_g = 0.15

    summaries = (
        "elbo",
        "loglikelihood",
        "log_prior",
        "entropy",
        "mse",
        "get_average_effect_size_variance",
        "get_heritability",
    )
    expected = {name: getattr(model, name)() for name in summaries}
    expected_mu = model.var_mu[1].copy()
    expected_zeta = model.zeta[1].copy()
    unscaled_std_beta = model.std_beta

    model._set_optimization_scale(100.0)

    np.testing.assert_allclose(model.std_beta[1], 10.0 * unscaled_std_beta[1])
    np.testing.assert_allclose(model.var_mu[1], 10.0 * expected_mu)
    np.testing.assert_allclose(model.zeta[1], 100.0 * expected_zeta)
    for name, value in expected.items():
        np.testing.assert_allclose(getattr(model, name)(), value, rtol=1e-12)

    expected_sigma_epsilon = (
        1.0
        - 2.0 * model.std_beta[1].dot(model.eta[1]) / 100.0
        + model._sigma_g / 100.0
    )
    expected_tau_beta = model.d * max(
        model.var_gamma[1].sum()
        / np.dot(model.d, model.compute_zeta(sum_axis=0)[1] / 100.0),
        1.0 / np.min(model.d),
    )
    model.fix_params = {}
    model.update_sigma_epsilon()
    model.update_tau_beta()
    np.testing.assert_allclose(model.sigma_epsilon, expected_sigma_epsilon)
    np.testing.assert_allclose(
        model.tau_beta * model._parameter_scale, expected_tau_beta
    )

    model._set_optimization_scale(1.0)

    assert model.std_beta is unscaled_std_beta
    np.testing.assert_allclose(model.var_mu[1], expected_mu)
    np.testing.assert_allclose(model.zeta[1], expected_zeta)
    np.testing.assert_allclose(model.tau_beta, expected_tau_beta)


def test_nonzero_warm_start_initializes_q_from_eta():
    model = VIPRS.__new__(VIPRS)
    model.float_precision = "float64"
    model.order = "F"
    model.shapes = {1: 3}
    model.n_per_snp = {1: np.full(3, 100.0)}
    model.sigma_epsilon = 1.0
    model.tau_beta = 2.0
    model.pi = 0.2

    r = np.array(
        [
            [1.0, 0.2, -0.1],
            [0.2, 1.0, 0.3],
            [-0.1, 0.3, 1.0],
        ]
    )
    mu = np.array([0.5, -0.25, 0.1])
    gamma = np.array([0.4, 0.6, 0.2])

    expected_eta = gamma * mu
    operators = [
        LDLinearOperator(
            np.array([0, 3, 6, 9], dtype=np.int32),
            r.ravel(),
            np.zeros(3, dtype=np.int32),
            symmetric=True,
        ),
        LDLinearOperator(
            np.array([0, 2, 3, 3], dtype=np.int32),
            np.array([0.2, -0.1, 0.3]),
            np.array([1, 2, 3], dtype=np.int32),
            symmetric=False,
        ),
    ]

    for ld_lop in operators:
        model._ld_lop = {1: ld_lop}
        model.initialize_variational_parameters(
            {"mu": {1: mu}, "gamma": {1: gamma}}
        )

        np.testing.assert_allclose(model.eta[1], expected_eta)
        np.testing.assert_allclose(model.q[1], (r - np.eye(3)) @ expected_eta)


def test_fixed_parameter_aliases_and_constrained_tau_update():
    model = make_mix_model(m=2, k=2)
    model.pi = np.array([0.1, 0.2])
    model.set_fixed_params({"pi": 0.6, "tau_beta": 10.0})

    np.testing.assert_allclose(model.pi.sum(), 0.6)
    np.testing.assert_allclose(model.tau_beta, 10.0 * model.d)

    model.fix_params = {"pi": 0.6}
    model.var_gamma = {1: np.array([[0.2, 0.3], [0.1, 0.15]])}
    model.var_mu = {1: np.array([[0.5, 0.25], [0.1, 0.2]])}
    model.var_tau = {1: np.array([[4.0, 5.0], [6.0, 8.0]])}
    zetas = model.compute_zeta(sum_axis=0)[1]
    expected_scale = model.var_gamma[1].sum() / np.dot(model.d, zetas)

    model.update_tau_beta()

    expected_scale = max(expected_scale, 1.0 / np.min(model.d))
    np.testing.assert_allclose(
        model.tau_beta, model.d * expected_scale
    )

    fixed_taus = np.array([7.0, 11.0])
    model.set_fixed_params({"tau_betas": fixed_taus})
    model.update_tau_beta()
    np.testing.assert_array_equal(model.tau_beta, fixed_taus)


def test_fixed_shared_parameters_override_initial_component_values():
    model = make_mix_model(m=10, k=2)
    model.fix_params = {"pi": 0.3, "tau_beta": 20.0}

    model.initialize_theta(
        {
            "pis": np.array([0.1, 0.1]),
            "tau_betas": np.array([2.0, 4.0]),
            "sigma_epsilon": 0.8,
        }
    )

    np.testing.assert_allclose(model.pi.sum(), 0.3)
    np.testing.assert_allclose(model.tau_beta, 20.0 * model.d)


def test_dictionary_priors_use_snp_weighted_summaries():
    model = make_mix_model(m=3, k=2)
    model.shapes = {1: (2, 2), 2: (1, 2)}
    model.gdl = SimpleNamespace(m=3)
    model.initialize_theta(
        {
            "pis": {
                1: np.array([[0.1, 0.2], [0.2, 0.2]]),
                2: np.array([[0.3, 0.1]]),
            },
            "tau_betas": {
                1: np.array([[2.0, 4.0], [2.0, 4.0]]),
                2: np.array([[2.0, 4.0]]),
            },
            "sigma_epsilon": 0.8,
        }
    )
    model.n_per_snp = {1: np.full((2, 1), 100.0), 2: np.full((1, 1), 100.0)}
    model._ld_lop = {}
    model.initialize_variational_parameters()

    np.testing.assert_allclose(model.get_proportion_causal(), (0.3 + 0.4 + 0.4) / 3)
    np.testing.assert_allclose(model.get_null_pi(1), np.array([0.7, 0.6]))
    expected_variance = (
        0.1 / 2.0
        + 0.2 / 4.0
        + 0.2 / 2.0
        + 0.2 / 4.0
        + 0.3 / 2.0
        + 0.1 / 4.0
    ) / 3
    np.testing.assert_allclose(
        model.get_average_effect_size_variance(), expected_variance
    )
    assert model.var_tau[1].shape == (2, 2)
    assert model.var_tau[2].shape == (1, 2)


def test_compute_zeta_uses_double_precision():
    model = make_mix_model(m=1, k=2)
    model.var_gamma = {1: np.array([[0.2, 0.3]], dtype=np.float32)}
    model.var_mu = {1: np.array([[1e-4, 2e-4]], dtype=np.float32)}
    model.var_tau = {1: np.array([[1e8, 2e8]], dtype=np.float32)}

    assert model.compute_zeta()[1].dtype == np.float64

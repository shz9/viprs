import numpy as np
import pandas as pd
from magenpy.stats.h2.ldsc import simple_ldsc

from ..utils.compute_utils import dict_sum
from .vi.e_step_cpp import cpp_e_step_mixture
from .VIPRS import VIPRS


class VIPRSMix(VIPRS):
    """
    A class for the Variational Inference for Polygenic Risk Scores (VIPRS) model
    parametrized with the sparse mixture prior on the effect sizes. The class inherits
    many of the methods and attributes from the `VIPRS` class unchanged. However,
    there are many important updates and changes to the model, including the dimensionality
    of the arrays representing the variational parameters.

    Details for the algorithm can be found in the Supplementary Material of the following paper:

    > Zabad S, Gravel S, Li Y. Fast and accurate Bayesian polygenic risk modeling with variational inference.
    Am J Hum Genet. 2023 May 4;110(5):741-761. doi: 10.1016/j.ajhg.2023.03.009.
    Epub 2023 Apr 7. PMID: 37030289; PMCID: PMC10183379.

    :ivar K: The number of causal (i.e. non-null) components in the mixture prior (minimum 1). When `K=1`, this
    effectively reduces `VIPRSMix` to the `VIPRS` model.
    :ivar d: Multiplier for the prior precision of each non-null component
        (vector of size K). Smaller values correspond to wider components.

    """

    def __init__(self, gdl, K=1, prior_multipliers=None, **kwargs):
        """
        :param gdl: An instance of `GWADataLoader`
        :param K: The number of causal (i.e. non-null) components in the mixture prior (minimum 1). When `K=1`, this
            effectively reduces `VIPRSMix` to the `VIPRS` model.
        :param prior_multipliers: Multipliers for the prior precisions of the
            non-null mixture components (vector of size K). Smaller values
            correspond to wider components.
        :param kwargs: Additional keyword arguments to pass to the VIPRS model.
        """

        # Make sure that the matrices follow the C-contiguous order:
        kwargs["order"] = "C"

        super().__init__(gdl, **kwargs)

        # Sanity checks:
        assert K > 0  # Check that there is at least 1 causal component
        self.K = K

        if prior_multipliers is not None:
            assert len(prior_multipliers) == K
            self.d = np.array(prior_multipliers).astype(self.float_precision)
        else:
            self.d = 2 ** np.linspace(-min(K - 1, 7), 0, K).astype(self.float_precision)

        if np.any(self.d <= 0.0):
            raise ValueError("Prior precision multipliers must be strictly positive.")

        # Populate/update relevant fields:
        self.shapes = {c: (shp, self.K) for c, shp in self.shapes.items()}
        self.n_per_snp = {
            c: n[:, None].astype(self.float_precision, order=self.order)
            for c, n in self.n_per_snp.items()
        }

    def initialize_theta(self, theta_0=None):
        """
        Initialize the global hyperparameters of the model
        :param theta_0: A dictionary of initial values for the hyperparameters theta
        """

        theta_0 = {} if theta_0 is None else theta_0.copy()
        theta_0.update(self.fix_params)
        if "tau_beta" in self.fix_params and "tau_betas" not in self.fix_params:
            theta_0.pop("tau_betas", None)

        # ----------------------------------------------
        # (1) Initialize pi from a uniform
        if "pis" in theta_0:
            self.pi = self._cast_parameter(theta_0["pis"])

            # If only the total causal proportion is fixed, preserve the
            # supplied component ratios and rescale them to that total.
            if "pi" in self.fix_params and "pis" not in self.fix_params:
                overall_pi = self.fix_params["pi"]
                if isinstance(self.pi, dict):
                    self.pi = {
                        c: overall_pi * pi / pi.sum(axis=1, keepdims=True)
                        for c, pi in self.pi.items()
                    }
                else:
                    self.pi *= overall_pi / self.pi.sum()
        else:
            overall_pi = theta_0["pi"] if "pi" in theta_0 else np.random.uniform(
                low=max(0.005, 1.0 / self.n_snps), high=0.1
            )

            self.pi = overall_pi * np.random.dirichlet(np.ones(self.K))

        # ----------------------------------------------
        # (2) Initialize sigma_epsilon and component precisions. Under the
        # standardized model, h2 = sum_jk(pi_jk / tau_jk) = 1 - sigma_epsilon.
        if "tau_betas" in theta_0:
            self.tau_beta = self._cast_parameter(theta_0["tau_betas"])
        elif "tau_beta" in theta_0:
            self.tau_beta = theta_0["tau_beta"] * self.d
        else:
            if "sigma_epsilon" in theta_0:
                h2g_estimate = 1.0 - theta_0["sigma_epsilon"]
            else:
                try:
                    h2g_estimate = np.clip(simple_ldsc(self.gdl), 1e-3, 1.0 - 1e-3)
                except Exception:
                    h2g_estimate = np.random.uniform(low=0.001, high=0.999)

            self.tau_beta = self.d * (
                self._prior_variance_sum(self.d) / h2g_estimate
            )

        if "sigma_epsilon" in theta_0:
            self.sigma_epsilon = theta_0["sigma_epsilon"]
        else:
            self.sigma_epsilon = np.clip(
                1.0 - self._prior_variance_sum(self.tau_beta),
                a_min=1e-4,
                a_max=1.0 - 1e-4,
            )

        # Cast all the hyperparameters to conform to the precision set by the user:
        self.sigma_epsilon = np.dtype(self.float_precision).type(self.sigma_epsilon)
        self.pi = self._cast_parameter(self.pi)
        self.tau_beta = self._cast_parameter(self.tau_beta)
        self.lambda_min = self._cast_parameter(self.lambda_min)
        self._sigma_g = np.dtype(self.float_precision).type(0.0)

    def _cast_parameter(self, value):
        """Cast scalar, array, or chromosome-indexed parameters."""

        if isinstance(value, dict):
            return {
                c: np.asarray(v, dtype=self.float_precision, order=self.order)
                for c, v in value.items()
            }
        if np.isscalar(value):
            return np.dtype(self.float_precision).type(value)
        return np.asarray(value, dtype=self.float_precision, order=self.order)

    def _prior_variance_sum(self, tau_beta):
        """Return the total prior variance contributed by all variants."""

        total = 0.0
        for c, shape in self.shapes.items():
            c_pi = self.pi[c] if isinstance(self.pi, dict) else self.pi
            c_tau = tau_beta[c] if isinstance(tau_beta, dict) else tau_beta
            if np.ndim(c_pi) == 1 and np.ndim(c_tau) == 1:
                total += shape[0] * np.sum(c_pi / c_tau, dtype=np.float64)
            else:
                total += np.sum(np.asarray(c_pi) / np.asarray(c_tau), dtype=np.float64)
        return total

    def e_step(self):
        """
        Run the E-Step of the Variational EM algorithm.
        Here, we update the variational parameters for each variant using coordinate
        ascent optimization techniques. The update equations are outlined in
        the Supplementary Material of the following paper:

        > Zabad S, Gravel S, Li Y. Fast and accurate Bayesian polygenic risk modeling with variational inference.
        Am J Hum Genet. 2023 May 4;110(5):741-761. doi: 10.1016/j.ajhg.2023.03.009.
        Epub 2023 Apr 7. PMID: 37030289; PMCID: PMC10183379.
        """

        for c, shapes in self.shapes.items():
            # Get the priors:
            tau_beta = self.get_tau_beta(c)
            pi = self.get_pi(c)

            lambda_min = self.lambda_min[c] if isinstance(self.lambda_min, dict) else self.lambda_min
            if not np.isscalar(lambda_min):
                lambda_min = np.asarray(lambda_min)[:, None]

            # Updates for tau variational parameters:
            self.var_tau[c] = (
                self.n_per_snp[c] * (1.0 + lambda_min) / self.sigma_epsilon
            ) + tau_beta
            np.log(self.var_tau[c], out=self._log_var_tau[c])

            if isinstance(self.pi, dict):
                log_null_pi = np.log(1.0 - self.pi[c].sum(axis=1))
            else:
                log_null_pi = np.ones_like(self.eta[c]) * np.log(1.0 - self.pi.sum())

            # Compute some quantities that are needed for the per-SNP updates:
            mu_mult = (
                self.n_per_snp[c] / (self.var_tau[c] * self.sigma_epsilon)
            ).astype(self.float_precision)
            u_logs = (
                np.log(pi)
                + 0.5 * (np.log(tau_beta) - self._log_var_tau[c])
            ).astype(self.float_precision)

            cpp_e_step_mixture(
                self.ld_left_bound[c],
                self.ld_indptr[c],
                self.ld_data[c],
                self.std_beta[c],
                self.var_gamma[c],
                self.var_mu[c],
                self.eta[c],
                self.q[c],
                self.eta_diff[c],
                log_null_pi,
                u_logs,
                np.sqrt(0.5 * self.var_tau[c]).astype(self.float_precision),
                mu_mult,
                self.dequantize_scale,
                self.threads,
                self.low_memory,
            )

        self.zeta = self.compute_zeta()

    def set_fixed_params(self, fix_params):
        """Set fixed mixture hyperparameters, including their plural aliases."""

        assert isinstance(fix_params, dict), "The fixed parameters must be provided as a dictionary."

        # The plural forms fix component-specific values; the singular forms
        # fix only the total probability or shared precision scale.
        if "pis" in fix_params:
            self.fix_params.pop("pi", None)
        elif "pi" in fix_params:
            self.fix_params.pop("pis", None)
        if "tau_betas" in fix_params:
            self.fix_params.pop("tau_beta", None)
        elif "tau_beta" in fix_params:
            self.fix_params.pop("tau_betas", None)

        self.fix_params.update(fix_params)

        if "sigma_epsilon" in fix_params:
            self.sigma_epsilon = np.dtype(self.float_precision).type(
                fix_params["sigma_epsilon"]
            )
        if "lambda_min" in fix_params:
            self.lambda_min = self._cast_parameter(fix_params["lambda_min"])
        if "tau_betas" in fix_params:
            self.tau_beta = self._cast_parameter(fix_params["tau_betas"])
        elif "tau_beta" in fix_params:
            self.tau_beta = fix_params["tau_beta"] * self.d
        if "pis" in fix_params:
            self.pi = self._cast_parameter(fix_params["pis"])
        elif "pi" in fix_params and self.pi is not None:
            overall_pi = fix_params["pi"]
            if isinstance(self.pi, dict):
                self.pi = {
                    c: overall_pi * pi / pi.sum(axis=1, keepdims=True)
                    for c, pi in self.pi.items()
                }
            else:
                self.pi *= overall_pi / self.pi.sum()

    def update_pi(self):
        """
        Update the prior mixing proportions `pi`
        """

        if "pis" not in self.fix_params:
            pi_estimate = dict_sum(self.var_gamma, axis=0)

            if "pi" in self.fix_params:
                # If the user provides an estimate for the total proportion of causal variants,
                # update the pis such that the proportion of SNPs in the null component becomes 1. - pi.
                pi_estimate = self.fix_params["pi"] * pi_estimate / pi_estimate.sum()
            else:
                pi_estimate /= self.n_snps

            # Set pi to the new estimate:
            self.pi = pi_estimate

    def update_tau_beta(self):
        """
        Update the prior precision (inverse variance) for the effect sizes, `tau_beta`
        """

        if "tau_betas" not in self.fix_params and "tau_beta" not in self.fix_params:
            # Estimate a shared precision scale and apply the component-specific
            # precision multipliers. This is the constrained mixture M-step.

            zetas = sum(self.compute_zeta(sum_axis=0).values())
            total_responsibility = dict_sum(self.var_gamma)

            global_tau = total_responsibility / np.dot(self.d, zetas)
            # Preserve the exact component ratios while enforcing tau_k >= 1.
            global_tau = np.maximum(global_tau, 1.0 / np.min(self.d))

            self.tau_beta = self.d * global_tau

    def get_null_pi(self, chrom=None):
        """
        Get the proportion of SNPs in the null component
        :param chrom: If provided, get the mixing proportion for the null component on a given chromosome.
        :return: The value of the mixing proportion for the null component
        """

        pi = self.get_pi(chrom=chrom)

        if isinstance(pi, dict):
            return {c: 1.0 - c_pi.sum(axis=-1) for c, c_pi in pi.items()}
        return 1.0 - np.sum(pi, axis=-1)

    def get_proportion_causal(self):
        """
        :return: The proportion of variants in the non-null components.
        """
        if isinstance(self.pi, dict):
            return np.dtype(self.float_precision).type(
                sum(np.sum(pis, dtype=np.float64) for pis in self.pi.values())
                / self.n_snps
            )
        return np.sum(self.pi)

    def get_average_effect_size_variance(self):
        """
        :return: The average per-SNP variance for the prior mixture components
        """

        total = self._prior_variance_sum(self.tau_beta)
        return total / self.n_snps

    def compute_pip(self):
        """
        :return: The posterior inclusion probability
        """
        return {c: gamma.sum(axis=1) for c, gamma in self.var_gamma.items()}

    def compute_eta(self):
        """
        :return: The mean for the effect size under the variational posterior.
        """
        return {c: (v * self.var_mu[c]).sum(axis=1) for c, v in self.var_gamma.items()}

    def compute_zeta(self, sum_axis=1):
        """
        :return: The expectation of the squared effect size under the variational posterior.
        """
        return {
            c: np.sum(
                v.astype(np.float64)
                * (
                    self.var_mu[c].astype(np.float64) ** 2
                    + (1.0 / self.var_tau[c].astype(np.float64))
                ),
                axis=sum_axis,
            )
            for c, v in self.var_gamma.items()
        }

    def to_theta_table(self):
        """
        :return: A `pandas` DataFrame containing information about the estimated hyperparameters of the model.
        """

        table = super().to_theta_table()

        extra_theta = []

        if isinstance(self.tau_beta, dict):
            average_taus = dict_sum(self.tau_beta, axis=0) / self.n_snps
            for i, tau in enumerate(np.atleast_1d(average_taus), 1):
                table.loc[table["Parameter"] == f"tau_beta_{i}", "Value"] = tau

        if isinstance(self.pi, dict):
            pis = dict_sum(self.pi, axis=0) / self.n_snps
        else:
            pis = self.pi

        for i in range(self.K):
            extra_theta.append({"Parameter": f"pi_{i + 1}", "Value": pis[i]})

        return pd.concat([table, pd.DataFrame(extra_theta)])

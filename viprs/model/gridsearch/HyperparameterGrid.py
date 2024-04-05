import numpy as np
import pandas as pd
import itertools


class HyperparameterGrid(object):
    """
    A utility class to facilitate generating grids for the
    hyperparameters of the standard `VIPRS` models. It is designed to
    interface with models that operate on grids of hyperparameters,
    such as `VIPRSGridSeach` and `VIPRSBMA`. The hyperparameters for
    the standard VIPRS model are:

    * `sigma_epsilon`: The residual variance for the phenotype.
    * `tau_beta`: The precision (inverse variance) of the prior for the effect sizes.
    * `pi`: The proportion of non-zero effect sizes (polygenicity).

    :ivar sigma_epsilon: A grid of values for the residual variance hyperparameter.
    :ivar tau_beta: A grid of values for the precision of the prior for the effect sizes.
    :ivar pi: A grid of values for the proportion of non-zero effect sizes.
    :ivar h2_est: An estimate of the heritability for the trait under consideration.
    :ivar h2_se: The standard error of the heritability estimate.
    :ivar n_snps: The number of common variants that may be relevant for this analysis.

    """

    def __init__(self,
                 sigma_epsilon_grid=None,
                 sigma_epsilon_steps=None,
                 tau_beta_grid=None,
                 tau_beta_steps=None,
                 pi_grid=None,
                 pi_steps=None,
                 h2_est=None,
                 h2_se=None,
                 n_snps=1e6):
        """

        Create a hyperparameter grid for the standard VIPRS model with the
        spike-and-slab prior. The hyperparameters for this model are:

        * `sigma_epsilon`: The residual variance
        * `tau_beta`: The precision (inverse variance) of the prior for the effect sizes
        * `pi`: The proportion of non-zero effect sizes

        For each of these hyperparameters, we can provide a grid of values to search over.
        If the heritability estimate and standard error (from e.g. LDSC) are provided,
        we can generate grids for sigma_epsilon and tau_beta that are informed by these estimates.

        For each hyperparameter to be included in the grid, user must specify either the grid
        itself, or the number of steps to use to generate the grid.

        :param sigma_epsilon_grid: An array containing a grid of values for the sigma_epsilon hyperparameter.
        :param sigma_epsilon_steps: The number of steps for the sigma_epsilon grid
        :param tau_beta_grid: An array containing a grid of values for the tau_beta hyperparameter.
        :param tau_beta_steps: The number of steps for the tau_beta grid
        :param pi_grid: An array containing a grid of values for the pi hyperparameter
        :param pi_steps: The number of steps for the pi grid
        :param h2_est: An estimate of the heritability for the trait under consideration. If provided,
        we can generate grids for some of the hyperparameters that are consistent with this estimate.
        :param h2_se: The standard error of the heritability estimate. If provided, we can generate grids
        for some of the hyperparameters that are consistent with this estimate.
        :param n_snps: Number of common variants that may be relevant for this analysis. This estimate can
        be used to generate grids that are based on this number.
        """

        # If the heritability estimate is not provided, use a reasonable default value of 0.1
        # with a wide standard error of 0.1.
        if h2_est is None:
            self.h2_est = 0.1
            self.h2_se = 0.1
        else:
            self.h2_est = h2_est
            self.h2_se = h2_se

        self.n_snps = n_snps
        self._search_params = []

        # Initialize the grid for sigma_epsilon:
        self.sigma_epsilon = sigma_epsilon_grid
        if self.sigma_epsilon is not None:
            self._search_params.append('sigma_epsilon')
        elif sigma_epsilon_steps is not None:
            self.generate_sigma_epsilon_grid(steps=sigma_epsilon_steps)

        # Initialize the grid for the tau_beta:
        self.tau_beta = tau_beta_grid
        if self.tau_beta is not None:
            self._search_params.append('tau_beta')
        elif tau_beta_steps is not None:
            self.generate_tau_beta_grid(steps=tau_beta_steps)

        # Initialize the grid for pi:
        self.pi = pi_grid
        if self.pi is not None:
            self._search_params.append('pi')
        elif pi_steps is not None:
            self.generate_pi_grid(steps=pi_steps)

    def _generate_h2_grid(self, steps=5):
        """
        Use the heritability estimate and standard error to generate a grid of values for
        the heritability parameter. Specifically, given the estimate and standard error, we
        generate heritability estimates from the percentiles of the normal distribution,
        with mean `h2_est` and standard deviation `h2_se`. The grid values range from the 10th
        percentile to the 90th percentile of this normal distribution.

        :param steps: The number of steps for the heritability grid.
        :return: A grid of values for the heritability parameter.

        """

        assert steps > 0
        assert self.h2_est is not None

        # If the heritability standard error is not provided, we use half of the heritability estimate
        # by default.
        # *Justification*: Under the assumption that heritability for the trait being analyzed
        # is significantly greater than 0, the standard error should be, at a maximum,
        # half of the heritability estimate itself to get us a Z-score with absolute value
        # greater than 2.
        if self.h2_se is None:
            h2_se = self.h2_est * 0.5
        else:
            h2_se = self.h2_se

        # Sanity checking steps:
        assert 0. < self.h2_est < 1.
        assert h2_se > 0

        from scipy.stats import norm

        # First, determine the percentile boundaries to avoid producing
        # invalid values for the heritability grid:

        percentile_start = max(0.1, norm.cdf(1e-5, loc=self.h2_est, scale=h2_se))
        percentile_stop = min(0.9, norm.cdf(1. - 1e-5, loc=self.h2_est, scale=h2_se))

        # Generate the heritability grid:
        return norm.ppf(np.linspace(percentile_start, percentile_stop, steps),
                        loc=self.h2_est, scale=h2_se)

    def generate_sigma_epsilon_grid(self, steps=5):
        """
        Generate a grid of values for the `sigma_epsilon` (residual variance) hyperparameter.

        :param steps: The number of steps for the sigma_epsilon grid.
        """

        assert steps > 0

        h2_grid = self._generate_h2_grid(steps)
        self.sigma_epsilon = 1. - h2_grid

        if 'sigma_epsilon' not in self._search_params:
            self._search_params.append('sigma_epsilon')

    def generate_tau_beta_grid(self, steps=5):
        """
        Generate a grid of values for the `tau_beta`
        (precision of the prior for the effect sizes) hyperparameter.
        :param steps: The number of steps for the `tau_beta` grid
        """

        assert steps > 0

        h2_grid = self._generate_h2_grid(steps)
        # Assume ~1% of SNPs are causal:
        self.tau_beta = 0.01*self.n_snps / h2_grid

        if 'tau_beta' not in self._search_params:
            self._search_params.append('tau_beta')

    def generate_pi_grid(self, steps=5):
        """
        Generate a grid of values for the `pi` (proportion of non-zero effect sizes) hyperparameter.
        :param steps: The number of steps for the `pi` grid
        """

        assert steps > 0

        self.pi = np.unique(np.clip(10. ** (-np.linspace(np.floor(np.log10(self.n_snps)), 0., steps)),
                                    a_min=1. / self.n_snps,
                                    a_max=1. - 1. / self.n_snps))

        if 'pi' not in self._search_params:
            self._search_params.append('pi')

    def combine_grids(self):
        """
        Weave together the different hyperparameter grids and return a list of
        dictionaries where the key is the hyperparameter name and the value is
        value for that hyperparameter.

        :return: A list of dictionaries containing the hyperparameter values.
        :raises ValueError: If all the grids are empty.

        """
        hyp_names = [name for name, value in self.__dict__.items()
                     if value is not None and name in self._search_params]

        if len(hyp_names) > 0:
            hyp_values = itertools.product(*[hyp_grid for hyp_name, hyp_grid in self.__dict__.items()
                                             if hyp_grid is not None and hyp_name in hyp_names])

            return [dict(zip(hyp_names, hyp_v)) for hyp_v in hyp_values]
        else:
            raise ValueError("All the grids are empty!")

    def to_table(self):
        """
        :return: The hyperparameter grid as a pandas `DataFrame`.
        """

        combined_grids = self.combine_grids()
        if combined_grids:
            return pd.DataFrame(combined_grids)
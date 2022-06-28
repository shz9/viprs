import numpy as np
import pandas as pd
import itertools


class VIPRSGrid(object):
    """
    A utility class to generate grids for the hyperparameters of VIPRS models.
    """

    def __init__(self, sigma_epsilon=None, sigma_beta=None, pi=None,
                 search_params=('pi', 'sigma_epsilon', 'sigma_beta')):
        """
        Create a hyperparameter grid for the VIPRS model
        :param sigma_epsilon: A grid of values for the sigma_epsilon hyperparameter
        :param sigma_beta: A grid of values for the sigma_beta hyperparameter
        :param pi: A grid of values for the pi hyperparameter
        :param search_params: The hyperparameters to perform the search over.
        """

        self.sigma_epsilon = sigma_epsilon
        self.sigma_beta = sigma_beta
        self.pi = pi

        self.search_params = search_params

    def update_search_parameters(self, search_params):
        self.search_params = search_params

    def generate_sigma_epsilon_grid(self, steps=5, min_val=1e-6, max_val=1.-1e-6,
                                    h2=None, base=1.25):
        """
        :param steps: The number of steps for the sigma_epsilon grid.
        :param min_val: The minimum value for the sigma_epsilon grid.
        :param max_val: The maximum value for the sigma_epsilon grid.
        :param h2: An estimate of the heritability that can be used to
        guide the grid creation for sigma epsilon.
        :param base: The base to use for the multipliers around the provided
        heritability value.
        """
        if steps > 0:
            if h2 is None:
                self.sigma_epsilon = np.linspace(min_val, max_val, steps)
            else:
                h2_grid = (base ** (np.arange(-np.floor(steps / 2), np.ceil(steps / 2)))) * h2

                self.sigma_epsilon = np.unique(np.clip(1. - h2_grid,
                                                       a_min=min_val,
                                                       a_max=max_val))

    def generate_sigma_beta_grid(self, steps=5, n_snps=1e6,
                                 h2=None, base=1.25):
        """
        :param steps: The number of steps for the sigma_beta grid
        :param n_snps: The number of variants included in the model
        :param h2: An estimate of the heritability that can be used to
        guide the grid creation for sigma epsilon.
        :param base: The base to use for the multipliers around the provided
        heritability value.
        """
        if steps > 0:

            if h2 is None:
                self.sigma_beta = (1./n_snps)*np.linspace(1. / steps, 1., steps)
            else:
                h2_grid = np.unique(np.clip((base ** (np.arange(-np.floor(steps / 2), np.ceil(steps / 2)))) * h2,
                                            a_min=1e-4, a_max=1.-1e-4))
                self.sigma_beta = h2_grid/n_snps

    def generate_pi_grid(self, steps=5, n_snps=1e6):
        """
        :param steps: The number of steps for the pi grid
        :param n_snps: The number of variants included in the model
        """
        if steps > 0:
            self.pi = np.clip(10. ** (-np.linspace(np.floor(np.log10(n_snps)), 0., steps)),
                              a_min=1. / n_snps, a_max=1. - 1. / n_snps)

    def combine_grids(self):
        """
        Weave together the different hyperparameter grids and return a list of
        dictionaries where the key is the hyperparameter name and the value is
        value for that hyperparameter.

        NOTE: This function assumes that all instances attributes are hyperparameters
        and uses their names as keys.
        """
        hyp_names = [name for name, value in self.__dict__.items()
                     if value is not None and name in self.search_params]

        if len(hyp_names) > 0:
            hyp_values = itertools.product(*[hyp_grid for hyp_name, hyp_grid in self.__dict__.items()
                                             if hyp_grid is not None and hyp_name in hyp_names])

            return [dict(zip(hyp_names, hyp_v)) for hyp_v in hyp_values]
        else:
            raise ValueError("All the grids are empty!")

    def to_table(self):

        combined_grids = self.combine_grids()
        if combined_grids:
            return pd.DataFrame(combined_grids)


class VIPRSAlphaGrid(VIPRSGrid):

    def __init__(self, sigma_epsilon=None, sigma_beta=None, pi=None, alpha=None,
                 search_params=('pi', 'sigma_epsilon', 'sigma_beta', 'alpha')):

        super().__init__(sigma_epsilon=sigma_epsilon, sigma_beta=sigma_beta,
                         pi=pi, search_params=search_params)
        self.alpha = alpha

    def generate_alpha_grid(self, steps=5):
        """
        :param steps: The number of steps for the pi grid
        """
        self.alpha = np.linspace(-1., 0., steps)

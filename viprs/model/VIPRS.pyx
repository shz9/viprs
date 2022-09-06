# cython: linetrace=False
# cython: profile=False
# cython: binding=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: language_level=3
# cython: infer_types=True

import numpy as np
cimport numpy as np
import pandas as pd
import warnings
from tqdm import tqdm

from .PRSModel cimport PRSModel
from magenpy.stats.h2.ldsc import simple_ldsc
from viprs.utils.exceptions import OptimizationDivergence
from viprs.utils.math_utils cimport elementwise_add_mult, sigmoid, clip
from viprs.utils.compute_utils import dict_mean, dict_sum, dict_concat, fits_in_memory


cdef class VIPRS(PRSModel):
    """
    The base class for performing Variational Inference of Polygenic Risk Scores (VIPRS).
    """

    def __init__(self, gdl, fix_params=None, load_ld='auto', tracked_theta=None, verbose=True, threads=1):
        """
        :param gdl: An instance of GWADataLoader
        :param fix_params: A dictionary of hyperparameters with their fixed values.
        :param load_ld: A flag that specifies whether to load the LD matrix to memory (Default: `auto`).
        :param tracked_theta: A list of hyperparameters to track throughout the optimization procedure. Useful
        for debugging/model checking. Currently, we allow the user to track the following:
            - The proportion of causal variants (`pi`).
            - The heritability ('heritability').
            - The residual variance (`sigma_epsilon`).
        :param verbose: Verbosity of the information printed to standard output
        :param threads: The number of threads to use (experimental)
        """

        super().__init__(gdl)

        # Variational parameters:
        self.var_gamma = {}
        self.var_mu = {}
        self.var_sigma = {}

        # Properties of proposed distribution:
        self.eta = {}  # The posterior mean, E[B] = \gamma*\mu_beta
        self.zeta = {}  # The expectation of B^2 under the posterior, E[B^2] = \gamma*(\mu_beta^2 + \sigma_beta^2)

        # q-factor (keeps track of LD-related terms)
        self.q = {}

        # ---------- Inputs to the model: ----------

        # LD-related quantities:
        self.ld = self.gdl.get_ld_matrices()
        self.ld_bounds = self.gdl.get_ld_boundaries()

        # If load_ld is set to `auto`, then determine whether to load
        # the LD matrices by examining the available memory resources:
        if load_ld == 'auto':
            self.load_ld = fits_in_memory(sum([ld.estimate_uncompressed_size() for ld in self.ld.values()]))
        else:
            self.load_ld = load_ld

        # Standardized betas:
        self.std_beta = {c: ss.get_snp_pseudo_corr() for c, ss in self.gdl.sumstats_table.items()}

        # ---------- General properties: ----------

        self.inv_temperature = 1.
        self.threads = threads
        self.fix_params = fix_params or {}

        self.verbose = verbose
        self.history = {}
        self.tracked_theta = tracked_theta or []

    cpdef initialize(self, theta_0=None, param_0=None):
        """
        A convenience method to initialize all the objects associated with the model.
        :param theta_0: A dictionary of initial values for the hyperparameters theta
        :param param_0: A dictionary of initial values for the variational parameters
        """

        if self.verbose:
            print("> Initializing model parameters")

        self.initialize_theta(theta_0)
        self.initialize_variational_parameters(param_0)
        self.init_history()

    cpdef init_history(self):
        """
        Initialize the history object to track various quantities.
        """

        self.history = {
            'ELBO': []
        }

        for tt in self.tracked_theta:
            self.history[tt] = []

    cpdef initialize_theta(self, theta_0=None):
        """
        Initialize the global hyper-parameters
        :param theta_0: A dictionary of initial values for the hyperparameters theta
        """

        if theta_0 is not None and self.fix_params is not None:
            theta_0.update(self.fix_params)
        elif self.fix_params is not None:
            theta_0 = self.fix_params
        elif theta_0 is None:
            theta_0 = {}

        # ----------------------------------------------
        # (1) If 'pi' is not set, initialize from a uniform
        if 'pi' not in theta_0:
            self.pi = np.random.uniform(low=max(0.005, 1. / self.n_snps), high=.1)
        else:
            self.pi = theta_0['pi']

        # ----------------------------------------------
        # (2) Initialize sigma_epsilon and sigma_beta
        # Assuming that the genotype and phenotype are normalized,
        # these two quantities are conceptually linked.
        # The initialization routine here assumes that:
        # Var(y) = h2 + sigma_epsilon
        # Where, by assumption, Var(y) = 1,
        # And h2 ~= pi*M*sigma_beta

        if 'sigma_epsilon' not in theta_0:
            if 'sigma_beta' not in theta_0:

                # If neither sigma_beta nor sigma_epsilon are given,
                # then initialize using the SNP heritability estimate

                try:
                    naive_h2g = clip(simple_ldsc(self.gdl), 1e-3, 1. - 1e-3)
                except Exception as e:
                    naive_h2g = np.random.uniform(low=.001, high=.999)

                self.sigma_epsilon = 1. - naive_h2g
                self.sigma_beta = naive_h2g / (self.pi * self.n_snps)
            else:

                # If sigma_beta is given, use it to initialize sigma_epsilon

                self.sigma_beta = theta_0['sigma_beta']
                self.sigma_epsilon = np.clip(1. - self.sigma_beta*(self.pi * self.n_snps),
                                             a_min=1e-12, a_max=1. - 1e-12)
        else:

            # If sigma_epsilon is given, use it in the initialization

            self.sigma_epsilon = theta_0['sigma_epsilon']

            if 'sigma_beta' in theta_0:
                self.sigma_beta = theta_0['sigma_beta']
            else:
                self.sigma_beta = (1. - self.sigma_epsilon) / (self.pi * self.n_snps)

    cpdef initialize_variational_parameters(self, param_0=None):
        """
        Initialize the variational parameters.
        :param param_0: A dictionary of initial values for the variational parameters
        """

        param_0 = param_0 or {}

        self.var_mu = {}
        self.var_sigma = {}
        self.var_gamma = {}

        for c, shapes in self.shapes.items():

            # Initialize the variational parameters according to the derived update equations,
            # ignoring correlations between SNPs.
            if 'sigma' in param_0:
                self.var_sigma[c] = param_0['sigma'][c]
            else:
                self.var_sigma[c] = self.sigma_epsilon / (
                        self.Nj[c] + self.sigma_epsilon / self.get_sigma_beta(c)
                )

            if 'mu' in param_0:
                self.var_mu[c] = param_0['mu'][c]
            else:
                self.var_mu[c] = np.zeros(shapes)

            if 'gamma' in param_0:
                self.var_gamma[c] = param_0['gamma'][c]
            else:
                pi = self.get_pi(c)
                if isinstance(self.pi, dict):
                    self.var_gamma[c] = pi.copy()
                else:
                    self.var_gamma[c] = pi*np.ones(shapes)

        self.eta = self.compute_eta()
        self.zeta = self.compute_zeta()
        self.q = {c: np.zeros_like(eta) for c, eta in self.eta.items()}

    cpdef e_step(self):
        """
        In the E-step, update the variational parameters for each SNP 
        in a coordinate-wise fashion.
        """

        # Define memoryviews objects for fast access
        cdef:
            unsigned int j, start, end
            double u_j, eta_diff
            double[::1] std_beta, Dj  # Inputs
            double[::1] var_gamma, var_mu, var_sigma  # Variational parameters
            double[::1] mu_mult, u_logs, recip_sigma  # Helpers + other quantities that we need inside the for loop
            double[::1] eta, q  # Properties of proposed distribution
            long[:, ::1] ld_bound

        for c, c_size in self.shapes.items():

            # Get the priors:
            sigma_beta = self.get_sigma_beta(c)
            pi = self.get_pi(c)

            # Updates for sigma_beta variational parameters:
            self.var_sigma[c] = self.sigma_epsilon / (self.inv_temperature*(
                    self.Nj[c] + self.sigma_epsilon / sigma_beta
            ))

            # Compute some quantities that are needed for the per-SNP updates:
            mu_mult = self.inv_temperature*self.Nj[c]*self.var_sigma[c]/self.sigma_epsilon
            u_logs = np.log(pi / (1. - pi)) + .5*np.log(self.var_sigma[c] / sigma_beta)
            recip_sigma = .5/self.var_sigma[c]

            # Set the numpy vectors into memoryviews for fast access:
            std_beta = self.std_beta[c]
            var_gamma = self.var_gamma[c]
            var_mu = self.var_mu[c]
            eta = self.eta[c]
            ld_bound = self.ld_bounds[c]
            q = self.q[c]

            for j, Dj in enumerate(self.ld[c]):

                start, end = ld_bound[:, j]

                # Compute the variational mu beta:
                var_mu[j] = mu_mult[j]*(std_beta[j] - q[j])

                # Compute the variational gamma:
                u_j = self.inv_temperature*(u_logs[j] + recip_sigma[j]*var_mu[j]*var_mu[j])
                var_gamma[j] = clip(sigmoid(u_j), 1e-8, 1. - 1e-8)

                # Compute the difference between the new and old values for the posterior mean:
                eta_diff = var_gamma[j]*var_mu[j] - eta[j]

                # Update the q factors for all neighboring SNPs that are in LD with SNP j
                elementwise_add_mult(q[start: end], Dj, eta_diff)
                # Operation above updates the q factor for SNP j, so we correct that here:
                q[j] = q[j] - eta_diff

                # Update the posterior mean:
                eta[j] = eta[j] + eta_diff

            # Convert memoryviews back to numpy arrays:
            self.var_gamma[c] = np.asarray(var_gamma)
            self.var_mu[c] = np.asarray(var_mu)
            self.q[c] = np.asarray(q)
            self.eta[c] = np.asarray(eta)
            self.zeta[c] = self.var_gamma[c]*(self.var_mu[c]**2 + self.var_sigma[c])

    cpdef update_pi(self):
        """
        Update the prior probability of a variant being causal, pi
        """

        if 'pi' not in self.fix_params:

            # Get the average of the gammas:
            self.pi = dict_mean(self.var_gamma, axis=0)

    cpdef update_sigma_beta(self):
        """
        Update the prior variance on the effect size, sigma_beta
        """

        if 'sigma_beta' not in self.fix_params:

            # Sigma_beta estimate:
            sigma_beta_estimate = dict_sum(self.zeta, axis=0) / dict_sum(self.var_gamma, axis=0)
            # Clip value:
            self.sigma_beta = np.clip(sigma_beta_estimate, 1e-12, 1. - 1e-12)

    cpdef update_sigma_epsilon(self):
        """
        Update the residual variance, sigma_epsilon.
        """

        if 'sigma_epsilon' not in self.fix_params:

            sig_eps = 0.

            for c, _ in self.shapes.items():

                sig_eps += np.sum(
                    - 2.*(self.eta[c].T*self.std_beta[c]).T+
                    self.zeta[c] +
                    self.eta[c]*self.q[c]
                , axis=0)

            self.sigma_epsilon = np.clip(1. + sig_eps, 1e-12, 1. - 1e-12)

    cpdef m_step(self):
        """
        In the M-step, update the global hyperparameters of the model.
        """

        self.update_pi()
        self.update_sigma_beta()
        self.update_sigma_epsilon()

    cpdef objective(self):
        return self.elbo()
    cpdef elbo(self, sum_axis=None):
        """
        Compute the variational objective, the Evidence Lower-BOund (ELBO),
        from GWAS summary statistics.
        """

        # Concatenate the dictionary items for easy computation:
        var_gamma = dict_concat(self.var_gamma)
        null_gamma = 1. - dict_concat(self.compute_pip())  # The gamma for the null component
        var_mu = dict_concat(self.var_mu)
        var_sigma = dict_concat(self.var_sigma)

        if isinstance(self.pi, dict):
            pi = dict_concat(self.pi)
            null_pi = dict_concat(self.get_null_pi())
        else:
            pi = self.pi
            null_pi = self.get_null_pi()

        if isinstance(self.sigma_beta, dict):
            sigma_beta = dict_concat(self.sigma_beta)
        else:
            sigma_beta = self.sigma_beta

        q = dict_concat(self.q)
        eta = dict_concat(self.eta)
        zeta = dict_concat(self.zeta)

        std_beta = dict_concat(self.std_beta)

        elbo = 0.

        # -----------------------------------------------
        # (1) Compute the log of the joint density:

        #
        # (1.1) The following terms are an expansion of ||Y - X\beta||^2
        #
        # -N/2log(2pi*sigma_epsilon)
        elbo -= .5 * self.N * np.log(2 * np.pi * self.sigma_epsilon)

        # -Y'Y/(2*sigma_epsilon), where we assume Y'Y = N
        elbo -= .5 * (self.N / self.sigma_epsilon)

        # + (1./sigma_epsilon)*\beta*(XY), where we assume XY = N\hat{\beta}
        elbo += (self.N / self.sigma_epsilon) * np.sum((eta.T*std_beta).T, axis=0)

        # (-1/2sigma_epsilon)\beta'X'X\beta, where we assume X_j'X_j = N
        # Note that the q factor is equivalent to X'X\beta (excluding diagonal)
        elbo -= .5 * (self.N / self.sigma_epsilon) * (
                np.sum(eta*q + zeta, axis=0)
        )

        elbo -= (var_gamma * np.log(var_gamma / pi)).sum(axis=sum_axis)
        elbo -= (null_gamma * np.log(null_gamma / null_pi)).sum(axis=sum_axis)

        elbo += .5 * (var_gamma * (1. + np.log(var_sigma / sigma_beta) -
                                   (var_mu ** 2 + var_sigma) / sigma_beta)).sum(axis=sum_axis)

        try:
            if len(elbo) == 1:
                return elbo[0]
            else:
                return elbo
        except TypeError:
            return elbo

    cpdef get_sigma_epsilon(self):
        """
        Get the value of the hyperparameter sigma_epsilon
        """
        return self.sigma_epsilon

    cpdef get_sigma_beta(self, chrom=None):
        """
        Get the value of the hyperparameter sigma_beta
        :param chrom: Get the value of `sigma_beta` for a given chromosome.
        """
        if chrom is None:
            return self.sigma_beta
        else:
            if isinstance(self.sigma_beta, dict):
                return self.sigma_beta[chrom]
            else:
                return self.sigma_beta

    cpdef get_pi(self, chrom=None):
        """
        Get the value of the hyperparameter pi
        :param chrom: Get the value of `pi` for a given chromosome.
        """

        if chrom is None:
            return self.pi
        else:
            if isinstance(self.pi, dict):
                return self.pi[chrom]
            else:
                return self.pi

    cpdef get_null_pi(self, chrom=None):

        pi = self.get_pi(chrom=chrom)

        if isinstance(pi, dict):
            return {c: 1. - c_pi for c, c_pi in pi.items()}
        else:
            return 1. - pi

    cpdef get_proportion_causal(self):
        """
        Get the proportion of causal variants for the trait.
        """
        if isinstance(self.pi, dict):
            return dict_mean(self.pi, axis=0)
        else:
            return self.pi

    cpdef get_average_effect_size_variance(self):
        """
        Get the average per-SNP variance for the prior mixture components
        """
        if isinstance(self.pi, dict):
            pi = dict_concat(self.pi, axis=0)
        else:
            pi = self.pi

        if isinstance(self.sigma_beta, dict):
            sigma_beta = dict_concat(self.sigma_beta, axis=0)
        else:
            sigma_beta = self.sigma_beta

        return np.sum(pi * sigma_beta, axis=0)

    cpdef get_heritability(self):
        """
        Estimate the heritability, or proportion of variance explained by SNPs.
        """

        sigma_g = np.sum([
            np.sum(self.zeta[c] + self.q[c] * self.eta[c], axis=0)
            for c in self.shapes.keys()
        ], axis=0)

        h2g = sigma_g / (sigma_g + self.sigma_epsilon)

        return h2g

    cpdef to_theta_table(self):
        """
        Output the values for the hyperparameters (theta) to a pandas table.
        """

        theta_table = [
            {'Parameter': 'Residual_variance', 'Value': self.sigma_epsilon},
            {'Parameter': 'Heritability', 'Value': self.get_heritability()},
            {'Parameter': 'Proportion_causal', 'Value': self.get_proportion_causal()},
            {'Parameter': 'Average_effect_variance', 'Value': self.get_average_effect_size_variance()}
        ]

        if isinstance(self.sigma_beta, dict):
            sigmas = dict_mean(self.sigma_beta, axis=0)
        else:
            sigmas = self.sigma_beta

        try:
            sigmas = list(sigmas)
            for i in range(len(sigmas)):
                theta_table.append({'Parameter': f'sigma_beta_{i+1}', 'Value': sigmas[i]})
        except TypeError:
            theta_table.append({'Parameter': 'sigma_beta', 'Value': sigmas})

        return pd.DataFrame(theta_table)

    cpdef write_inferred_theta(self, f_name, sep="\t"):
        """
        Write the inferred (and fixed) hyperparameters to file.
        :param f_name: The file name
        :param sep: The separator for the hyperparameter file.
        """

        # Write the table to file:
        try:
            self.to_theta_table().to_csv(f_name, sep=sep, index=False)
        except Exception as e:
            raise e

    cpdef update_theta_history(self):
        """
        For all the tracked hyperparameters set in `tracked_theta`, append 
        their current values to the history object.
        """

        for tt in self.tracked_theta:
            if tt == 'pi':
                self.history['pi'].append(self.get_proportion_causal())
            if tt == 'heritability':
                self.history['heritability'].append(self.get_heritability())
            if tt == 'sigma_epsilon':
                self.history['sigma_epsilon'].append(self.sigma_epsilon)

    cpdef compute_pip(self):
        """
        Compute the posterior inclusion probability
        """
        return self.var_gamma

    cpdef compute_eta(self):
        """
        Compute the mean for the effect size under the variational posterior.
        """
        return {c: v*self.var_mu[c] for c, v in self.var_gamma.items()}

    cpdef compute_zeta(self):
        """
        Compute the expectation of the squared effect size under the variational posterior.
        """
        return {c: (v * (self.var_mu[c] ** 2 + self.var_sigma[c]))
                for c, v in self.var_gamma.items()}

    cpdef update_posterior_moments(self):
        """
        Update the posterior objects containing the posterior moments,
        including the PIP and posterior mean and variance for the effect size.
        """

        self.pip = {c: pip.copy() for c, pip in self.compute_pip().items()}
        self.post_mean_beta = {c: eta.copy() for c, eta in self.eta.items()}
        self.post_var_beta = {c: zeta - self.eta[c]**2 for c, zeta in self.zeta.items()}

    cpdef pseudo_validate(self,
                          validation_gdl=None,
                          sumstats_table=None,
                          std_beta=None, metric='pearson_correlation'):
        """
        Perform pseudo-validation of the inferred effect sizes by comparing them to 
        standardized marginal betas from an independent validation set. Here, we follow the pseudo-validation 
        procedures outlined in Mak et al. (2017) and Yang and Zhou (2020), where 
        the correlation between the PRS and the phenotype in an independent validation 
        cohort can be approximated with:
        
        Corr(PRS, y) ~= r'b / sqrt(b'Sb)
        
        Where `r` is the standardized marginal beta from a validation set, 
        `b` is the posterior mean for the effect size of each variant and `S` is the LD matrix.
        
        Alternatively, we can also approximate the R-squared from summary statistics as:
        
        R2(PRS, y) ~= 2*r'b - b'Sb
        
        The user can provide a `GWADataLoader` from the validation dataset, a `SumstatsTable` object,
        or a dictionary where the keys are the chromosome number and the values are the standardized 
        marginal betas.
        
        :param validation_gdl: An instance of `GWADataLoader` with the summary statistics table initialized.
        :param sumstats_table: An instance of `SumstatsTable`.
        :param std_beta: A dictionary where the keys are the chromosome number and the values are the standardized 
        marginal betas
        :param metric: The summary statistics metric to compute. Options are: `pearson_correlation` or `r2` (R-squared).
        """

        assert validation_gdl is not None or sumstats_table is not None or std_beta is not None

        if validation_gdl is not None or sumstats_table is not None:

            if validation_gdl is not None:
                eff_table = validation_gdl.to_snp_table(col_subset=['CHR', 'SNP', 'A1', 'A2', 'STD_BETA'],
                                                        per_chromosome=False)
            else:
                eff_table = sumstats_table.get_table(col_subset=['CHR', 'SNP', 'A1', 'A2', 'STD_BETA'])

            from magenpy.utils.model_utils import merge_snp_tables

            c_df = merge_snp_tables(self.gdl.to_snp_table(col_subset=['CHR', 'SNP', 'A1', 'A2'],
                                                          per_chromosome=False),
                                    eff_table,
                                    how='left',
                                    signed_statistics=['STD_BETA'])

            # Extract the standardized BETA from the validation GWAS:
            c_df['STD_BETA'] = c_df['STD_BETA'].fillna(0.)
            std_beta = {c: c_df.loc[c_df['CHR'] == c, 'STD_BETA'].values
                        for c in c_df['CHR'].unique()}

        else:
            # If the user provides a dictionary of standardized marginal betas,
            # check that the shapes match:
            assert all([len(b) == self.gdl.shapes[c] for c, b in std_beta.items()])

        # Compute the dot product between the standardized beta from the validation
        # summary statistics and the posterior mean beta:
        rb = np.sum([
            np.sum((post_mean.T * std_beta[c]).T, axis=0)
            for c, post_mean in self.post_mean_beta.items()
        ], axis=0)

        # Compute the variance of the PRS (using LD information from training)
        bsb = np.sum([
            np.sum(post_mean*(self.q[c] + post_mean), axis=0)
            for c, post_mean in self.post_mean_beta.items()
        ], axis=0)

        if metric == 'pearson_correlation':
            return rb / np.sqrt(bsb)
        else:
            return 2.*rb - bsb

    cpdef fit(self, max_iter=1000, theta_0=None, param_0=None,
              continued=False, f_abs_tol=1e-3, x_abs_tol=1e-8, max_elbo_drops=10,
              annealing_schedule='linear', annealing_steps=0, init_temperature=5.):
        """
        Fit the model parameters to data.

        :param max_iter: Maximum number of iterations. 
        :param theta_0: A dictionary of values to initialize the hyperparameters
        :param param_0: A dictionary of values to initialize the variational parameters
        :param continued: If true, continue the model fitting for more iterations.
        :param f_abs_tol: The absolute tolerance threshold for the objective (ELBO)
        :param x_abs_tol: The absolute tolerance threshold for the parameters (gamma)
        :param max_elbo_drops: The maximum number of times the objective is allowed to drop before termination.
        :param annealing_schedule: The schedule for the variational annealing procedure 
        (Options: 'linear', 'harmonic', 'geometric').
        :param annealing_steps: The number of annealing steps to perform before turning back to 
        optimizing the original variational objective.
        :param init_temperature: The initial temperature for the annealing procedure (T > 1).
        """

        if not continued:
            self.initialize(theta_0, param_0)
            start_idx = 1
        else:
            start_idx = len(self.history['ELBO']) + 1

        if self.load_ld:

            if self.verbose:
                print("> Loading LD matrices into memory...")

            self.gdl.load_ld()


        if annealing_schedule is not None and annealing_steps > 0:

            assert annealing_schedule in ('linear', 'harmonic', 'geometric')
            assert init_temperature > 1.

            self.inv_temperature = 1./init_temperature

            for i in tqdm(range(annealing_steps + 1, 1, -1),
                          total=annealing_steps,
                          desc="Annealing steps: "):

                self.e_step()
                self.m_step()

                if annealing_schedule == 'linear':
                    delta = (1. - 1./init_temperature) / (annealing_steps - 1)
                    self.inv_temperature = 1./init_temperature + delta*(annealing_steps - i)
                elif annealing_schedule == 'harmonic':
                    delta = (init_temperature - 1) / (annealing_steps - 1)
                    self.inv_temperature = 1./(1. + delta*(i - 1))
                elif annealing_schedule == 'geometric':
                    delta = init_temperature**(1./ (annealing_steps - 1)) - 1
                    self.inv_temperature = (1 + delta)**(-(i - 1))

            self.inv_temperature = 1.

        elbo_dropped_count = 0
        converged = False

        if self.verbose:
            print("> Performing model fit...")
            print(f"> Using up to {self.threads} threads.")

        for i in tqdm(range(start_idx, start_idx + max_iter), disable=not self.verbose):

            self.update_theta_history()

            self.e_step()
            self.m_step()

            self.history['ELBO'].append(self.elbo())

            if i > 1:

                curr_elbo = self.history['ELBO'][i - 1]
                prev_elbo = self.history['ELBO'][i - 2]

                # Check for convergence in the objective + parameters:
                if np.isclose(prev_elbo, curr_elbo, atol=f_abs_tol, rtol=0.):
                    print(f"Converged at iteration {i} || ELBO: {curr_elbo:.6f}")
                    converged = True
                    break
                elif all([
                    np.allclose(v, self.pip[c], atol=x_abs_tol, rtol=0.)
                    for c, v in self.compute_pip().items()
                ]):
                    print(f"Converged at iteration {i} | ELBO: {curr_elbo:.6f}")
                    converged = True
                    break

                # Check to see if the objective drops due to numerical instabilities:
                if curr_elbo < prev_elbo:
                    elbo_dropped_count += 1
                    warnings.warn(f"Iteration {i}: ELBO dropped from {prev_elbo:.6f} "
                                  f"to {curr_elbo:.6f}.")

                    if elbo_dropped_count > max_elbo_drops:
                        warnings.warn("The optimization is halted due to numerical instabilities!")
                        break

                    continue

                # Check if the ELBO behaves in unexpected/pathological ways:
                if abs((curr_elbo - prev_elbo) / prev_elbo) > 1. and abs(curr_elbo - prev_elbo) > 1e3:
                    raise OptimizationDivergence(f"Stopping at iteration {i}: "
                                                 f"The optimization algorithm is not converging!\n"
                                                 f"Previous ELBO: {prev_elbo:.6f} | "
                                                 f"Current ELBO: {curr_elbo:.6f}")
                elif self.get_heritability() >= 1.:
                    raise OptimizationDivergence(f"Stopping at iteration {i}: "
                                                 f"The optimization algorithm is not converging!\n"
                                                 f"Value of estimated heritability exceeded 1.")
            self.update_posterior_moments()

        if converged:
            self.update_posterior_moments()
        elif i - start_idx == max_iter - 1:
            warnings.warn("Maximum iterations reached without convergence. "
                          "You may need to run the model for more iterations.")

        if self.verbose:
            print(f"> Final ELBO: {self.history['ELBO'][len(self.history['ELBO'])-1]:.6f}")
            print(f"> Estimated heritability: {self.get_heritability():.6f}")
            print(f"> Estimated proportion of causal variants: {self.get_proportion_causal():.6f}")

        return self

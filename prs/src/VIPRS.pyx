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
import warnings
from tqdm import tqdm
from libc.math cimport log

from .PRSModel cimport PRSModel
from .exceptions import OptimizationDivergence
from .c_utils cimport dot, mt_sum, elementwise_add_mult, sigmoid, clip


cdef class VIPRS(PRSModel):

    def __init__(self, gdl, fix_params=None, load_ld=True, verbose=True, threads=4):
        """
        TODO: Restructure the code to use sample size per SNP instead of total GWAS sample size.
        :param gdl: An instance of GWAS data loader
        :param fix_params: A dictionary of parameters with their fixed values.
        :param load_ld: A flag that specifies whether to load the LD matrix to memory.
        """

        super().__init__(gdl)

        self.threads = threads
        self.var_gamma = {}
        self.var_mu_beta = {}
        self.var_sigma_beta = {}
        self.mean_beta = {}  # E[B] = \gamma*\mu_beta
        self.mean_beta_sq = {}  # E[B^2] = \gamma*(\mu_beta^2 + \sigma_beta^2)
        self.q = {}

        self.load_ld = load_ld
        self.ld = self.gdl.get_ld_matrices()
        self.ld_bounds = self.gdl.get_ld_boundaries()
        self.beta_hat = self.gdl.compute_xy_per_snp()

        self.fix_params = fix_params or {}

        self.verbose = verbose
        self.history = {}

    cpdef initialize(self):

        if self.verbose:
            print("> Initializing model parameters")

        self.initialize_theta()
        self.initialize_variational_params()
        self.init_history()

    cpdef init_history(self):

        self.history = {
            'ELBO': []
        }

    cpdef initialize_theta(self):
        """
        This method initializes the global hyper-parameters
        TODO: Implement better strategies for initializing hyperparameters
        :return:
        """

        if 'pi' not in self.fix_params:
            self.pi = np.random.uniform(low=1. / self.M, high=.5)
        else:
            self.pi = self.fix_params['pi']

        if 'sigma_epsilon' not in self.fix_params:
            if 'sigma_beta' not in self.fix_params:
                try:
                    naive_h2g = self.gdl.estimate_snp_heritability()
                except Exception as e:
                    naive_h2g = np.random.uniform(low=.01, high=.99)

                self.sigma_epsilon = clip(1. - naive_h2g, .01, .99)
                self.sigma_beta = naive_h2g / (self.pi * self.M)
            else:
                self.sigma_beta = self.fix_params['sigma_beta']
                self.sigma_epsilon = clip(1. - self.sigma_beta*(self.pi * self.M), .01, .99)
        else:
            self.sigma_epsilon = self.fix_params['sigma_epsilon']

            if 'sigma_beta' in self.fix_params:
                self.sigma_beta = self.fix_params['sigma_beta']
            else:
                self.sigma_beta = (1. - self.sigma_epsilon) / (self.pi * self.M)


    cpdef initialize_variational_params(self):
        """
        This method initializes the variational parameters.
        :return:
        """

        self.var_mu_beta = {}
        self.var_sigma_beta = {}
        self.var_gamma = {}

        for c, c_size in self.shapes.items():

            self.var_gamma[c] = np.repeat(self.pi, c_size)
            self.var_mu_beta[c] = np.random.normal(scale=self.sigma_beta, size=c_size)
            self.var_sigma_beta[c] = np.repeat(self.sigma_beta, c_size)

            self.mean_beta[c] = self.var_gamma[c]*self.var_mu_beta[c]
            self.mean_beta_sq[c] = self.var_gamma[c]*(self.var_mu_beta[c]**2 + self.var_sigma_beta[c])

    cpdef e_step(self):
        """
        In the E-step, we update the variational parameters
        for each SNP.
        :return:
        """

        cdef:
            unsigned int j, start, end, j_idx
            double u_j
            double[::1] var_gamma, var_mu_beta, var_sigma_beta, mean_beta, mean_beta_sq, beta_hat, q, Dj
            long[::1] N
            long[:, ::1] ld_bound
            # The log(pi) for the gamma updates
            double logodds_pi = log(self.pi / (1. - self.pi))

        for c, c_size in self.shapes.items():

            # Updates for sigma_beta variational parameters:
            self.var_sigma_beta[c] = self.sigma_epsilon / (self.Nj[c] + self.sigma_epsilon / self.sigma_beta)

            beta_hat = self.beta_hat[c]
            var_gamma = self.var_gamma[c]
            var_mu_beta = self.var_mu_beta[c]
            var_sigma_beta = self.var_sigma_beta[c]
            mean_beta = self.mean_beta[c]
            mean_beta_sq = self.mean_beta_sq[c]
            ld_bound = self.ld_bounds[c]
            N = self.Nj[c]
            q = np.zeros(shape=c_size)

            for j, Dj in enumerate(self.ld[c]):

                start, end = ld_bound[:, j]
                j_idx = j - start

                var_mu_beta[j] = (beta_hat[j] - dot(Dj, mean_beta[start: end], self.threads) +
                                  Dj[j_idx]*mean_beta[j]) / (1. + self.sigma_epsilon / (N[j] * self.sigma_beta))

                u_j = (logodds_pi + .5*log(var_sigma_beta[j] / self.sigma_beta) +
                       (.5/var_sigma_beta[j])*var_mu_beta[j]*var_mu_beta[j])
                var_gamma[j] = clip(sigmoid(u_j), 1e-6, 1. - 1e-6)

                mean_beta[j] = var_gamma[j]*var_mu_beta[j]
                mean_beta_sq[j] = var_gamma[j]*(var_mu_beta[j]*var_mu_beta[j] + var_sigma_beta[j])

                if j_idx > 0:
                    # Update the q factor for snp j by adding the contribution of previous SNPs.
                    q[j] = dot(Dj[:j_idx], mean_beta[start: j], self.threads)
                    # Update the q factors for all previously updated SNPs that are in LD with SNP j
                    q[start: j] = elementwise_add_mult(q[start: j], Dj[:j_idx], mean_beta[j], self.threads)

            self.var_gamma[c] = np.array(var_gamma)
            self.var_mu_beta[c] = np.array(var_mu_beta)

            self.mean_beta[c] = np.array(mean_beta)
            self.mean_beta_sq[c] = np.array(mean_beta_sq)
            self.q[c] = np.array(q)

    cpdef update_pi(self):

        if 'pi' not in self.fix_params:

            var_gamma_sum = np.sum([
                mt_sum(self.var_gamma[c], self.threads)
                for c in self.shapes
            ])

            self.pi = clip(var_gamma_sum / self.M, 1./self.M, 1. - 1./self.M)

    cpdef update_sigma_beta(self):

        if 'sigma_beta' not in self.fix_params:

            var_gamma_sum = np.sum([
                mt_sum(self.var_gamma[c], self.threads)
                for c in self.shapes
            ])

            self.sigma_beta = np.sum([mt_sum(self.mean_beta_sq[c], self.threads)
                                      for c in self.shapes]) / var_gamma_sum

            self.sigma_beta = clip(self.sigma_beta, 1e-12, 1. - 1e-12)

    cpdef update_sigma_epsilon(self):

        if 'sigma_epsilon' not in self.fix_params:

            sig_eps = 0.

            for c, c_size in self.shapes.items():

                sig_eps += (
                        - 2. * dot(self.mean_beta[c], self.beta_hat[c], self.threads) +
                        mt_sum(self.mean_beta_sq[c], self.threads) +
                        dot(self.mean_beta[c], self.q[c], self.threads)
                )

            self.sigma_epsilon = clip(1. + sig_eps, 1e-12, 1. - 1e-12)


    cpdef m_step(self):
        """
        In the M-step, we update the global parameters of
        the model.
        :return:
        """

        self.update_pi()
        self.update_sigma_beta()
        self.update_sigma_epsilon()

    cpdef objective(self):

        loglik = 0.  # log of joint density
        ent = 0.  # entropy

        # Add the fixed quantities:
        loglik -= .5 * self.N * (log(2 * np.pi * self.sigma_epsilon) + 1. / self.sigma_epsilon)
        loglik -= .5 * self.M * log(2. * np.pi * self.sigma_beta)
        loglik += self.M * log(1. - self.pi)

        ent += .5 * self.M * log(2. * np.pi * np.e * self.sigma_beta)

        for c in self.shapes:

            loglik += (-.5 * self.N / self.sigma_epsilon) * (
                    - 2. * dot(self.mean_beta[c], self.beta_hat[c], self.threads)
                    + dot(self.mean_beta[c], self.q[c], self.threads)
                    + mt_sum(self.mean_beta_sq[c], self.threads)
            )

            loglik += (-.5 / self.sigma_beta) * (
                    np.sum(self.mean_beta_sq[c]) +
                    self.sigma_beta * np.sum(1 - self.var_gamma[c])
            )

            loglik += log(self.pi / (1. - self.pi)) * mt_sum(self.var_gamma[c], self.threads)

            ent += .5 * dot(self.var_gamma[c], np.log(self.var_sigma_beta[c] / self.sigma_beta), self.threads)

            ent -= dot(self.var_gamma[c], np.log(self.var_gamma[c] / (1. - self.var_gamma[c])), self.threads)
            ent -= mt_sum(np.log(1. - self.var_gamma[c]), self.threads)

        elbo = loglik + ent

        return elbo

    cpdef get_proportion_causal(self):
        return self.pi

    cpdef get_heritability(self):

        sigma_g = np.sum([
            mt_sum(self.mean_beta_sq[c], self.threads) +
            dot(self.q[c], self.mean_beta[c], self.threads)
            for c in self.shapes
        ])

        h2g = sigma_g / (sigma_g + self.sigma_epsilon)

        return h2g

    cpdef fit(self, max_iter=1000, continued=False, ftol=1e-4, xtol=1e-4, max_elbo_drops=10):

        if not continued:
            self.initialize()

        if self.load_ld:
            self.gdl.load_ld()

        elbo_dropped_count = 0
        converged = False

        if self.verbose:
            print("> Performing model fit...")
            print(f"> Using up to {self.threads} threads.")

        for i in tqdm(range(1, max_iter + 1), disable=not self.verbose):
            self.e_step()
            self.m_step()

            self.history['ELBO'].append(self.objective())

            if i > 1:

                curr_elbo = self.history['ELBO'][i - 1]
                prev_elbo = self.history['ELBO'][i - 2]

                if curr_elbo < prev_elbo:
                    elbo_dropped_count += 1
                    warnings.warn(f"Iteration {i}: ELBO dropped from {prev_elbo:.6f} "
                                  f"to {curr_elbo:.6f}.")

                if abs(curr_elbo - prev_elbo) <= ftol:
                    print(f"Converged at iteration {i} | ELBO: {curr_elbo:.6f}")
                    break
                elif max([np.abs(v - self.inf_beta[c]).max() for c, v in self.mean_beta.items()]) <= xtol:
                    print(f"Converged at iteration {i} | ELBO: {curr_elbo:.6f}")
                    break
                elif elbo_dropped_count > max_elbo_drops:
                    warnings.warn("The optimization is halted due to numerical instabilities!")
                    break

                if i > 2:
                    if abs((curr_elbo - prev_elbo) / prev_elbo) > 1. and abs(curr_elbo - prev_elbo) > 10.:
                        raise OptimizationDivergence(f"Stopping at iteration {i}: "
                                                     f"The optimization algorithm is not converging!\n"
                                                     f"Previous ELBO: {prev_elbo:.6f} | "
                                                     f"Current ELBO: {curr_elbo:.6f}")
                    elif self.get_heritability() >= 1.:
                        raise OptimizationDivergence(f"Stopping at iteration {i}: "
                                                     f"The optimization algorithm is not converging!\n"
                                                     f"Value of estimated heritability exceeded 1.")

            self.pip = {c: v.copy() for c, v in self.var_gamma.items()}
            self.inf_beta = {c: v.copy() for c, v in self.mean_beta.items()}

        if i == max_iter:
            warnings.warn("Max iterations reached without convergence. "
                          "You may need to run the model for more iterations.")

        if self.verbose:
            print(f"> Final ELBO: {self.history['ELBO'][i-1]:.6f}")
            print(f"> Estimated heritability: {self.get_heritability():.6f}")
            print(f"> Estimated proportion of causal variants: {self.get_proportion_causal():.6f}")

        return self

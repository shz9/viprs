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
from threadpoolctl import threadpool_limits
from libc.math cimport log
from .PRSModel cimport PRSModel
from .exceptions import OptimizationDivergence
from .c_utils cimport dot, elementwise_add_mult, sigmoid, clip


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
        self.var_mu_beta = {}
        self.var_sigma_beta = {}
        self.var_gamma = {}
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

        self.initialize_variational_params()
        self.initialize_theta()
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

        if 'sigma_beta' not in self.fix_params:
            self.sigma_beta = np.random.uniform(low=1e-6, high=.1)
        else:
            self.sigma_beta = self.fix_params['sigma_beta']

        if 'sigma_epsilon' not in self.fix_params:
            self.sigma_epsilon = np.random.uniform(low=.5, high=1.)
        else:
            self.sigma_epsilon = self.fix_params['sigma_epsilon']

        if 'pi' not in self.fix_params:
            self.pi = np.random.uniform(low=1. / self.M, high=.5)
        else:
            self.pi = self.fix_params['pi']

    cpdef initialize_variational_params(self):
        """
        This method initializes the variational parameters.
        :return:
        """

        self.var_mu_beta = {}
        self.var_sigma_beta = {}
        self.var_gamma = {}

        for c, c_size in self.shapes.items():

            self.var_gamma[c] = np.random.uniform(size=c_size)
            self.var_mu_beta[c] = np.random.normal(scale=1./np.sqrt(self.M), size=c_size)
            self.var_sigma_beta[c] = np.repeat(1./self.M, c_size)

    cpdef e_step(self):
        """
        In the E-step, we update the variational parameters
        for each SNP.
        :return:
        """

        cdef:
            unsigned int j, start, end, j_idx
            double u_j
            double[::1] var_prod, var_mu_beta, var_sigma_beta, var_gamma, beta_hat, q, Dj
            long[:, ::1] ld_bound
            # The denominator for the mu_beta updates:
            double denom = (1. + self.sigma_epsilon / (self.N * self.sigma_beta))
            # The log(pi) for the gamma updates
            double logodds_pi = log(self.pi / (1. - self.pi))

        for c, c_size in self.shapes.items():

            # Updates for sigma_beta variational parameters:
            self.var_sigma_beta[c] = np.repeat(
                self.sigma_epsilon / (self.N + self.sigma_epsilon / self.sigma_beta),
                c_size
            )

            beta_hat = self.beta_hat[c]
            var_mu_beta = self.var_mu_beta[c]
            var_sigma_beta = self.var_sigma_beta[c]
            var_gamma = self.var_gamma[c]
            ld_bound = self.ld_bounds[c]
            q = np.zeros(shape=c_size)

            var_prod = np.multiply(var_gamma, var_mu_beta)

            for j, Dj in enumerate(self.ld[c]):

                start, end = ld_bound[:, j]
                j_idx = j - start

                var_mu_beta[j] = (beta_hat[j] - dot(Dj, var_prod[start: end]) +
                                  Dj[j_idx]*var_prod[j]) / denom

                u_j = (logodds_pi + .5*log(var_sigma_beta[j] / self.sigma_beta) +
                       (.5/var_sigma_beta[j])*var_mu_beta[j]*var_mu_beta[j])
                var_gamma[j] = clip(sigmoid(u_j), 1e-6, 1. - 1e-6)

                var_prod[j] = var_gamma[j]*var_mu_beta[j]

                if j_idx > 0:
                    # Update the q factor for snp j by adding the contribution of previous SNPs.
                    q[j] = dot(Dj[:j_idx], var_prod[start: j])
                    # Update the q factors for all previously updated SNPs that are in LD with SNP j
                    q[start: j] = elementwise_add_mult(q[start: j], Dj[:j_idx], var_prod[j])


            self.q[c] = np.array(q)
            self.var_gamma[c] = np.array(var_gamma)
            self.var_mu_beta[c] = np.array(var_mu_beta)

    cpdef update_pi(self):

        if 'pi' not in self.fix_params:

            var_gamma_sum = np.sum([
                np.sum(self.var_gamma[c])
                for c in self.shapes
            ])

            self.pi = clip(var_gamma_sum / self.M, 1./self.M, 1.)

    cpdef update_sigma_beta(self):

        if 'sigma_beta' not in self.fix_params:

            var_gamma_sum = np.sum([
                np.sum(self.var_gamma[c])
                for c in self.shapes
            ])

            self.sigma_beta = np.sum([
                np.dot(self.var_gamma[c],
                       self.var_mu_beta[c] ** 2 + self.var_sigma_beta[c])
                for c in self.var_mu_beta]) / var_gamma_sum

            self.sigma_beta = clip(self.sigma_beta, 1e-12, 1e12)

    cpdef update_sigma_epsilon(self):

        if 'sigma_epsilon' not in self.fix_params:

            sig_eps = 0.

            for c, c_size in self.shapes.items():
                beta_hat = self.beta_hat[c]
                var_gamma = self.var_gamma[c]
                var_mu_beta = self.var_mu_beta[c]
                var_sigma_beta = self.var_sigma_beta[c]
                var_prod = np.multiply(var_gamma, var_mu_beta)
                q = self.q[c]

                sig_eps += (
                        - 2. * np.dot(var_prod, beta_hat) +
                        np.dot(var_gamma, var_mu_beta * var_mu_beta + var_sigma_beta) +
                        np.dot(var_prod, q)
                )

            self.sigma_epsilon = clip(1. + sig_eps, 1e-12, 1e12)


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
        loglik -= .5 * self.N * (np.log(2 * np.pi * self.sigma_epsilon) + 1. / self.sigma_epsilon)
        loglik -= .5 * self.M * np.log(2. * np.pi * self.sigma_beta)
        loglik += self.M * np.log(1. - self.pi)

        ent += .5 * self.M * np.log(2. * np.pi * np.e * self.sigma_beta)

        for c in self.shapes:
            beta_hat = self.beta_hat[c]
            gamma_mu = self.var_gamma[c] * self.var_mu_beta[c]
            gamma_mu_sig = self.var_gamma[c] * (self.var_mu_beta[c] ** 2 + self.var_sigma_beta[c])

            loglik += (-.5 * self.N / self.sigma_epsilon) * (
                    - 2. * np.dot(gamma_mu, beta_hat)
                    + np.dot(gamma_mu, self.q[c])
                    + np.sum(gamma_mu_sig)
            )

            loglik += (-.5 / self.sigma_beta) * (
                    np.sum(gamma_mu_sig) +
                    self.sigma_beta * np.sum(1 - self.var_gamma[c])
            )

            loglik += np.log(self.pi / (1. - self.pi)) * np.sum(self.var_gamma[c])

            ent += .5 * np.dot(self.var_gamma[c], np.log(self.var_sigma_beta[c] / self.sigma_beta))

            ent -= np.dot(self.var_gamma[c], np.log(self.var_gamma[c] / (1. - self.var_gamma[c])))
            ent -= np.sum(np.log(1. - self.var_gamma[c]))

        elbo = loglik + ent

        return elbo

    cpdef get_proportion_causal(self):
        return self.pi

    cpdef get_heritability(self):

        sigma_g = np.sum([
            np.sum(self.var_gamma[c] * (self.var_mu_beta[c] ** 2 + self.var_sigma_beta[c])) +
            np.dot(self.q[c], self.var_gamma[c] * self.var_mu_beta[c])
            for c in self.var_gamma
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

        with threadpool_limits(limits=self.threads, user_api='blas'):

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
                    elif max([np.abs(v - self.pip[c]).max() for c, v in self.var_gamma.items()]) <= xtol:
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

                self.pip = {c: v.copy() for c, v in self.var_gamma.items()}
                self.inf_beta = {c: v * self.var_mu_beta[c]
                                 for c, v in self.var_gamma.items()}

        if i == max_iter:
            warnings.warn("Max iterations reached without convergence. "
                          "You may need to run the model for more iterations.")

        return self

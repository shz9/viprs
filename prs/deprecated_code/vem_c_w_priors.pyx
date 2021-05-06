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
from libc.math cimport exp, log
from ..src.PRSModel cimport PRSModel
from ..src.c_utils cimport dot
from prs.gwasimulator.c_utils import zarr_islice

cdef class vem_prs_wp(PRSModel):

    """
    This is an implementation of the EM algorithm with priors,
    similar to Algorithm 1 in Huang et al. (2016)
    """

    cdef public double pi, sigma_beta, sigma_epsilon, ld_prod, b_a0, b_b0, ig_a0, ig_b0
    cdef public dict var_mu_beta, var_sigma_beta, var_gamma, beta_hat, ld, ld_bounds, history, shapes, fix_params

    def __init__(self, gdl, fix_params=None,
                 b_a0=0.05, b_b0=0.5,
                 ig_a0=.5, ig_b0=.5):
        """
        :param gdl: An instance of GWAS data loader
        """

        super().__init__(gdl)

        self.b_a0 = b_a0
        self.b_b0 = b_b0
        self.ig_a0 = ig_a0
        self.ig_b0 = ig_b0

        self.ld = self.gdl.get_ld_matrices()
        self.ld_bounds = self.gdl.get_ld_boundaries()
        self.beta_hat = self.gdl.beta_hats

        self.fix_params = fix_params or {}

        # Helpers:
        self.shapes = {c: gt['G'].shape[1] for c, gt in gdl.genotypes.items()}

        self.initialize()

    def initialize(self):
        self.beta_hat = self.gdl.beta_hats
        self.initialize_variational_params()
        self.initialize_theta()
        self.init_history()

    def init_history(self):

        self.history = {
            'ELBO': [],
            'pi': [],
            'sigma_beta': [],
            'sigma_epsilon': [],
            'heritability': []
        }

    def initialize_theta(self):

        if 'sigma_beta' in self.fix_params:
            self.sigma_beta = self.fix_params['sigma_beta']
        else:
            self.sigma_beta = 1./ self.M #np.random.uniform()

        if 'sigma_epsilon' in self.fix_params:
            self.sigma_epsilon = self.fix_params['sigma_epsilon']
        else:
            self.sigma_epsilon = 0.8 #np.random.uniform()

        if 'pi' in self.fix_params:
            self.pi = self.fix_params['pi']
        else:
            self.pi = 0.1 #np.random.uniform()

    def initialize_variational_params(self):

        self.var_mu_beta = {}
        self.var_sigma_beta = {}
        self.var_gamma = {}

        for c, c_size in self.shapes.items():

            self.var_gamma[c] = np.zeros(shape=c_size)
            self.var_mu_beta[c] = np.zeros(shape=c_size) #np.random.normal(size=c_size)
            self.var_sigma_beta[c] = np.repeat(1./self.M, c_size) #np.random.uniform(size=c_size)

    def e_step(self):
        """
        In the E-step, we update the variational parameters
        for each SNP.
        :return:
        """

        cdef unsigned int i
        cdef double u_j
        cdef double[::1] var_prod, var_mu_beta, var_sigma_beta, var_gamma, beta_hat, Di
        cdef long[:, ::1] ld_bound

        cdef double denom = (1. + self.sigma_epsilon / (self.N * self.sigma_beta))
        cdef double log_pi = log(self.pi / (1. - self.pi))

        for c, c_size in self.shapes.items():

            # Updates for sigma_beta variational parameters:
            self.var_sigma_beta[c] = np.repeat(
                self.sigma_epsilon / (self.N + self.sigma_epsilon / self.sigma_beta),
                c_size
            )

            beta_hat = self.beta_hat[c].values
            var_mu_beta = self.var_mu_beta[c]
            var_sigma_beta = self.var_sigma_beta[c]
            var_gamma = self.var_gamma[c]
            ld_bound = self.ld_bounds[c]

            var_prod = np.multiply(var_gamma, var_mu_beta)

            for i, Di in enumerate(zarr_islice(self.ld[c])):

                var_mu_beta[i] = (beta_hat[i] - dot(Di, var_prod[ld_bound[0, i]: ld_bound[1, i]]) +
                                  Di[i - ld_bound[0, i]]*var_prod[i]) / denom

                u_i = (log_pi + .5*log(var_sigma_beta[i] / self.sigma_beta) +
                       (.5/var_sigma_beta[i])*var_mu_beta[i]*var_mu_beta[i])
                var_gamma[i] = 1./(1. + exp(-u_i))

                var_prod[i] = var_gamma[i]*var_mu_beta[i]

            self.var_gamma[c] = np.clip(var_gamma, 1e-6, 1. - 1e-6)
            self.var_mu_beta[c] = np.array(var_mu_beta)

    def m_step(self):
        """
        In the M-step, we update the global parameters of
        the model.
        :return:
        """

        # Update pi:

        var_gamma_sum = np.sum([
            np.sum(self.var_gamma[c])
            for c in self.var_gamma
        ])

        if 'pi' not in self.fix_params:
            b_a = var_gamma_sum + self.b_a0
            b_b = self.M - var_gamma_sum + self.b_b0
            self.pi = b_a / (b_a + b_b)
            self.pi = np.clip(self.pi, 1./self.M, 1.)

        self.history['pi'].append(self.pi)

        if 'sigma_beta' not in self.fix_params:
            # Update sigma_beta:
            self.sigma_beta = np.sum([
                np.dot(self.var_gamma[c],
                       self.var_mu_beta[c] ** 2 + self.var_sigma_beta[c])
                for c in self.var_mu_beta]) / var_gamma_sum

            self.sigma_beta = np.clip(self.sigma_beta, 1e-12, np.inf)

        self.history['sigma_beta'].append(self.sigma_beta)

        # Update sigma_epsilon

        cdef double sig_e = 0., ld_prod = 0.
        cdef unsigned int i, j, c_size
        cdef double[::1] var_prod, var_gamma, var_mu_beta, var_sigma_beta, beta_hat
        cdef const double[::1] Di
        cdef long[:, ::1] ld_bound

        for c, c_size in self.shapes.items():

            beta_hat = self.beta_hat[c].values
            var_gamma = self.var_gamma[c]
            var_mu_beta = self.var_mu_beta[c]
            var_sigma_beta = self.var_sigma_beta[c]
            var_prod = np.multiply(var_gamma, var_mu_beta)
            ld_bound = self.ld_bounds[c]

            for i, Di in enumerate(zarr_islice(self.ld[c])):

                sig_e += .5 * var_gamma[i] * (pow(var_mu_beta[i], 2) + var_sigma_beta[i])
                sig_e -= var_prod[i] * beta_hat[i]

                for j in range(i + 1, ld_bound[1, i]):
                    ld_prod += Di[j - ld_bound[0, i]] * var_prod[i] * var_prod[j]

        sig_e += ld_prod
        self.ld_prod = 2. * ld_prod

        if 'sigma_epsilon' not in self.fix_params:
            sig_e_mle = 1. + 2.*sig_e
            final_sig_e = (self.N*sig_e_mle + 2.*self.ig_b0) / (2.*self.ig_a0 + self.N + 2)
            self.sigma_epsilon = np.clip(final_sig_e, 1e-12, np.inf)

        self.history['sigma_epsilon'].append(self.sigma_epsilon)

    def objective(self):

        loglik = 0.  # log of joint density
        ent = 0.  # entropy

        # Add the fixed quantities:

        loglik -= .5 * self.N * (np.log(2 * np.pi * self.sigma_epsilon) + 1. / self.sigma_epsilon)
        loglik -= .5 * self.M * np.log(2. * np.pi * self.sigma_beta)
        loglik += self.M * np.log(1. - self.pi)

        ent += .5 * self.M * np.log(2. * np.pi * np.e * self.sigma_beta)

        for c in self.var_mu_beta:
            beta_hat = self.beta_hat[c]
            gamma_mu = self.var_gamma[c] * self.var_mu_beta[c]
            gamma_mu_sig = self.var_gamma[c] * (self.var_mu_beta[c] ** 2 + self.var_sigma_beta[c])

            loglik += (-.5 * self.N / self.sigma_epsilon) * (
                    - 2. * np.dot(gamma_mu, beta_hat)
                    + self.ld_prod
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

        self.history['ELBO'].append(elbo)
        #self.history['loglikelihood'].append(loglik)
        #self.history['entropy'].append(ent)

        return elbo

    def get_heritability(self):

        sigma_g = np.sum([
            np.sum(self.var_gamma[c] * (self.var_mu_beta[c] ** 2 + self.var_sigma_beta[c]))
            for c in self.var_gamma
        ])

        h2g = sigma_g / (sigma_g + self.sigma_epsilon)

        self.history['heritability'].append(h2g)

        return h2g

    def fit(self, max_iter=500, continued=False, tol=1e-6, max_elbo_drops=10):

        if not continued:
            self.initialize()

        elbo_dropped_count = 0
        converged = False

        for i in range(1, max_iter + 1):

            self.e_step()
            self.m_step()
            self.objective()
            self.get_heritability()

            if i > 1:

                if self.history['ELBO'][-1] < self.history['ELBO'][-2]:
                    elbo_dropped_count += 1
                    print(f"Warning (Iteration {i}): ELBO dropped from {self.history['ELBO'][-2]:.6f} "
                          f"to {self.history['ELBO'][-1]:.6f}!")

                if np.abs(self.history['ELBO'][-1] - self.history['ELBO'][-2]) <= tol:
                    print(f"Converged at iteration {i} | ELBO: {self.history['ELBO'][-1]:.6f}")
                    break
                elif elbo_dropped_count > max_elbo_drops:
                    print("The optimization is halted due to numerical instabilities!")
                    break

        if i == max_iter:
            print("Max iterations reached without convergence. "
                  "You may need to run the model for more iterations.")

        self.pip = self.var_gamma
        self.inf_beta = {c: self.var_gamma[c] * self.var_mu_beta[c]
                         for c, v in self.var_gamma.items()}

        return self

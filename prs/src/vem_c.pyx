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
from libc.math cimport log
from .PRSModel cimport PRSModel
from .c_utils cimport dot, sigmoid, clip


cdef class vem_prs(PRSModel):

    cdef public:
        double pi, sigma_beta, sigma_epsilon  # Global parameters
        double ld_prod  # Need to keep track of this quantity
        bint scale_prior, load_ld
        dict var_mu_beta, var_sigma_beta, var_gamma  # Variational parameters
        dict beta_hat, ld, ld_bounds  # Inputs to the algorithm
        dict history, fix_params  # Helpers

    def __init__(self, gdl, scale_prior=False, fix_params=None, load_ld=True):
        """
        :param scale_prior: If set to true, scale the prior over the parameters
                following the Carbonetto & Stephens (2012) model).
        :param gdl: An instance of GWAS data loader
        """

        super().__init__(gdl)

        self.var_mu_beta = {}
        self.var_sigma_beta = {}
        self.var_gamma = {}

        self.load_ld = load_ld
        self.ld = self.gdl.get_ld_matrices()
        self.ld_bounds = self.gdl.get_ld_boundaries()
        self.beta_hat = {c: b.values for c, b in self.gdl.beta_hats.items()}

        self.scale_prior = scale_prior
        self.fix_params = fix_params or {}
        self.fix_params = {k: np.array(v).flatten() for k, v in self.fix_params.items()}

        self.history = {}

        self.initialize()

    cpdef initialize(self):
        self.initialize_variational_params()
        self.initialize_theta()
        self.init_history()

    cpdef init_history(self):

        self.history = {
            'ELBO': [],
            'pi': [],
            'sigma_beta': [],
            'sigma_epsilon': [],
            'heritability': []
        }

    cpdef initialize_theta(self):

        if 'sigma_beta' not in self.fix_params:
            self.sigma_beta = np.random.uniform()
        else:
            self.sigma_beta = self.fix_params['sigma_beta'][0]

        if 'sigma_epsilon' not in self.fix_params:
            self.sigma_epsilon = np.random.uniform()
        else:
            self.sigma_epsilon = self.fix_params['sigma_epsilon'][0]

        if 'pi' not in self.fix_params:
            self.pi = np.random.uniform()
        else:
            self.pi = self.fix_params['pi'][0]

    cpdef initialize_variational_params(self):

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

        cdef unsigned int j, start, end
        cdef double u_j
        cdef double[::1] var_prod, var_mu_beta, var_sigma_beta, var_gamma, beta_hat, Dj
        cdef long[:, ::1] ld_bound

        # The prior variance parameter (can be defined in two ways):
        cdef double prior_var = self.sigma_epsilon*self.sigma_beta if self.scale_prior else self.sigma_beta

        # The denominator for the mu_beta updates:
        cdef double denom = (1. + self.sigma_epsilon / (self.N * prior_var))

        # The log(pi) for the gamma updates
        cdef double logodds_pi = log(self.pi / (1. - self.pi))

        for c, c_size in self.shapes.items():

            # Updates for sigma_beta variational parameters:
            self.var_sigma_beta[c] = np.repeat(
                self.sigma_epsilon / (self.N + self.sigma_epsilon / prior_var),
                c_size
            )

            beta_hat = self.beta_hat[c]
            var_mu_beta = self.var_mu_beta[c]
            var_sigma_beta = self.var_sigma_beta[c]
            var_gamma = self.var_gamma[c]
            ld_bound = self.ld_bounds[c]

            var_prod = np.multiply(var_gamma, var_mu_beta)

            for j, Dj in enumerate(self.ld[c]):

                start, end = ld_bound[:, j]

                var_mu_beta[j] = (beta_hat[j] - dot(Dj, var_prod[start: end]) +
                                  Dj[j - start]*var_prod[j]) / denom

                u_j = (logodds_pi + .5*log(var_sigma_beta[j] / prior_var) +
                       (.5/var_sigma_beta[j])*var_mu_beta[j]*var_mu_beta[j])
                var_gamma[j] = clip(sigmoid(u_j), 1e-6, 1. - 1e-6)

                var_prod[j] = var_gamma[j]*var_mu_beta[j]

            self.var_gamma[c] = np.array(var_gamma)
            self.var_mu_beta[c] = np.array(var_mu_beta)

    cpdef m_step(self):
        """
        In the M-step, we update the global parameters of
        the model.
        :return:
        """

        # Update pi:

        var_gamma_sum = np.sum([
            np.sum(self.var_gamma[c])
            for c in self.shapes
        ])

        if 'pi' not in self.fix_params:
            self.pi = var_gamma_sum / self.M
            self.pi = np.clip(self.pi, 1./self.M, 1.)

        self.history['pi'].append(self.pi)

        if 'sigma_beta' not in self.fix_params:
            # Update sigma_beta:
            self.sigma_beta = np.sum([
                np.dot(self.var_gamma[c],
                       self.var_mu_beta[c] ** 2 + self.var_sigma_beta[c])
                for c in self.var_mu_beta]) / var_gamma_sum

            if self.scale_prior:
                self.sigma_beta /= self.sigma_epsilon

            self.sigma_beta = np.clip(self.sigma_beta, 1e-12, self.sigma_epsilon)

        self.history['sigma_beta'].append(self.sigma_beta)

        # Update sigma_epsilon

        cdef double sig_e = 0., ld_prod = 0.
        cdef unsigned int i, j, c_size
        cdef double[::1] var_prod, var_gamma, var_mu_beta, var_sigma_beta, beta_hat, Di
        cdef long[:, ::1] ld_bound
        cdef double scale_prior_adj = (1. + 1./(self.N*self.sigma_beta)) if self.scale_prior else 1.

        for c, c_size in self.shapes.items():

            beta_hat = self.beta_hat[c]
            var_gamma = self.var_gamma[c]
            var_mu_beta = self.var_mu_beta[c]
            var_sigma_beta = self.var_sigma_beta[c]
            var_prod = np.multiply(var_gamma, var_mu_beta)
            ld_bound = self.ld_bounds[c]

            for i, Di in enumerate(self.ld[c]):

                sig_e += .5 * scale_prior_adj * var_gamma[i] * (var_mu_beta[i]*var_mu_beta[i] + var_sigma_beta[i])
                sig_e -= var_prod[i] * beta_hat[i]

                for j in range(i + 1, ld_bound[1, i]):
                    ld_prod += Di[j - ld_bound[0, i]] * var_prod[i] * var_prod[j]

        sig_e += ld_prod
        self.ld_prod = 2. * ld_prod

        if 'sigma_epsilon' not in self.fix_params:

            final_sig_e = 1. + 2. * sig_e
            if self.scale_prior:
                final_sig_e *= (self.N / (self.N + var_gamma_sum))

            self.sigma_epsilon = np.clip(final_sig_e, 1e-12, np.inf)

        self.history['sigma_epsilon'].append(self.sigma_epsilon)

    cpdef objective(self):

        loglik = 0.  # log of joint density
        ent = 0.  # entropy

        cdef double prior_var = self.sigma_epsilon * self.sigma_beta if self.scale_prior else self.sigma_beta

        # Add the fixed quantities:

        loglik -= .5 * self.N * (np.log(2 * np.pi * self.sigma_epsilon) + 1. / self.sigma_epsilon)
        loglik -= .5 * self.M * np.log(2. * np.pi * prior_var)
        loglik += self.M * np.log(1. - self.pi)

        ent += .5 * self.M * np.log(2. * np.pi * np.e * prior_var)

        for c in self.var_mu_beta:
            beta_hat = self.beta_hat[c]
            gamma_mu = self.var_gamma[c] * self.var_mu_beta[c]
            gamma_mu_sig = self.var_gamma[c] * (self.var_mu_beta[c] ** 2 + self.var_sigma_beta[c])

            loglik += (-.5 * self.N / self.sigma_epsilon) * (
                    - 2. * np.dot(gamma_mu, beta_hat)
                    + self.ld_prod
                    + np.sum(gamma_mu_sig)
            )

            loglik += (-.5 / prior_var) * (
                    np.sum(gamma_mu_sig) +
                    prior_var * np.sum(1 - self.var_gamma[c])
            )

            loglik += np.log(self.pi / (1. - self.pi)) * np.sum(self.var_gamma[c])

            ent += .5 * np.dot(self.var_gamma[c], np.log(self.var_sigma_beta[c] / prior_var))

            ent -= np.dot(self.var_gamma[c], np.log(self.var_gamma[c] / (1. - self.var_gamma[c])))
            ent -= np.sum(np.log(1. - self.var_gamma[c]))

        elbo = loglik + ent

        self.history['ELBO'].append(elbo)
        #self.history['loglikelihood'].append(loglik)
        #self.history['entropy'].append(ent)

        return elbo

    cpdef get_proportion_causal(self):
        return self.pi

    cpdef get_heritability(self):

        sigma_g = np.sum([
            np.sum(self.var_gamma[c] * (self.var_mu_beta[c] ** 2 + self.var_sigma_beta[c]))
            for c in self.var_gamma
        ]) + self.ld_prod

        h2g = sigma_g / (sigma_g + self.sigma_epsilon)

        self.history['heritability'].append(h2g)

        return h2g

    cpdef fit(self, max_iter=500, continued=False, tol=1e-6, max_elbo_drops=10):

        if not continued:
            self.initialize()

        if self.load_ld:
            self.gdl.load_ld()

        elbo_dropped_count = 0
        converged = False

        for i in range(1, max_iter + 1):
            self.e_step()
            self.m_step()
            self.objective()
            self.get_heritability()

            if i > 1:

                if self.history['ELBO'][i-1] < self.history['ELBO'][i-2]:
                    elbo_dropped_count += 1
                    print(f"Warning (Iteration {i}): ELBO dropped from {self.history['ELBO'][i-2]:.6f} to {self.history['ELBO'][i-1]:.6f}.")

                if np.abs(self.history['ELBO'][i-1] - self.history['ELBO'][i-2]) <= tol:
                    print(f"Converged at iteration {i} | ELBO: {self.history['ELBO'][i-1]:.6f}")
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

        if self.load_ld:
            self.gdl.release_ld()

        return self

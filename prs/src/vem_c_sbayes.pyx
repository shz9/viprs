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
from .PRSModel cimport PRSModel
from .c_utils cimport dot


cdef class vem_prs_sbayes(PRSModel):

    cdef public:
        double pi, sigma_beta, sigma_epsilon  # Global parameters
        double ld_prod  # Need to keep track of this quantity
        bint scale_prior
        dict var_mu_beta, var_sigma_beta, var_gamma  # Variational parameters
        dict beta_hat, ld, ld_bounds, yy, sig_e_snp  # Inputs to the algorithm
        dict shapes, history, fix_params  # Helpers

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

        if load_ld:
            self.gdl.load_ld()

        self.ld = self.gdl.get_ld_matrices()
        self.ld_bounds = self.gdl.get_ld_boundaries()
        self.beta_hat = self.gdl.beta_hats
        self.yy = self.gdl.compute_yy_per_snp()
        self.sig_e_snp = {}

        self.scale_prior = scale_prior
        self.shapes = self.gdl.shapes
        self.fix_params = fix_params or {}
        self.fix_params = {k: np.array(v).flatten() for k, v in self.fix_params}

        self.history = {}

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

        if 'sigma_beta' not in self.fix_params:
            self.sigma_beta = 1./ self.M #np.random.uniform()
        else:
            self.sigma_beta = self.fix_params['sigma_beta'][0]

        if 'sigma_epsilon' not in self.fix_params:
            self.sigma_epsilon = 0.8 #np.random.uniform()
        else:
            self.sigma_epsilon = self.fix_params['sigma_epsilon'][0]

        self.sig_e_snp = {c: np.repeat(self.sigma_epsilon, c_size)
                          for c, c_size in self.shapes.items()}

        if 'pi' not in self.fix_params:
            self.pi = 0.1 #np.random.uniform()
        else:
            self.pi = self.fix_params['pi'][0]

    def initialize_variational_params(self):

        self.var_mu_beta = {}
        self.var_sigma_beta = {}
        self.var_gamma = {}

        for c, c_size in self.shapes.items():

            self.var_gamma[c] = np.repeat(0.5, c_size) #np.random.uniform(size=c_size)
            self.var_mu_beta[c] = np.zeros(shape=c_size) #np.random.normal(scale=1./np.sqrt(self.M), size=c_size)
            self.var_sigma_beta[c] = np.repeat(1./self.M, c_size)

    def e_step(self):
        """
        In the E-step, we update the variational parameters
        for each SNP.
        :return:
        """

        cdef:
            unsigned int i
            double u_j
            double[::1] var_prod, var_mu_beta, var_sigma_beta, var_gamma, beta_hat, sig_e, prior_var, Di
            long[:, ::1] ld_bound

        # The log(pi) for the gamma updates
        cdef double log_pi = log(self.pi / (1. - self.pi))

        for c, c_size in self.shapes.items():

            beta_hat = self.beta_hat[c].values
            var_mu_beta = self.var_mu_beta[c]
            var_sigma_beta = self.var_sigma_beta[c]
            var_gamma = self.var_gamma[c]
            ld_bound = self.ld_bounds[c]
            sig_e = self.sig_e_snp[c]
            prior_var = self.sig_e_snp[c]*self.sigma_beta if self.scale_prior else np.repeat(self.sigma_beta, c_size)

            var_prod = np.multiply(var_gamma, var_mu_beta)

            for i, Di in enumerate(self.ld[c].iterate()):

                var_sigma_beta[i] = sig_e[i] / (self.N + sig_e[i] / prior_var[i])

                var_mu_beta[i] = (beta_hat[i] - dot(Di, var_prod[ld_bound[0, i]: ld_bound[1, i]]) +
                                  Di[i - ld_bound[0, i]]*var_prod[i]) / (1. + sig_e[i] / (self.N * prior_var[i]))

                u_i = (log_pi + .5*log(var_sigma_beta[i] / prior_var[i]) +
                       (.5/var_sigma_beta[i])*var_mu_beta[i]*var_mu_beta[i])
                var_gamma[i] = 1./(1. + exp(-u_i))

                var_prod[i] = var_gamma[i]*var_mu_beta[i]

            self.var_gamma[c] = np.clip(var_gamma, 1e-6, 1. - 1e-6)
            self.var_sigma_beta[c] = np.array(var_sigma_beta)
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
            self.pi = var_gamma_sum / self.M
            self.pi = np.clip(self.pi, 1./self.M, 1.)

        self.history['pi'].append(self.pi)

        if 'sigma_beta' not in self.fix_params:
            # Update sigma_beta:
            self.sigma_beta = np.sum([
                np.dot(self.var_gamma[c],
                       self.var_mu_beta[c]**2 + self.var_sigma_beta[c])
                for c in self.var_mu_beta]) / var_gamma_sum

            if self.scale_prior:
                self.sigma_beta /= self.sigma_epsilon

            self.sigma_beta = np.clip(self.sigma_beta, 1e-12, np.inf)

        self.history['sigma_beta'].append(self.sigma_beta)

        # Update sigma_epsilon

        cdef:
            double global_sig_e = 0., ld_prod = 0., snp_res, snp_ld
            unsigned int i, j, c_size
            double[::1] var_prod, var_gamma, var_mu_beta, var_sigma_beta, beta_hat, sig_e, yy, Di
            long[:, ::1] ld_bound
            double scale_prior_adj = (1. + 1./(self.N*self.sigma_beta)) if self.scale_prior else 1.

        for c, c_size in self.shapes.items():

            beta_hat = self.beta_hat[c].values
            var_gamma = self.var_gamma[c]
            var_mu_beta = self.var_mu_beta[c]
            var_sigma_beta = self.var_sigma_beta[c]
            var_prod = np.multiply(var_gamma, var_mu_beta)
            ld_bound = self.ld_bounds[c]
            sig_e = self.sig_e_snp[c]
            yy = self.yy[c].values

            for i, Di in enumerate(self.ld[c].iterate()):
                snp_res = 0.
                snp_ld = 0.

                snp_res += .5 * scale_prior_adj * var_gamma[i] * (var_mu_beta[i]*var_mu_beta[i] + var_sigma_beta[i])
                snp_res -= var_prod[i] * beta_hat[i]

                for j in range(i + 1, ld_bound[1, i]):
                    snp_ld += Di[j - ld_bound[0, i]] * var_prod[i] * var_prod[j]

                sig_e[i] = yy[i] + 2.*(snp_res + snp_ld)
                global_sig_e += snp_res
                ld_prod += snp_ld

            self.sig_e_snp[c] = np.array(sig_e)

        global_sig_e += ld_prod
        self.ld_prod = 2. * ld_prod

        if 'sigma_epsilon' not in self.fix_params:

            final_sig_e = 1. + 2. * global_sig_e
            if self.scale_prior:
                final_sig_e *= (self.N / (self.N + var_gamma_sum))

            self.sigma_epsilon = np.clip(final_sig_e, 1e-12, np.inf)

        self.history['sigma_epsilon'].append(self.sigma_epsilon)

    def objective(self):

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

    def get_heritability(self):

        sigma_g = np.sum([
            np.sum(self.var_gamma[c] * (self.var_mu_beta[c] ** 2 + self.var_sigma_beta[c]))
            for c in self.var_gamma
        ]) + self.ld_prod

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


import dask.array as da
import numpy as np
from prs.ext.prs_models import PRSModel


class vem_prs_ss(PRSModel):

    def __init__(self, gdl, fix_params=None):
        """
        :param gdl: An instance of GWAS data loader
        """

        super().__init__(gdl)

        self.R_zero_diag = {c: (da.from_zarr(ld) - da.diag(da.ones(ld.shape[0]))).compute()
                            for c, ld in self.gdl.get_ld_matrices().items()}
        self.R_triu = {c: np.triu(ld, 1) for c, ld in self.R_zero_diag.items()}

        self.fix_params = fix_params or {}

        # Variational parameters:
        self.var_mu_beta = None
        self.var_sigma_beta = None
        self.var_gamma = None

        self.initialize_variational_params()

        # Global parameters (point estimates):
        self.pi = None
        self.sigma_beta = None
        self.sigma_epsilon = None

        self.history = {}

        self.initialize_theta()
        self.init_history()

    def init_history(self):

        self.history = {
            'ELBO': [],
            'loglikelihood': [],
            'entropy': [],
            'pi': [],
            'sigma_beta': [],
            'sigma_epsilon': [],
            'heritability': []
        }

    def initialize_theta(self):

        if 'sigma_beta' in self.fix_params:
            self.sigma_beta = self.fix_params['sigma_beta']
        else:
            self.sigma_beta = 1./self.M #np.random.uniform()

        if 'sigma_epsilon' in self.fix_params:
            self.sigma_epsilon = self.fix_params['sigma_epsilon']
        else:
            self.sigma_epsilon = 0.8 #np.random.uniform()

        if 'pi' in self.fix_params:
            self.pi = self.fix_params['pi']
        else:
            self.pi = 0.1 #np.random.uniform()

    def initialize_variational_params(self):
        """
        :return:
        """

        self.var_mu_beta = {}
        self.var_sigma_beta = {}
        self.var_gamma = {}

        for c in self.gdl.genotypes:

            if 'var_gamma' in self.fix_params:
                self.var_gamma[c] = np.clip(self.fix_params['var_gamma'][c], 1e-6, 1. - 1e-6)
            else:
                self.var_gamma[c] = np.zeros(
                    shape=self.gdl.genotypes[c]['G'].shape[1]
                )

            self.var_mu_beta[c] = np.zeros(
                shape=self.gdl.genotypes[c]['G'].shape[1]
            )

            self.var_sigma_beta[c] = np.repeat(1./self.M,
                                               self.gdl.genotypes[c]['G'].shape[1])

    def e_step_loop(self):
        """
        In the E-step, we update the variational parameters
        for each SNP.
        :return:
        """

        beta_hat = self.gdl.beta_hats

        for c in self.var_mu_beta:

            # Updates for sigma_beta variational parameters:
            self.var_sigma_beta[c] = np.repeat(
                self.sigma_epsilon / (self.N + self.sigma_epsilon/self.sigma_beta),
                len(self.var_sigma_beta[c])
            )

            # Updates for gamma variational parameters:

            """
            if 'var_gamma' not in self.fix_params:
                u = (
                        np.log(self.pi / (1. - self.pi)) +
                        .5 * np.log(self.var_sigma_beta[c] / self.sigma_beta) +
                        (.5 / self.var_sigma_beta[c]) * (self.var_mu_beta[c] ** 2)
                )

                self.var_gamma[c] = 1. / (1. + np.exp(-u))
                self.var_gamma[c] = np.clip(self.var_gamma[c], 1e-6, 1. - 1e-6)

        
            for j in range(len(self.var_mu_beta[c])):

                if 'var_gamma' not in self.fix_params:
                    # Updates for gamma variational parameters:
                    uj = (
                            np.log(self.pi/(1. - self.pi)) +
                            .5*np.log(self.var_sigma_beta[c][j]/self.sigma_beta) +
                            (.5/self.var_sigma_beta[c][j])*(self.var_mu_beta[c][j]**2)
                    )

                    self.var_gamma[c][j] = np.clip(1./(1. + np.exp(-uj)), 1e-6, 1. - 1e-6)
            """

            for j in range(len(self.var_mu_beta[c])):

                # Updates for mu_beta variational parameters:
                self.var_mu_beta[c][j] = (
                        beta_hat[c][j] -
                        self.R_zero_diag[c][j, :].dot(self.var_gamma[c]*self.var_mu_beta[c])
                    )/(1. + self.sigma_epsilon/(self.N*self.sigma_beta))

                uj = (
                        np.log(self.pi / (1. - self.pi)) +
                        .5 * np.log(self.var_sigma_beta[c][j] / self.sigma_beta) +
                        (.5 / self.var_sigma_beta[c][j]) * (self.var_mu_beta[c][j] ** 2)
                )

                self.var_gamma[c][j] = np.clip(1. / (1. + np.exp(-uj)), 1e-6, 1. - 1e-6)

    def e_step(self):
        """
        In the E-step, we update the variational parameters
        for each SNP.
        :return:
        """

        beta_hat = self.gdl.beta_hats

        for c in self.var_mu_beta:

            # Updates for sigma_beta variational parameters:
            self.var_sigma_beta[c] = np.repeat(
                self.sigma_epsilon / (self.N + self.sigma_epsilon / self.sigma_beta),
                len(self.var_sigma_beta[c])
            )

            # Updates for gamma variational parameters:

            if 'var_gamma' not in self.fix_params:
                u = (
                        np.log(self.pi / (1. - self.pi)) +
                        .5 * np.log(self.var_sigma_beta[c] / self.sigma_beta) +
                        (.5 / self.var_sigma_beta[c]) * (self.var_mu_beta[c] ** 2)
                )

                self.var_gamma[c] = 1. / (1. + np.exp(-u))
                self.var_gamma[c] = np.clip(self.var_gamma[c], 1e-6, 1. - 1e-6)

            # Updates for mu_beta variational parameters:

            self.var_mu_beta[c] = (
                    beta_hat[c] -
                    self.R_zero_diag[c].dot(self.var_gamma[c]*self.var_mu_beta[c])
                )/(1. + self.sigma_epsilon/(self.N*self.sigma_beta))

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
            self.pi = np.clip(self.pi, 1. / self.M, 1.)

        self.history['pi'].append(self.pi)

        # Update sigma_beta:

        if 'sigma_beta' not in self.fix_params:
            self.sigma_beta = np.sum([
                np.dot(self.var_gamma[c],
                       self.var_mu_beta[c]**2 + self.var_sigma_beta[c])
                for c in self.var_mu_beta]) / var_gamma_sum

            self.sigma_beta = np.clip(self.sigma_beta, 1e-12, np.inf)

        self.history['sigma_beta'].append(self.sigma_beta)

        # Update sigma_epsilon

        if 'sigma_epsilon' not in self.fix_params:

            beta_hat = self.gdl.beta_hats

            self.sigma_epsilon = 1. + np.sum([
                        np.dot(self.var_gamma[c], self.var_mu_beta[c]**2 + self.var_sigma_beta[c]) -
                        2. * np.dot(self.var_gamma[c]*self.var_mu_beta[c], beta_hat[c]) +
                        2. * np.dot(self.var_gamma[c]*self.var_mu_beta[c],
                                    self.R_triu[c].dot(self.var_gamma[c]*self.var_mu_beta[c]))
                        for c in self.var_mu_beta
                    ])

            self.sigma_epsilon = np.clip(self.sigma_epsilon, 1e-12, np.inf)

        self.history['sigma_epsilon'].append(self.sigma_epsilon)

    def objective(self):

        loglik = 0.  # log of joint density
        ent = 0.  # entropy

        # Add the fixed quantities:

        loglik -= .5*self.N*(np.log(2*np.pi*self.sigma_epsilon) + 1./self.sigma_epsilon)
        loglik -= .5*self.M*np.log(2.*np.pi*self.sigma_beta)
        loglik += self.M*np.log(1. - self.pi)

        ent += .5*self.M*np.log(2.*np.pi*np.e*self.sigma_beta)

        beta_hats = self.gdl.beta_hats

        for c in self.var_mu_beta:

            beta_hat = beta_hats[c]
            gamma_mu = self.var_gamma[c]*self.var_mu_beta[c]
            gamma_mu_sig = self.var_gamma[c]*(self.var_mu_beta[c]**2 + self.var_sigma_beta[c])

            loglik += (-.5*self.N/self.sigma_epsilon)*(
                    - 2.*np.dot(gamma_mu, beta_hat)
                    + 2.*np.dot(gamma_mu, self.R_triu[c].dot(gamma_mu))
                    + np.sum(gamma_mu_sig)
            )

            loglik += (-.5/self.sigma_beta)*(
                    np.sum(gamma_mu_sig) +
                    self.sigma_beta*np.sum(1 - self.var_gamma[c])
            )

            loglik += np.log(self.pi / (1. - self.pi))*np.sum(self.var_gamma[c])

            ent += .5*np.dot(self.var_gamma[c], np.log(self.var_sigma_beta[c]/self.sigma_beta))

            ent -= np.dot(self.var_gamma[c], np.log(self.var_gamma[c] / (1. - self.var_gamma[c])))
            ent -= np.sum(np.log(1. - self.var_gamma[c]))

        elbo = loglik + ent

        self.history['ELBO'].append(elbo)
        self.history['loglikelihood'].append(loglik)
        self.history['entropy'].append(ent)

        return elbo

    def get_heritability(self):

        tot_sig_beta = np.sum([
            np.sum(self.var_gamma[chrom]*self.sigma_beta)
            for chrom in self.var_gamma])
        tot_sig_beta2 = np.sum([
            np.sum(self.var_gamma[chrom]*self.var_sigma_beta[chrom])
            for chrom in self.var_sigma_beta
        ])
        tot_sig_beta3 = self.sigma_beta*self.M
        tot_sig_beta4 = np.sum([
            np.sum(self.var_sigma_beta[chrom])
            for chrom in self.var_sigma_beta
        ])

        tot_sig_beta5 = np.sum([
            np.sum(self.var_gamma[c]*(self.var_mu_beta[c]**2 + self.var_sigma_beta[c]))
            for c in self.var_gamma
        ])

        tot_sig_beta6 = np.sum([
            np.sum(self.var_gamma[c]*(self.var_mu_beta[c]**2 + self.var_sigma_beta[c])
                   + self.sigma_beta*(1. - self.var_gamma[c]))
            for c in self.var_gamma
        ])

        """
        tot_sig_beta7 = np.sum(
            [
                np.dot(self.var_gamma[c]*self.var_mu_beta[c],
                       self.N*self.gdl.ld[c][0].dot(self.var_gamma[c]*self.var_mu_beta[c]))
                for c in self.var_gamma
            ]
        )
        """

        h2 = tot_sig_beta5 / (tot_sig_beta5 + self.sigma_epsilon)

        self.history['heritability'].append(h2)
        return h2

    def fit(self, max_iter=500, continued=False,
            tol=1e-6, vectorize=False, max_elbo_drops=10):

        if not continued:
            self.initialize_theta()
            self.initialize_variational_params()
            self.init_history()

        elbo_dropped_count = 0
        converged = False
        i = 0

        while not converged:

            if vectorize:
                self.e_step()
            else:
                self.e_step_loop()

            self.m_step()

            self.objective()
            self.get_heritability()

            i += 1

            if len(self.history['ELBO']) > 1:

                if self.history['ELBO'][-1] < self.history['ELBO'][-2]:
                    elbo_dropped_count += 1
                    print(f"Warning (Iteration {i}): ELBO dropped from {self.history['ELBO'][-2]} "
                          f"to {self.history['ELBO'][-1]}!")

                if np.abs(self.history['ELBO'][-1] - self.history['ELBO'][-2]) <= tol:
                    print(f"Converged at iteration {i} | ELBO: {self.history['ELBO'][-1]}")
                    converged = True
                elif i >= max_iter:
                    print("Max iterations reached without convergence. "
                          "You may need to run the model for more iterations.")
                    converged = True
                elif elbo_dropped_count > max_elbo_drops:
                    print("The optimization is halted due to numerical instabilities!")
                    converged = True

        self.pip = self.var_gamma
        self.inf_beta = {c: self.var_gamma[c]*self.var_mu_beta[c]
                         for c, v in self.var_gamma.items()}

        return self


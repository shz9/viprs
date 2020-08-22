import numpy as np
from scipy.sparse import triu


class vem_prs(object):

    def __init__(self, gdl):
        """
        :param gdl: An instance of GWAS data loader
        """

        self.gdl = gdl

        self.N = gdl.N  # GWAS Sampele size
        self.M = gdl.M  # Total number of SNPs

        # Variational parameters:
        self.var_mu_beta = None
        self.var_sigma_beta = None
        self.var_gamma = None

        self.initialize_variational_params()

        # Global parameters (point estimates):

        self.pi = None
        self.sigma_beta = None
        self.sigma_epsilon = None

        self.initialize_theta()

    def initialize_theta(self):

        self.sigma_beta = np.random.uniform()
        self.sigma_epsilon = np.random.uniform()
        self.pi = 0.01  # np.random.uniform()

    def initialize_variational_params(self):

        self.var_mu_beta = {}
        self.var_sigma_beta = {}
        self.var_gamma = {}

        for chrom in self.gdl.genotypes:
            self.var_mu_beta[chrom] = self.gdl.beta_hats[chrom].compute()
            """
                np.random.normal(
                size=self.gdl.beta_hats[chrom].shape
            )
            """
            self.var_sigma_beta[chrom] = np.random.uniform(
                size=self.gdl.beta_hats[chrom].shape
            )
            self.var_gamma[chrom] = np.random.uniform(
                size=self.gdl.beta_hats[chrom].shape
            )

    def e_step(self):
        """
        In the E-step, we update the variational parameters
        for each SNP.
        :return:
        """
        for chrom in self.var_mu_beta:

            # Updates for sigma_beta variational parameters:

            self.var_sigma_beta[chrom] = np.repeat(
                self.sigma_epsilon / (self.N + self.sigma_epsilon/self.sigma_beta),
                len(self.var_sigma_beta[chrom])
            )

            # Updates for mu_beta variational parameters:

            self.var_mu_beta[chrom] = (
                    self.gdl.beta_hats[chrom].compute() -
                    self.gdl.ld[chrom][0].dot(self.var_gamma[chrom] * self.var_mu_beta[chrom]) +
                    self.var_gamma[chrom] * self.var_mu_beta[chrom]
                )/(1. + self.sigma_epsilon/(self.N*self.sigma_beta))

            # Updates for gamma variational parameters:

            u = (
                    np.log(self.pi/(1. - self.pi)) +
                    .5*np.log(self.var_sigma_beta[chrom]/self.sigma_beta) +
                    (.5/(self.var_sigma_beta[chrom]))*self.var_mu_beta[chrom]**2
            )

            self.var_gamma[chrom] = 1./(1. + np.exp(-u))

    def m_step(self):
        """
        In the M-step, we update the global parameters of
        the model.
        :return:
        """

        # Update pi:

        self.pi = np.clip(np.sum([
            np.sum(self.var_gamma[chrom])
            for chrom in self.var_gamma
        ]) / self.M, 1./self.M, 1.)

        # Update sigma_beta:

        self.sigma_beta = np.clip(np.sum([
            np.dot(self.var_gamma[chrom],
                   self.var_mu_beta[chrom]**2 + self.var_sigma_beta[chrom]**2)
            for chrom in self.var_mu_beta]) / (self.M*self.pi),
                                  1e-10, np.inf)

        # Update sigma_epsilon

        self.sigma_epsilon = np.clip((
                1. + np.sum([
                    np.sum(self.var_gamma[chrom]*self.var_mu_beta[chrom]**2) -
                    np.dot(self.var_gamma[chrom]*self.var_mu_beta[chrom], self.gdl.beta_hats[chrom].compute()) +
                    2. * np.dot(self.var_mu_beta[chrom].T, triu(self.gdl.ld[chrom][0], 1).dot(self.var_mu_beta[chrom]))
                    for chrom in self.var_mu_beta
                ])
        ), 1e-10, np.inf)

    def objective(self):
        pass

    def get_h2g(self):
        return self.sigma_beta / (self.sigma_beta + self.sigma_epsilon)

    def iterate(self, max_iter=50):

        for i in range(max_iter):
            print(f'Iteration {i}')
            print("----------------")
            print("Hyperparameters:")
            print("Heritability:", self.get_h2g())
            print("sigma beta:", self.sigma_beta)
            print("sigma_epsilon:", self.sigma_epsilon)
            print("pi", self.pi)
            print("----------------")
            print("Variational parameters:")
            print("sigma_beta_j:", self.var_sigma_beta)
            print("mu_beta_j:", self.var_mu_beta)
            print("gamma_beta_j:", self.var_gamma)

            self.e_step()
            self.m_step()

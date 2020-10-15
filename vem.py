from tqdm import tqdm
import dask.array as da
import numpy as np
from scipy.sparse import triu
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score


class vem_prs_ss(object):

    def __init__(self, gdl):
        """
        :param gdl: An instance of GWAS data loader
        """

        self.gdl = gdl

        self.N = gdl.N  # GWAS Sample size
        self.M = gdl.M  # Total number of SNPs

        self.R_triu = {c: triu(ld[0], 1) for c, ld in self.gdl.ld.items()}
        self.R_diag = {c: ld[0].diagonal() - self.gdl.lam for c, ld in self.gdl.ld.items()}
        self.beta_hat = {c: b.compute() for c, b in self.gdl.beta_hats.items()}

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

    def init_history(self):

        self.history = {
            'ELBO': [],
            'pi': [],
            'sigma_beta': [],
            'sigma_epsilon': [],
            'heritability': []
        }

    def initialize_theta(self):

        self.sigma_beta = np.random.uniform()
        self.sigma_epsilon = np.random.uniform()
        self.pi = np.random.uniform()

    def initialize_variational_params(self):

        self.var_mu_beta = {}
        self.var_sigma_beta = {}
        self.var_gamma = {}

        for c in self.gdl.genotypes:
            self.var_mu_beta[c] = np.random.normal(size=self.gdl.beta_hats[c].shape) #self.beta_hat[c]

            self.var_sigma_beta[c] = np.random.uniform(
                size=self.gdl.beta_hats[c].shape
            )
            self.var_gamma[c] = np.random.uniform(
                size=self.gdl.beta_hats[c].shape
            )

    def e_step(self):
        """
        In the E-step, we update the variational parameters
        for each SNP.
        :return:
        """

        for c in self.var_mu_beta:

            # Updates for sigma_beta variational parameters:
            self.var_sigma_beta[c] = np.repeat(
                self.sigma_epsilon / (self.N + self.sigma_epsilon/self.sigma_beta),
                len(self.var_sigma_beta[c])
            )

            # Updates for mu_beta variational parameters:

            self.var_mu_beta[c] = (
                    self.beta_hat[c] -
                    self.R_triu[c].dot(self.var_gamma[c]*self.var_mu_beta[c])
                )/(1. + self.sigma_epsilon/(self.N*self.sigma_beta))

            # Updates for gamma variational parameters:

            u = (
                    np.log(self.pi/(1. - self.pi)) +
                    .5*np.log(self.var_sigma_beta[c]/self.sigma_beta) +
                    (.5/self.var_sigma_beta[c])*self.var_mu_beta[c]**2
            )

            self.var_gamma[c] = 1./(1. + np.exp(-u))
            self.var_gamma[c] = np.clip(self.var_gamma[c], 1e-6, 1. - 1e-6)

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

        self.pi = var_gamma_sum / self.M

        self.pi = np.clip(self.pi, 1. / self.M, 1.)

        self.history['pi'].append(self.pi)

        # Update sigma_beta:

        self.sigma_beta = np.sum([
            np.dot(self.var_gamma[c],
                   self.var_mu_beta[c]**2 + self.var_sigma_beta[c])
            for c in self.var_mu_beta]) / var_gamma_sum

        self.sigma_beta = np.clip(self.sigma_beta, 1e-12, np.inf)
        self.history['sigma_beta'].append(self.sigma_beta)

        # Update sigma_epsilon

        self.sigma_epsilon = 1. + np.sum([
                    np.dot(self.var_gamma[c], self.var_mu_beta[c]**2 + self.var_sigma_beta[c]) -
                    np.dot(self.var_gamma[c]*self.var_mu_beta[c], self.beta_hat[c]) +
                    2. * np.dot(self.var_gamma[c]*self.var_mu_beta[c],
                                self.R_triu[c].dot(self.var_gamma[c]*self.var_mu_beta[c]))
                    for c in self.var_mu_beta
                ])

        self.sigma_epsilon = np.clip(self.sigma_epsilon, 1e-12, np.inf)
        self.history['sigma_epsilon'].append(self.sigma_epsilon)

    def objective(self):

        f = 0.

        for c in self.var_mu_beta:

            beta_hat = self.beta_hat[c]
            gamma_mu = self.var_gamma[c]*self.var_mu_beta[c]
            gamma_mu_sig = self.var_gamma[c]*(self.var_mu_beta[c]**2 + self.var_sigma_beta[c])

            f += (-.5*self.N/self.sigma_epsilon)*(
                    self.sigma_epsilon*np.log(2.*np.pi*self.sigma_epsilon) + 1.
                    - 2.*np.dot(gamma_mu, beta_hat)
                    + 2.*np.dot(gamma_mu, self.R_triu[c].dot(gamma_mu))
                    + np.dot(self.R_diag[c], gamma_mu_sig)
            )

            f += (-.5/self.sigma_beta)*(
                    np.sum(gamma_mu_sig) +
                    self.sigma_beta*np.sum(1 - self.var_gamma[c])
            )

            f += self.M*np.log(1 - self.pi) + np.log(self.pi / (1. - self.pi))*np.sum(self.var_gamma[c]) + self.M/2

            f += .5*np.dot(self.var_gamma[c], np.log(self.var_sigma_beta[c]/self.sigma_beta))

            f -= np.dot(self.var_gamma[c], np.log(self.var_gamma[c] / (1. - self.var_gamma[c])))
            f -= np.sum(np.log(1. - self.var_gamma[c]))

        self.history['ELBO'].append(f)

        return f

    def get_h2g(self):

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

        self.history['heritability'].append((tot_sig_beta / (tot_sig_beta + self.sigma_epsilon),
                tot_sig_beta2 / (tot_sig_beta2 + self.sigma_epsilon),
                tot_sig_beta3 / (tot_sig_beta3 + self.sigma_epsilon),
                tot_sig_beta4 / (tot_sig_beta4 + self.sigma_epsilon),
                tot_sig_beta5 / (tot_sig_beta5 + self.sigma_epsilon),
                tot_sig_beta6 / (tot_sig_beta6 + self.sigma_epsilon)))

        return (tot_sig_beta / (tot_sig_beta + self.sigma_epsilon),
                tot_sig_beta2 / (tot_sig_beta2 + self.sigma_epsilon),
                tot_sig_beta3 / (tot_sig_beta3 + self.sigma_epsilon),
                tot_sig_beta4 / (tot_sig_beta4 + self.sigma_epsilon),
                tot_sig_beta5 / (tot_sig_beta5 + self.sigma_epsilon),
                tot_sig_beta6 / (tot_sig_beta6 + self.sigma_epsilon))

    def iterate(self, max_iter=1000, continued=False, tol=1e-6):

        if not continued:
            self.initialize_theta()
            self.initialize_variational_params()
            self.init_history()

        converged = False
        i = 0

        while not converged:

            self.e_step()
            self.m_step()
            self.objective()
            self.get_h2g()

            i += 1

            if len(self.history['ELBO']) > 1:
                if np.abs(self.history['ELBO'][-1] - self.history['ELBO'][-2]) <= tol:
                    print(f"Converged at iteration {i} | ELBO: {self.history['ELBO'][-1]}")
                    converged = True
                elif i > max_iter:
                    print("Max iterations reached without convergence. You may need more itreations.")
                    converged = True

    def predict_phenotype(self, beta, subset_idx=None):

        if subset_idx is None:
            g_comp = da.zeros(shape=self.N)
        else:
            g_comp = da.zeros(shape=len(subset_idx))

        for c in self.gdl.genotypes:
            if subset_idx is None:
                g_comp += da.dot(self.gdl.genotypes[c]['G'], beta[c])
            else:
                g_comp += da.dot(self.gdl.genotypes[c]['G'][subset_idx, :], beta[c])

        return g_comp

    def k_fold_cross_validation(self, k=5):

        ind_idx = np.arange(len(self.gdl.phenotypes))
        np.random.shuffle(ind_idx)

        kf = KFold(n_splits=k)

        r2 = {
            'Marginal Beta': [],
            'Posterior Beta': []
        }

        for i, (train, test) in enumerate(kf.split(ind_idx)):
            print(f"Cross validation iteration {i}")
            self.N = len(train)
            self.beta_hat = {c: (da.dot(gt['G'][train, :].T, self.gdl.phenotypes[train]) / self.N).compute()
                             for c, gt in self.gdl.genotypes.items()}
            self.iterate()

            post_pred_phenotypes = self.predict_phenotype({c: self.var_gamma[c]*self.var_mu_beta[c]
                                                           for c in self.var_gamma}, test)
            pred_phenotypes = self.predict_phenotype(self.beta_hat, test)

            r2['Marginal Beta'].append(r2_score(self.gdl.phenotypes[test], pred_phenotypes))
            r2['Posterior Beta'].append(r2_score(self.gdl.phenotypes[test], post_pred_phenotypes))

        self.N = self.gdl.N
        self.beta_hat = {c: b.compute() for c, b in self.gdl.beta_hats.items()}

        return r2

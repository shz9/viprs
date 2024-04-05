from .BayesPRSModel import BayesPRSModel


class LDPredInf(BayesPRSModel):
    """
    A wrapper class implementing the LDPred-inf model.
    The LDPred-inf model is a Bayesian model that uses summary statistics
    from GWAS to estimate the posterior mean effect sizes of the SNPs. It is equivalent
    to performing ridge regression, with the penalty proportional to the inverse of
    the per-SNP heritability.

    Refer to the following references for details about the LDPred-inf model:
    * Vilhjálmsson et al. AJHG. 2015
    * Privé et al. Bioinformatics. 2020

    :ivar gdl: An instance of `GWADataLoader`
    :ivar h2: The heritability for the trait (can also be chromosome-specific)

    """

    def __init__(self,
                 gdl,
                 h2=None):
        """
        Initialize the LDPred-inf model.
        :param gdl: An instance of GWADataLoader
        :param h2: The heritability for the trait (can also be chromosome-specific)
        """
        super().__init__(gdl)

        if h2 is None:
            from magenpy.stats.h2.ldsc import simple_ldsc
            self.h2 = simple_ldsc(self.gdl)
        else:
            self.h2 = h2

    def get_heritability(self):
        """
        :return: The heritability estimate for the trait of interest.
        """
        return self.h2

    def fit(self, solver='minres', **solver_kwargs):
        """
        Fit the summary statistics-based ridge regression,
        following the specifications of the LDPred-inf model.

        !!! warning
            Not tested yet.

        Here, we use `lsqr` or `minres` solvers to solve the system of equations:

        (D + lam*I)BETA = BETA_HAT

        where D is the LD matrix, BETA is ridge regression
        estimate that we wish to obtain and BETA_HAT is the
        marginal effect sizes estimated from GWAS.

        In this case, lam = M / N*h2, where M is the number of SNPs,
        N is the number of samples and h2 is the heritability
        of the trait.

        https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsqr.html
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.minres.html

        :param solver: The solver for the system of linear equations. Options: `minres` or `lsqr`
        :param solver_kwargs: keyword arguments for the solver.
        """

        assert solver in ('lsqr', 'minres')

        import numpy as np
        from scipy.sparse.linalg import lsqr, minres
        from scipy.sparse import identity, block_diag

        if solver == 'lsqr':
            solve = lsqr
        else:
            solve = minres

        # Lambda, the regularization parameter for the
        # ridge regression estimator. For LDPred-inf model,
        # we set this to 'M / N*h2', where M is the number of SNPs,
        # N is the number of samples and h2 is the heritability
        # of the trait.
        lam = self.n_snps / (self.n * self.h2)

        chroms = self.gdl.chromosomes

        # Extract the LD matrices for all the chromosomes represented and
        # concatenate them into one block diagonal matrix:
        ld_mats = []
        for c in chroms:
            self.gdl.ld[c].load(dtype=np.float32)
            ld_mats.append(self.gdl.ld[c].csr_matrix)

        ld = block_diag(ld_mats, format='csr')

        # Extract the marginal GWAS effect sizes:
        marginal_beta = np.concatenate([self.gdl.sumstats_table[c].marginal_beta
                                        for c in chroms])

        # Estimate the BETAs under the ridge penalty:
        res = solve(ld + lam * identity(ld.shape[0]), marginal_beta, **solver_kwargs)

        # Extract the estimates and populate them in `post_mean_beta`
        start = 0
        self.post_mean_beta = {}

        for c in chroms:
            self.post_mean_beta[c] = res[0][start:start + self.shapes[c]]
            start += self.shapes[c]

        return self

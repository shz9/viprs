from .PRSModel import PRSModel


class LDPredinf(PRSModel):
    """
    Implementation for the summary statistics-based
    ridge-regression estimator, also known as LDPred-inf.

    See also:
    - Vilhjálmsson et al. AJHG. 2015
    - Privé et al. Bioinformatics. 2020

    """

    def __init__(self, gdl, h2=None, verbose=True, threads=1):
        """
        :param gdl: An instance of GWADataLoader
        :param h2: The heritability for the trait (can also be chromosome-specific)
        :param verbose: Verbosity of the information printed to standard output
        :param threads: The number of threads to use (experimental)
        """
        super().__init__(gdl)

        if h2 is None:
            from magenpy.stats.h2.ldsc import simple_ldsc
            self.h2 = simple_ldsc(self.gdl)
        else:
            self.h2 = h2

    def get_heritability(self):
        return self.h2

    def fit(self, solver='minres', **solver_kwargs):
        """
        Fit the summary statistics-based ridge regression,
        following the specifications of the LDPred-inf model.

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
        # N is the number of sampels and h2 is the heritability
        # of the trait.
        lam = self.n_snps / (self.N * self.h2)

        chroms = self.gdl.chromosomes

        # Extract the LD matrices for all the chromosomes represented and
        # concatenate them into one block diagonal matrix:
        ld = block_diag([self.gdl.ld[c].to_csr_matrix() for c in chroms],
                         format='csr')

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

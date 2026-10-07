from collections.abc import Mapping

import numpy as np

from .BayesPRSModel import BayesPRSModel


class LDPredInf(BayesPRSModel):
    """
    Implementation of the infinitesimal LDpred model.

    LDpred-inf estimates posterior mean effect sizes by solving the system:

    ``(D + LAMBDA) BETA = BETA_HAT``

    where ``D`` is the LD matrix and ``BETA_HAT`` contains standardized
    marginal effect sizes. ``LAMBDA`` is a diagonal ridge penalty determined by
    heritability and the allele-frequency parameter ``alpha``. The system is
    symmetric and is solved with MINRES.

    :ivar h2: Genome-wide heritability, or a dictionary of chromosome-specific
        heritability estimates.
    :ivar alpha: The exponent controlling the dependence of standardized effect
        size variance on allele frequency.
    :ivar penalty: A dictionary containing the scalar or per-variant ridge
        penalty used for each chromosome.
    """

    def __init__(self,
                 gdl,
                 h2=None,
                 alpha=0.):
        """
        Initialize the LDpred-inf model.

        If ``h2`` is not provided, it is estimated from the GWAS summary
        statistics using a simplified LD score regression estimator.

        :param gdl: An instance of `GWADataLoader` containing harmonized GWAS
            summary statistics and LD matrices.
        :param h2: Genome-wide heritability, or a dictionary mapping chromosomes
            to chromosome-specific heritability estimates.
        :param alpha: The exponent in the prior variance model
            ``Var(BETA_j) proportional to [2 * MAF_j * (1 - MAF_j)] ** alpha``.
            The default of zero recovers the standard LDpred-inf model.
        """

        super().__init__(gdl)

        # Estimate heritability when it is not supplied by the user:
        if h2 is None:
            from magenpy.stats.h2.ldsc import simple_ldsc
            h2 = simple_ldsc(gdl)

        self.h2 = h2
        self.alpha = alpha
        self.penalty = None

        # Invalid prior parameters give undefined ridge penalties:
        self._validate_prior_parameters()

    def _validate_prior_parameters(self):
        """
        Validate the heritability estimates and allele-frequency exponent.

        LDpred-inf requires every heritability estimate to be finite and
        strictly positive. For chromosome-specific estimates, this method also
        checks that an estimate is available for every chromosome in the model.
        The allele-frequency exponent may take any finite value.

        :raises ValueError: If an estimate is invalid or a chromosome is missing.
        """

        if isinstance(self.h2, Mapping):
            h2_values = self.h2.values()
        else:
            h2_values = (self.h2,)

        if any(not np.isfinite(h2) or h2 <= 0 for h2 in h2_values):
            raise ValueError("Heritability must contain only finite, positive values.")

        if isinstance(self.h2, Mapping):
            missing_chromosomes = set(self.chromosomes).difference(self.h2)
            if missing_chromosomes:
                raise ValueError(
                    f"Missing heritability for chromosomes: {sorted(missing_chromosomes)}"
                )

        if not np.isscalar(self.alpha) or not np.isfinite(self.alpha):
            raise ValueError("Alpha must be a finite scalar.")

    def get_heritability(self):
        """
        Return the heritability used by the model.

        :return: The genome-wide or chromosome-specific heritability estimates.
        """

        return self.h2

    def get_heterozygosity(self, chromosome):
        """
        Return the per-variant heterozygosity for a chromosome.

        Summary-statistic allele frequencies are preferred because they
        describe the GWAS population. If they are unavailable, allele
        frequencies stored with the LD reference panel are used instead.

        :param chromosome: The chromosome for which to retrieve heterozygosity.

        :return: An array containing ``2 * MAF * (1 - MAF)`` for each variant.
        :raises ValueError: If allele frequencies are unavailable or invalid.
        """

        maf = self.gdl.sumstats_table[chromosome].maf
        if maf is None:
            maf = self.gdl.ld[chromosome].maf

        if maf is None:
            raise ValueError(
                "Allele frequencies are required when alpha is non-zero."
            )

        maf = np.asarray(maf, dtype=self.float_precision)
        if (
            maf.shape != (self.shapes[chromosome],)
            or np.any(~np.isfinite(maf))
            or np.any((maf <= 0.) | (maf >= 1.))
        ):
            raise ValueError(
                f"Invalid allele frequencies for chromosome {chromosome}."
            )

        return 2. * maf * (1. - maf)

    def get_penalties(self, penalty_factor=1.0):
        """
        Compute the LDpred-inf ridge penalty for each chromosome.

        With ``alpha = 0``, the standard scalar ``M / (N * h2)`` penalty is
        recovered. Otherwise, standardized effect-size prior variances are
        proportional to heterozygosity raised to ``alpha`` and normalized to
        sum to the supplied heritability. The inverse prior variances determine
        the per-variant ridge penalties.

        :param penalty_factor: A positive multiplier for the theoretical penalty.

        :return: A dictionary mapping chromosomes to ridge penalties.
        :raises ValueError: If ``penalty_factor`` is not finite and positive.
        """

        if not np.isfinite(penalty_factor) or penalty_factor <= 0:
            raise ValueError("The penalty factor must be finite and positive.")

        # Preserve the original scalar penalties when allele frequency does not
        # affect the prior variance. This path does not require MAF information:
        if self.alpha == 0.:
            if isinstance(self.h2, Mapping):
                return {
                    c: penalty_factor * self.shapes[c] / (
                        np.max(self.n_per_snp[c]) * self.h2[c]
                    )
                    for c in self.chromosomes
                }

            sample_size = max(np.max(self.n_per_snp[c]) for c in self.chromosomes)
            penalty = penalty_factor * self.n_snps / (sample_size * self.h2)

            return dict.fromkeys(self.chromosomes, penalty)

        # Compute the relative prior effect-size variance for every variant:
        variance_weights = {
            c: self.get_heterozygosity(c) ** self.alpha
            for c in self.chromosomes
        }

        invalid_weights = any(
            np.any(~np.isfinite(weights))
            for weights in variance_weights.values()
        )
        if invalid_weights:
            raise ValueError("Alpha produced non-finite effect-size variance weights.")

        # Normalize locally when chromosome-specific heritability is supplied:
        if isinstance(self.h2, Mapping):
            return {
                c: penalty_factor * np.sum(variance_weights[c]) / (
                    np.max(self.n_per_snp[c]) * self.h2[c] * variance_weights[c]
                )
                for c in self.chromosomes
            }

        # For genome-wide heritability, normalize the weights across chromosomes:
        sample_size = max(np.max(self.n_per_snp[c]) for c in self.chromosomes)
        total_weight = sum(np.sum(weights) for weights in variance_weights.values())

        return {
            c: penalty_factor * total_weight / (
                sample_size * self.h2 * variance_weights[c]
            )
            for c in self.chromosomes
        }

    def _solve(self,
               penalty_factor=1.0,
               **solver_kwargs):
        """
        Solve the LDpred-inf system independently for every chromosome.

        Chromosomes are independent blocks in the genome-wide LD matrix. Solving
        them separately avoids constructing a redundant block-diagonal matrix
        and permits chromosome-specific penalties.

        :param penalty_factor: A positive multiplier for the theoretical penalty.
        :param solver_kwargs: Keyword arguments passed to `scipy.sparse.linalg.minres`.

        :return: The fitted model.
        :raises RuntimeError: If MINRES does not converge for a chromosome.
        """

        from scipy.sparse import diags, identity
        from scipy.sparse.linalg import minres

        # Compute and retain the penalties for inspection after model fitting:
        self.penalty = self.get_penalties(penalty_factor)
        self.post_mean_beta = {}

        for c in self.chromosomes:

            # Load a symmetric CSR representation of the chromosome LD matrix:
            ld_mat = self.gdl.ld[c].load(dtype=self.float_precision).to_csr()

            # Add either the standard scalar penalty or the alpha-dependent
            # per-variant penalty to the diagonal:
            if np.isscalar(self.penalty[c]):
                penalty_mat = self.penalty[c] * identity(
                    ld_mat.shape[0],
                    format='csr',
                    dtype=ld_mat.dtype
                )
            else:
                penalty_mat = diags(
                    self.penalty[c],
                    format='csr',
                    dtype=ld_mat.dtype
                )

            system = ld_mat + penalty_mat

            # Solve against the standardized marginal effect sizes:
            post_mean_beta, solver_info = minres(
                system,
                self.std_beta[c],
                **solver_kwargs
            )

            if solver_info != 0:
                raise RuntimeError(
                    f"MINRES failed for chromosome {c} (info={solver_info})."
                )

            self.post_mean_beta[c] = post_mean_beta.astype(
                self.float_precision,
                copy=False
            )

        return self

    def fit(self,
            penalty_factor=1.0,
            **solver_kwargs):
        """
        Fit the LDpred-inf model.

        :param penalty_factor: A positive multiplier for the theoretical
            ``M / (N * h2)`` penalty.
        :param solver_kwargs: Keyword arguments passed to `scipy.sparse.linalg.minres`.

        :return: The fitted model.
        """

        return self._solve(penalty_factor, **solver_kwargs)

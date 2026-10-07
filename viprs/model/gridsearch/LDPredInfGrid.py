import numpy as np
import pandas as pd

from ..LDPredInf import LDPredInf


class LDPredInfGrid(LDPredInf):
    """
    LDpred-inf with penalty selection by summary-statistic splitting.

    The model searches over combinations of the allele-frequency exponent and
    multiplicative factors for the theoretical LDpred-inf penalty. Each
    candidate is fitted to a PUMAS training split and evaluated against the
    held-out summary statistics using pseudo-validation R-squared. The selected
    model is then fitted to the full summary statistics.

    :ivar penalty_factors: The candidate multipliers for the LDpred-inf penalty.
    :ivar selected_penalty_factor: The multiplier selected by pseudo-validation.
    :ivar alpha_grid: The candidate allele-frequency exponents.
    :ivar selected_alpha: The exponent selected by pseudo-validation.
    :ivar validation_result: A table containing the score for every candidate.
    """

    def __init__(self,
                 gdl,
                 h2=None,
                 penalty_factors=None,
                 alpha=0.,
                 alpha_grid=None):
        """
        Initialize an LDpred-inf penalty grid search.

        :param gdl: An instance of `GWADataLoader` containing harmonized GWAS
            summary statistics and LD matrices.
        :param h2: Genome-wide heritability, or a dictionary mapping chromosomes
            to chromosome-specific heritability estimates.
        :param penalty_factors: Positive multipliers for the theoretical penalty.
            By default, nine logarithmically spaced values from ``0.01`` to
            ``100`` are evaluated.
        :param alpha: The allele-frequency exponent used when ``alpha_grid`` is
            not provided. The default of zero recovers standard LDpred-inf.
        :param alpha_grid: Optional allele-frequency exponents to include in the
            grid search.
        """

        super().__init__(gdl, h2=h2, alpha=alpha)

        # Use a broad grid around the theoretical LDpred-inf penalty by default:
        if penalty_factors is None:
            penalty_factors = np.logspace(-2, 2, 9)

        self.penalty_factors = np.asarray(penalty_factors, dtype=float)

        if alpha_grid is None:
            alpha_grid = [alpha]

        self.alpha_grid = np.asarray(alpha_grid, dtype=float)

        # At least two valid candidates are needed to perform model selection:
        if (
            self.penalty_factors.ndim != 1
            or len(self.penalty_factors) < 2
            or np.any(~np.isfinite(self.penalty_factors))
            or np.any(self.penalty_factors <= 0)
        ):
            raise ValueError(
                "penalty_factors must contain at least two finite, positive values."
            )

        if (
            self.alpha_grid.ndim != 1
            or len(self.alpha_grid) < 1
            or np.any(~np.isfinite(self.alpha_grid))
        ):
            raise ValueError("alpha_grid must contain at least one finite value.")

        self.selected_penalty_factor = None
        self.selected_alpha = None
        self.validation_result = None

    def to_validation_table(self):
        """
        Return the results of the penalty grid search.

        :return: A copy of the validation table containing alpha values,
            penalty factors, pseudo-validation scores, and the selected model
            indicator.
        :raises ValueError: If the model has not been fitted.
        """

        if self.validation_result is None:
            raise ValueError("Validation result is not set. Call `.fit()` first.")

        return self.validation_result.copy()

    def fit(self,
            prop_train=0.8,
            seed=None,
            refit=True,
            **solver_kwargs):
        """
        Select the penalty factor and fit the LDpred-inf model.

        This method first creates training and validation summary statistics,
        fits every combination of ``alpha`` and penalty factor to the training
        split, and selects the candidate with the highest finite
        pseudo-validation R-squared. By default, the selected model is finally
        refitted to the full GWAS.

        :param prop_train: The proportion of the GWAS sample assigned to the
            training summary statistics.
        :param seed: The random seed for summary-statistic splitting.
        :param refit: If True, refit the selected model to the full summary statistics.
        :param solver_kwargs: Keyword arguments passed to `scipy.sparse.linalg.minres`.

        :return: The fitted model using the selected penalty factor.
        :raises RuntimeError: If pseudo-validation produces no finite scores.
        """

        # Always begin with the original summary statistics. This prevents
        # repeated calls to `fit` from repeatedly reducing the sample size:
        self.initialize_input_data_arrays()
        self.split_gwas_sumstats(prop_train=prop_train, seed=seed)

        # Fit and evaluate every alpha and penalty-factor combination:
        candidate_alphas = []
        candidate_penalty_factors = []
        validation_scores = []
        for alpha in self.alpha_grid:
            self.alpha = alpha
            for penalty_factor in self.penalty_factors:
                self._solve(penalty_factor, **solver_kwargs)
                candidate_alphas.append(alpha)
                candidate_penalty_factors.append(penalty_factor)
                validation_scores.append(self.pseudo_validate())

        candidate_alphas = np.asarray(candidate_alphas)
        candidate_penalty_factors = np.asarray(candidate_penalty_factors)
        validation_scores = np.asarray(validation_scores, dtype=float)
        finite_scores = np.isfinite(validation_scores)

        if not np.any(finite_scores):
            raise RuntimeError("Pseudo-validation produced no finite scores.")

        # Exclude invalid scores and select the best remaining candidate:
        best_model_idx = np.argmax(
            np.where(finite_scores, validation_scores, -np.inf)
        )
        self.selected_alpha = candidate_alphas[best_model_idx]
        self.selected_penalty_factor = candidate_penalty_factors[best_model_idx]
        self.alpha = self.selected_alpha

        # Retain the complete search result for inspection and serialization:
        self.validation_result = pd.DataFrame({
            'Alpha': candidate_alphas,
            'Penalty_factor': candidate_penalty_factors,
            'Pseudo_Validation_R2': validation_scores,
            'Selected': np.arange(len(validation_scores)) == best_model_idx,
        })

        # Refit on all available summary statistics after model selection:
        if refit:
            self.initialize_input_data_arrays()

        return self._solve(self.selected_penalty_factor, **solver_kwargs)

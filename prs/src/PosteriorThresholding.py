import numpy as np
import pandas as pd
import multiprocessing
from .evaluation import compute_r2, compute_auc
from .PRSModel import PRSModel


class PosteriorThresholding(PRSModel):

    def __init__(self, gdl, n_thresholds=10, threshold_type='absolute', n_jobs=1):
        super().__init__(gdl)

        self.n_jobs = n_jobs
        self.thresholds = None
        self.validation_result = None

        self.n_thresholds = n_thresholds
        self.threshold_type = threshold_type

        assert self.threshold_type in ('absolute', 'quantile')
        assert self.gdl.phenotypes is not None

    def set_quantile_thresholds(self, quantiles=None):

        if self.pip is None:
            raise ValueError("The PIPs are not set.")

        if quantiles is None:
            quantiles = 1. - 2.**(-np.linspace(np.floor(np.log10(self.M)), 0., self.n_thresholds))

        self.thresholds = np.quantile(np.concatenate(list(self.pip.values())), quantiles)

    def set_absolute_thresholds(self, values=None):
        if values is None:
            values = np.arange(0., self.n_thresholds) / self.n_thresholds

        self.thresholds = values

    def write_validation_result(self, v_filename):

        if self.validation_result is None:
            raise Exception("Validation result is not set!")
        elif len(self.validation_result) < 1:
            raise Exception("Validation result is not set!")

        v_df = pd.DataFrame(self.validation_result)
        v_df.to_csv(v_filename, index=False, sep="\t")

    def fit(self):

        if self.pip is None or self.inf_beta is None:
            raise ValueError("PIPs or inferred betas are not set!")

        if self.thresholds is None:
            if self.threshold_type == 'quantile':
                self.set_quantile_thresholds()
            else:
                self.set_absolute_thresholds()

        # Check that one of the thresholds does not change the inferred effect sizes:
        assert 0. in self.thresholds and len(self.thresholds) > 1

        th_eff = {}

        for c, pip in self.pip.items():
            th_eff[c] = np.stack([self.inf_beta[c]*(pip >= th).astype(np.float64)
                                  for th in self.thresholds], axis=1)

        prs = self.gdl.predict(th_eff)

        if self.gdl.phenotype_likelihood == 'binomial':
            eval_func = compute_auc
        else:
            eval_func = compute_r2

        ctx = multiprocessing.get_context("spawn")

        with ctx.Pool(self.n_jobs, maxtasksperchild=1) as pool:
            validation_results = pool.starmap(eval_func,
                                              [(prs[:, i].flatten(), self.gdl.phenotypes)
                                               for i in range(len(self.thresholds))])

        self.validation_result = [{'PIP_Threshold': pip_th,
                                   f'Validation_score': vr} for pip_th, vr in
                                  zip(self.thresholds, validation_results)]

        best_prs_idx = np.argmax(validation_results)

        print("> Found the best PIP threshold to be: ", self.thresholds[best_prs_idx])

        self.inf_beta = {c: eff[:, best_prs_idx] for c, eff in th_eff.items()}
        self.pip = {c: pip*(pip >= self.thresholds[best_prs_idx]).astype(np.float64) for c, pip in self.pip.items()}

        return self

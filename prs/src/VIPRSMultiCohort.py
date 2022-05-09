
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from ..magenpy.GWASDataLoader import GWASDataLoader
from .PRSModel import PRSModel


class VIPRSMultiCohort(object):

    def __init__(self, viprs_models):

        assert len(viprs_models) > 1

        self.viprs_models = viprs_models
        self.pi_priors = None

        # Create objects for unified PRS predictor:

        self.unified_gdl = GWASDataLoader.from_table(
            pd.concat([
                prs_m.gdl.to_snp_table(col_subset=('CHR', 'SNP', 'A1', 'A2', 'POS'))
                for prs_m in self.viprs_models
            ]).drop_duplicates().reset_index(drop=True)
        )

        self.unified_gdl.n_per_snp = {
            c: np.repeat(np.mean([prs_m.N for prs_m in self.viprs_models]), c_size)
            for c, c_size in self.unified_gdl.shapes.items()
        }

        self.unified_prs = PRSModel(self.unified_gdl)

    def initialize(self):
        for v in self.viprs_models:
            try:
                del v.fix_params['pi']
            except KeyError:
                continue

    def update_priors(self):
        """
        TODO: Re-write this to use the unified_prs interface
        """

        priors = []
        self.pi_priors = []

        for v in self.viprs_models:
            priors.append(v.to_table(per_chromosome=True))
            self.pi_priors.append({c: None for c in priors[-1]})

        for i in range(len(priors)):
            for c in priors[i]:
                pi_prior = priors[i][c][['SNP', 'PIP']]
                for j in range(len(priors)):
                    if j != i:
                        try:
                            pi_prior = pi_prior.merge(priors[j][c][['SNP', 'PIP']], on='SNP', how='left')
                        except KeyError:
                            continue

                pi_prior = pi_prior.fillna(0.)
                self.pi_priors[i][c] = pi_prior[[col for col in pi_prior.columns
                                                 if col != 'SNP']].mean(axis=1).values

    def fit(self, max_outer_iter=20, outer_x_tol=1e-4, **fit_kwargs):

        self.initialize()

        for i in range(max_outer_iter):

            converged = [False for _ in range(len(self.viprs_models))]

            for idx, v in enumerate(self.viprs_models):
                if self.pi_priors is not None:
                    v.fix_params['pi'] = self.pi_priors[idx]
                v.fit(**fit_kwargs)

                if self.pi_priors is not None:
                    if max([np.abs(v - self.pi_priors[idx][c]).max()
                            for c, v in v.pip.items()]) <= outer_x_tol:
                        converged[idx] = True

            if all(converged):
                print(f"All models converged at iteration {i}")
                break

            self.update_priors()

    def update_unified_predictor(self, method='mean', validation_gdl=None):

        if method == 'linear':
            assert validation_gdl is not None

        unified_tables = self.unified_gdl.to_snp_table(per_chromosome=True,
                                                       col_subset=('SNP', 'A1'))
        cohort_tables = [prs_m.to_table(per_chromosome=True,
                                        col_subset=('SNP', 'A1'))
                         for prs_m in self.viprs_models]

        self.unified_prs.pip = {}
        self.unified_prs.inf_beta = {}

        for c in unified_tables:
            for c_tab in cohort_tables:
                unified_tables[c] = unified_tables[c].merge(c_tab[c], how='left')
            unified_tables[c] = unified_tables[c].fillna(0.)

        # Obtain the unified pip by taking the average of the PIPs:
        self.unified_prs.pip = {
            c: tab[[col for col in tab.columns if 'PIP' in col]].mean(axis=1).values
            for c, tab in unified_tables.items()
        }

        # Obtain the unified inferred beta according to the method specified by the user:
        if method == 'mean':
            # Simply take the average of the posterior effect sizes:
            self.unified_prs.inf_beta = {
                c: tab[[col for col in tab.columns if 'BETA' in col]].mean(axis=1).values
                for c, tab in unified_tables.items()
            }
        elif method == 'linear':
            # Combined the betas according to a linear model trained on a validation set.
            # If we have K PRS predictors (PRS_1, PRS_2, ..., PRS_K)
            # We train a linear model of the form: Y ~ beta_1*PRS_1 + beta_2*PRS_2 + ... + beta_K*PRS_K
            # And we use the weights (betas) to create a unified predictor.
            prs_pred = np.array([m.predict(gdl=validation_gdl) for m in self.viprs_models]).T
            reg = LinearRegression(fit_intercept=False).fit(prs_pred, validation_gdl.phenotypes)

            self.unified_prs.inf_beta = {
                c: (tab[[col for col in tab.columns if 'BETA' in col]].values*reg.coef_).sum(axis=1)
                for c, tab in unified_tables.items()
            }

    def write_inferred_params(self, f_names, unified=True, per_chromosome=False):

        if unified:
            if not isinstance(f_names, str):
                f_names = f_names[0]
            self.unified_prs.write_inferred_params(f_names, per_chromosome=per_chromosome)
        else:
            for prs_m, f_name in zip(self.viprs_models, f_names):
                prs_m.write_inferred_params(f_name, per_chromosome=per_chromosome)

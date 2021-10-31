
import numpy as np


class VIPRSMultiCohort(object):

    def __init__(self, viprs_models):

        assert len(viprs_models) > 1

        self.viprs_models = viprs_models
        self.pi_priors = None
        self.initialize()

    def initialize(self):
        for v in self.viprs_models:
            try:
                del v.fix_params['pi']
            except KeyError:
                continue

    def update_priors(self):

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

        for i in range(max_outer_iter):

            converged = [False for _ in range(len(self.viprs_models))]

            for idx, v in enumerate(self.viprs_models):
                if self.pi_priors is not None:
                    v.fix_params['pi'] = self.pi_priors[idx]
                v.fit(**fit_kwargs)

                if max([np.abs(v - self.pi_priors[idx][c]).max() for c, v in v.pip.items()]) <= outer_x_tol:
                    converged[idx] = True

            if all(converged):
                print(f"All models converged at iteration {i}")
                break

            self.update_priors()

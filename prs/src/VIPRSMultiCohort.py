

class VIPRSMultiCohort(object):

    def __init__(self, viprs_models):

        assert len(viprs_models) > 1

        self.viprs_models = viprs_models
        self.priors = None

    def update_priors(self):
        pass

    def fit(self, max_iter=10):

        for i in range(max_iter):
            for v in self.viprs_models:
                v.fit()

            self.update_priors()

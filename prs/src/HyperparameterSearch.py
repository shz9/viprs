import numpy as np
from scipy import stats
from skopt import gp_minimize
import itertools
import copy


class HyperparameterSearch(object):

    def __init__(self, prs_m, objective='ELBO', validation_gdl=None):

        self.prs_m = prs_m
        self._opt_objective = objective
        self._validation_gdl = validation_gdl

        assert self._opt_objective in ['ELBO', 'validation']

        if self._opt_objective == 'ELBO':
            assert hasattr(self.prs_m, 'objective')
        if self._opt_objective == 'validation':
            assert self._validation_gdl is not None
            assert self._validation_gdl.phenotypes is not None

    def objective(self):
        if self._opt_objective == 'ELBO':
            return self.prs_m.objective()
        else:
            prs = self.prs_m.predict(self._validation_gdl)
            _, _, r_val, _, _ = stats.linregress(prs, self._validation_gdl.phenotypes)
            return r_val**2

    def fit_bayes_opt(self, opt_params=('sigma_epsilon', 'pi'),
                      n_calls=20, n_random_starts=5, acq_func="LCB",
                      max_iter=100, tol=1e-4):
        """
        This function takes a variational PRS model and finds the
        hyperparameter settings that maximize its performance as
        measured by the ELBO.
        :param opt_params:
        :param n_calls:
        :param n_random_starts:
        :param acq_func:
        :param max_iter:
        :param tol:
        :return:
        """

        def opt_func(p):
            self.prs_m.fix_params = dict(zip(opt_params, p))
            try:
                self.prs_m.fit(max_iter=max_iter, ftol=tol, xtol=tol)
                return -self.objective()
            except Exception as e:
                return 1e12

        param_bounds = {
            'sigma_epsilon': (1e-6, 1.),
            'sigma_beta': (1e-12, .5),
            'pi': (1. / self.prs_m.M, 1. - 1. / self.prs_m.M)
        }

        res = gp_minimize(opt_func,  # the function to minimize
                          [param_bounds[p] for p in opt_params],  # the bounds on each dimension of x
                          acq_func=acq_func,  # the acquisition function
                          n_calls=n_calls,  # the number of evaluations of f
                          n_random_starts=n_random_starts)  # the random seed

        final_best_params = dict(zip(opt_params, res.x))
        print("Bayesian optimization identified best hyperparameters as:", final_best_params)

        self.prs_m.fix_params = final_best_params
        return self.prs_m.fit()

    def fit_grid_search(self, opt_params=('sigma_epsilon', 'pi'),
                        n_steps=10, max_iter=100, tol=1e-4):

        max_objective = -1e12
        best_params = None

        steps = {
            'sigma_epsilon': np.linspace(1. / n_steps, 1., n_steps),
            'pi': np.clip(10. ** (-np.linspace(np.floor(np.log10(self.prs_m.M)), 0., n_steps)),
                          a_min=1./self.prs_m.M, a_max=1. - 1. / self.prs_m.M)
        }

        for p in itertools.product(*[steps[k] for k in opt_params]):

            self.prs_m.fix_params = dict(zip(opt_params, p))

            try:
                self.prs_m.fit(max_iter=max_iter, ftol=tol, xtol=tol)
            except Exception:
                continue

            objective = self.objective()
            if objective > max_objective:
                max_objective = objective
                best_params = copy.deepcopy(p)

        final_best_params = dict(zip(opt_params, best_params))
        print("Grid search identified best hyperparameters as:", final_best_params)

        self.prs_m.fix_params = final_best_params
        return self.prs_m.fit()


def fit_model_averaging(vi_prs_m, opt_params=('sigma_epsilon', 'pi'),
                        n_steps=10, max_iter=200, tol=1e-4):
    steps = {
        'sigma_epsilon': np.linspace(1. / n_steps, 1., n_steps),
        'pi': np.clip(10. ** (-np.linspace(np.floor(np.log10(vi_prs_m.M)), 0., n_steps)),
                      a_min=1. / vi_prs_m.M, a_max=1. - 1. / vi_prs_m.M)
    }

    gamma = {c: np.zeros(c_size) for c, c_size in vi_prs_m.shapes.items()}
    mu_beta = {c: np.zeros(c_size) for c, c_size in vi_prs_m.shapes.items()}
    h2g = 0.
    pi = 0.
    elbo_sum = 0.

    for p in itertools.product(*[steps[k] for k in opt_params]):

        vi_prs_m.fix_params = dict(zip(opt_params, p))

        try:
            vi_prs_m.fit(max_iter=max_iter, ftol=tol, xtol=tol)
        except Exception as e:
            continue

        elbo = vi_prs_m.objective()
        elbo_sum += elbo
        h2g += elbo * vi_prs_m.get_heritability()
        pi += elbo * vi_prs_m.pi

        for c in vi_prs_m.shapes:
            gamma[c] += elbo * vi_prs_m.var_gamma[c]
            mu_beta[c] += elbo * vi_prs_m.var_mu_beta[c]

    h2g /= elbo_sum
    pi /= elbo_sum

    for c in vi_prs_m.shapes:
        gamma[c] /= elbo_sum
        mu_beta[c] /= elbo_sum

    return gamma, mu_beta, h2g, pi

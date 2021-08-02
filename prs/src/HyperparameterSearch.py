import numpy as np
from scipy import stats
from skopt import gp_minimize
from tqdm import tqdm
import itertools
from multiprocessing import Pool
import copy
from pprint import pprint

from .PRSModel import PRSModel
from .VIPRS import VIPRS


def generate_grid(M, n_steps=10, sigma_epsilon_steps=None, pi_steps=None, sigma_beta_steps=None):

    if sigma_epsilon_steps is None:
        sigma_epsilon_steps = n_steps
    if pi_steps is None:
        pi_steps = n_steps
    if sigma_beta_steps is None:
        sigma_beta_steps = n_steps

    return {
        'sigma_epsilon': np.linspace(1. / sigma_epsilon_steps, 1., sigma_epsilon_steps),
        'sigma_beta': (1./M)*np.linspace(1. / sigma_beta_steps, 1., sigma_beta_steps),
        'pi': np.clip(10. ** (-np.linspace(np.floor(np.log10(M)), 0., pi_steps)),
                      a_min=1. / M, a_max=1. - 1. / M)
    }


def fit_model_fixed_params(params):
    # vi_model, fixed_params, **fit_kwargs
    vi_model, fixed_params, fit_kwargs = params
    vi_model.fix_params = fixed_params

    try:
        vi_model.fit(**fit_kwargs)
    except Exception as e:
        return None, None, None, None

    return vi_model.objective(), vi_model.var_gamma, vi_model.var_mu_beta, vi_model.var_sigma_beta


class HyperparameterSearch(object):

    def __init__(self,
                 gdl,
                 viprs=None,
                 objective='ELBO',
                 validation_gdl=None,
                 verbose=False,
                 opt_params=('sigma_epsilon', 'pi')):

        self.gdl = gdl

        if viprs is None:
            self.viprs = VIPRS(gdl)
        else:
            self.viprs = viprs

        self.opt_params = opt_params
        self._opt_objective = objective
        self._validation_gdl = validation_gdl

        self.verbose = verbose
        self.viprs.verbose = verbose
        if self._validation_gdl is not None:
            self._validation_gdl.verbose = verbose

        assert self._opt_objective in ['ELBO', 'validation']

        if self._opt_objective == 'ELBO':
            assert hasattr(self.viprs, 'objective')
        if self._opt_objective == 'validation':
            assert self._validation_gdl is not None
            assert self._validation_gdl.phenotypes is not None

    def objective(self, fit_result):
        """
        A method that takes the output of `fit_model_fixed_params`
        and returns the objective (either ELBO, e.g. `fit_result[0]`
        or the validation R^2).
        :param fit_result:
        """
        if self._opt_objective == 'ELBO':
            return fit_result[0]
        else:
            _, gamma, beta, _ = fit_result
            prs = self._validation_gdl.predict({c: gamma[c]*b for c, b in beta.items()})
            _, _, r_val, _, _ = stats.linregress(prs, self._validation_gdl.phenotypes)
            return r_val**2

    def fit(self):
        raise NotImplementedError


class BayesOpt(HyperparameterSearch):

    def __init__(self,
                 gdl,
                 viprs=None,
                 objective='ELBO',
                 validation_gdl=None,
                 verbose=False,
                 opt_params=('sigma_epsilon', 'pi')):
        super().__init__(gdl, viprs=viprs,objective=objective,
                         validation_gdl=validation_gdl,
                         verbose=verbose,
                         opt_params=opt_params)

    def fit(self, n_calls=20, n_random_starts=5, acq_func="LCB",
            max_iter=100, tol=1e-4):

        """
        :param opt_params:
        :param n_calls:
        :param n_random_starts:
        :param acq_func:
        :param max_iter:
        :param tol:
        :return:
        """

        def opt_func(p):
            fit_result = fit_model_fixed_params((self.viprs,
                                                 dict(zip(self.opt_params, p)),
                                                 {'max_iter': max_iter, 'ftol': tol, 'xtol': tol}))

            if fit_result[0] is None:
                return 1e12
            else:
                return -self.objective(fit_result)

        param_bounds = {
            'sigma_epsilon': (1e-6, 1. - 1e-6),
            'sigma_beta': (1e-12, .5),
            'pi': (1. / self.gdl.M, 1. - 1. / self.gdl.M)
        }

        res = gp_minimize(opt_func,  # the function to minimize
                          [param_bounds[p] for p in self.opt_params],  # the bounds on each dimension of x
                          acq_func=acq_func,  # the acquisition function
                          n_calls=n_calls,  # the number of evaluations of f
                          n_random_starts=n_random_starts)  # the random seed

        final_best_params = dict(zip(self.opt_params, res.x))
        print("> Bayesian Optimization identified the best hyperparameters as:")
        pprint(final_best_params)

        print("> Refitting the model with the best hyperparameters...")

        self.viprs.fix_params = final_best_params
        return self.viprs.fit()


class GridSearch(HyperparameterSearch):

    def __init__(self,
                 gdl,
                 viprs=None,
                 objective='ELBO',
                 validation_gdl=None,
                 verbose=False,
                 opt_params=('sigma_epsilon', 'pi'),
                 n_proc=1):
        super().__init__(gdl, viprs=viprs, objective=objective,
                         validation_gdl=validation_gdl,
                         verbose=verbose,
                         opt_params=opt_params)

        self.viprs.threads = 1
        self.n_proc = n_proc

    def fit(self, max_iter=100, tol=1e-4, **grid_kwargs):

        steps = generate_grid(self.gdl.M, **grid_kwargs)
        print("> Performing Grid Search over the following grid:")
        pprint({k: v for k, v in steps.items() if k in self.opt_params})

        pool = Pool(self.n_proc)

        opts = [(self.viprs, dict(zip(self.opt_params, p)), {'max_iter': max_iter, 'ftol': tol, 'xtol': tol})
                for p in itertools.product(*[steps[k] for k in self.opt_params])]

        max_objective = -1e12

        for idx, fit_result in tqdm(enumerate(pool.imap(fit_model_fixed_params, opts)), total=len(opts)):

            if fit_result[0] is None:
                continue

            objective = self.objective(fit_result)
            if objective > max_objective:
                max_objective = objective
                best_params = copy.deepcopy([opts[idx][1][opt] for opt in self.opt_params])

        pool.close()
        pool.join()

        final_best_params = dict(zip(self.opt_params, best_params))
        print("> Grid search identified the best hyperparameters as:")
        pprint(final_best_params)

        print("> Refitting the model with the best hyperparameters...")

        self.viprs.fix_params = final_best_params
        return self.viprs.fit()


class BMA(PRSModel):
    """
    Bayesian Model Averaging fitting procedure
    """

    def __init__(self, gdl, viprs=None, opt_params=('sigma_epsilon', 'pi'),
                 verbose=False, obj_transform=None, n_proc=1):
        """

        :param gdl:
        :param viprs:
        :param opt_params:
        :param verbose:
        :param obj_transform:
        :param n_proc:
        """

        super().__init__(gdl)

        self.n_proc = n_proc
        self.verbose = verbose

        if viprs is None:
            self.viprs = VIPRS(gdl)
        else:
            self.viprs = viprs

        self.viprs.verbose = verbose
        self.viprs.threads = 1

        self.shapes = self.viprs.shapes
        self.opt_params = opt_params
        self.obj_transform = obj_transform or (lambda x: x)

        self.var_gamma = None
        self.var_mu_beta = None
        self.var_sigma_beta = None

        self.pip = None
        self.inf_beta = None

    def initialize(self):

        self.var_gamma = {c: np.zeros(c_size) for c, c_size in self.shapes.items()}
        self.var_mu_beta = {c: np.zeros(c_size) for c, c_size in self.shapes.items()}
        self.var_sigma_beta = {c: np.zeros(c_size) for c, c_size in self.shapes.items()}

    def fit(self, max_iter=100, tol=1e-4, **grid_kwargs):

        self.initialize()

        steps = generate_grid(self.M, **grid_kwargs)
        print("> Performing Bayesian Model Averaging with the following grid:")
        pprint({k: v for k, v in steps.items() if k in self.opt_params})

        pool = Pool(self.n_proc)

        opts = [(self.viprs, dict(zip(self.opt_params, p)), {'max_iter': max_iter, 'ftol': tol, 'xtol': tol})
                for p in itertools.product(*[steps[k] for k in self.opt_params])]

        elbo_sum = 0.

        for elbo, n_var_gamma, n_var_mu_beta, n_var_sigma_beta in \
                tqdm(pool.imap(fit_model_fixed_params, opts), total=len(opts)):

            if elbo is None:
                continue
            else:
                elbo = self.obj_transform(elbo)

            elbo_sum += elbo

            for c in self.shapes:
                self.var_gamma[c] += elbo * n_var_gamma[c]
                self.var_mu_beta[c] += elbo * n_var_mu_beta[c]
                self.var_sigma_beta[c] += elbo * n_var_sigma_beta[c]

        pool.close()
        pool.join()

        for c in self.shapes:
            self.var_gamma[c] /= elbo_sum
            self.var_mu_beta[c] /= elbo_sum
            self.var_sigma_beta[c] /= elbo_sum

        self.pip = self.var_gamma
        self.inf_beta = {c: self.var_gamma[c]*mu for c, mu in self.var_mu_beta.items()}

        return self

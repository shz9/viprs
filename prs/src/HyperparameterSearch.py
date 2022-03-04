import pandas as pd
import numpy as np
import copy
from tqdm import tqdm
import itertools
import multiprocessing
from pprint import pprint

from .evaluation import compute_r2, compute_auc
from .PRSModel import PRSModel
from .VIPRS import VIPRS


def generate_bounded_localized_grid(local_val, n_steps=9, base=1.75, a_min=1e-6, a_max=1. - 1e-6):
    """
    Generate a bounded and localized grid.
    This function takes a value between `a_min` and `a_max`
    and then generates a grid of `n_steps` around it using the specified base.
    """
    lb_grid = (base ** (np.arange(-np.floor(n_steps / 2), np.ceil(n_steps / 2))))*local_val
    return np.unique(np.clip(lb_grid, a_min=a_min, a_max=a_max))


def generate_grid(M,
                  n_steps=9,
                  h2g_estimate=None,
                  sigma_epsilon_steps=None,
                  pi_steps=None,
                  sigma_beta_steps=None,
                  alpha_steps=5):
    """
    TODO: Give user more fine-grained control over this. Maybe turn into a class
    that can be passed to GridSearch?
    :param M: The number of SNps included in the model
    :param n_steps: The overall number of steps
    :param sigma_epsilon_steps: The number of steps for sigma epsilon
    :param pi_steps: The number of steps for pi
    :param sigma_beta_steps: The number of steps for sigma_beta
    :param alpha_steps: The number of steps for alpha
    """

    if sigma_epsilon_steps is None:
        sigma_epsilon_steps = n_steps
    if pi_steps is None:
        pi_steps = n_steps
    if sigma_beta_steps is None:
        sigma_beta_steps = n_steps
    if alpha_steps is None:
        alpha_steps = n_steps

    grid = {
        'pi': np.clip(10. ** (-np.linspace(np.floor(np.log10(M)), 0., pi_steps)),
                      a_min=1. / M, a_max=1. - 1. / M),
        'alpha': np.linspace(-1., 0., alpha_steps)
    }

    if h2g_estimate is None:

        grid['sigma_epsilon'] = np.linspace(.1, .99, sigma_epsilon_steps)
        grid['sigma_beta'] = (1./M)*np.linspace(1. / sigma_beta_steps, 1., sigma_beta_steps)

    else:
        h2g_grid_e = generate_bounded_localized_grid(h2g_estimate, n_steps=sigma_epsilon_steps)
        grid['sigma_epsilon'] = 1. - h2g_grid_e

        h2g_grid_b = generate_bounded_localized_grid(h2g_estimate, n_steps=sigma_beta_steps)
        grid['sigma_beta'] = (1. / M) * h2g_grid_b

    return grid


def fit_model_fixed_params(params):
    # TODO: Figure out how to streamline this for VIPRSMix implementations
    # vi_model, fixed_params, **fit_kwargs
    vi_model, fixed_params, fit_kwargs = params
    vi_model.fix_params = fixed_params

    try:
        vi_model.fit(**fit_kwargs)
    except Exception as e:
        return None, None, None, None, None

    return vi_model.objective(), vi_model.var_gamma, vi_model.var_mu_beta, vi_model.var_sigma_beta, vi_model.inf_beta


class HyperparameterSearch(object):

    def __init__(self,
                 gdl,
                 viprs=None,
                 objective='ELBO',
                 validation_gdl=None,
                 verbose=False,
                 opt_params=('sigma_epsilon', 'pi'),
                 n_jobs=1):

        self.gdl = gdl
        self.n_jobs = n_jobs

        if viprs is None:
            self.viprs = VIPRS(gdl)
        else:
            self.viprs = viprs

        self.validation_result = None

        self.opt_params = opt_params
        self._opt_objective = objective
        self._validation_gdl = validation_gdl
        self._validation_to_train_map = None

        self.verbose = verbose
        self.viprs.verbose = verbose

        if self._validation_gdl is not None:
            self._validation_gdl.verbose = verbose

            # Match the index of SNPs in the training dataset
            # and SNPs in the validation dataset:
            self._validation_to_train_map = {}

            for c, train_snps in self.gdl.snps.items():
                valid_snps = self._validation_gdl.snps[c]

                train_df = pd.DataFrame({'SNP': train_snps}).reset_index()
                train_df.columns = ['train_index', 'SNP']

                valid_df = pd.DataFrame({'SNP': valid_snps}).reset_index()
                valid_df.columns = ['validation_index', 'SNP']

                self._validation_to_train_map[c] = train_df.merge(valid_df)[['train_index', 'validation_index']]

        assert self._opt_objective in ['ELBO', 'validation']

        if self._opt_objective == 'ELBO':
            assert hasattr(self.viprs, 'objective')
        if self._opt_objective == 'validation':
            assert self._validation_gdl is not None
            assert self._validation_gdl.phenotypes is not None

    def write_validation_result(self, v_filename):

        if self.validation_result is None:
            raise Exception("Validation result is not set!")
        elif len(self.validation_result) < 1:
            raise Exception("Validation result is not set!")

        v_df = pd.DataFrame(self.validation_result)
        v_df.to_csv(v_filename, index=False, sep="\t")

    def multi_objective(self, fit_results):
        """
        This method evaluates multiple models simultaneously
        :param fit_results:
        """

        if len(fit_results) == 1:
            return self.objective(fit_results[0])

        if self._opt_objective == 'ELBO':
            return [fr[0] for fr in fit_results]
        else:

            v_inf_beta = {}
            for c, shp in self._validation_gdl.shapes.items():
                valid_beta = np.zeros(shape=(shp, len(fit_results)))
                valid_beta[self._validation_to_train_map[c]['validation_index'].values, :] = np.array(
                    [inf_beta[c] for _, _, _, _, inf_beta in fit_results]).T[
                    self._validation_to_train_map[c]['train_index'].values, :
                ]
                v_inf_beta[c] = valid_beta

            prs = self._validation_gdl.predict(v_inf_beta)

            if self._validation_gdl.phenotype_likelihood == 'binomial':
                eval_func = compute_auc
            else:
                eval_func = compute_r2

            ctx = multiprocessing.get_context("spawn")

            with ctx.Pool(self.n_jobs, maxtasksperchild=1) as pool:
                r2_results = pool.starmap(eval_func,
                                          [(prs[:, i].flatten(), self._validation_gdl.phenotypes)
                                           for i in range(len(fit_results))])

            return r2_results

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
            _, _, _, _, inf_beta = fit_result

            # Match inferred betas with the SNPs in the validation GDL:
            v_inf_beta = {}
            for c, shp in self._validation_gdl.shapes.items():
                valid_beta = np.zeros(shp)
                valid_beta[self._validation_to_train_map[c]['validation_index'].values] = (inf_beta[c])[
                    self._validation_to_train_map[c]['train_index'].values
                ]
                v_inf_beta[c] = valid_beta

            # Predict:
            prs = self._validation_gdl.predict(v_inf_beta)

            if self._validation_gdl.phenotype_likelihood == 'binomial':
                eval_func = compute_auc
            else:
                eval_func = compute_r2

            return eval_func(prs, self._validation_gdl.phenotypes)

    def fit(self):
        raise NotImplementedError


class BayesOpt(HyperparameterSearch):

    def __init__(self,
                 gdl,
                 viprs=None,
                 objective='ELBO',
                 validation_gdl=None,
                 verbose=False,
                 opt_params=('sigma_epsilon', 'pi'),
                 n_jobs=1):
        
        super().__init__(gdl, viprs=viprs,
                         objective=objective,
                         validation_gdl=validation_gdl,
                         verbose=verbose,
                         opt_params=opt_params,
                         n_jobs=n_jobs)

    def fit(self, max_iter=50, f_abs_tol=1e-3, x_rel_tol=1e-3,
            n_calls=20, n_random_starts=5, acq_func="gp_hedge"):
        """
        :param n_calls: The number of model runs with different hyperparameter settings.
        :param n_random_starts: The number of random starts to initialize the optimizer.
        :param acq_func: The acquisition function (default: `gp_hedge`)
        :param max_iter: The maximum number of iterations within the search (default: 50).
        :param f_abs_tol: The absolute tolerance for the objective (ELBO) within the search
        :param x_rel_tol: The relative tolerance for the parameters within the search
        """

        from skopt import gp_minimize

        def opt_func(p):

            fix_params = dict(zip(self.opt_params, p))
            if 'pi' in fix_params:
                fix_params['pi'] = 10**fix_params['pi']

            fit_result = fit_model_fixed_params((self.viprs, fix_params,
                                                 {'max_iter': max_iter,
                                                  'f_abs_tol': f_abs_tol,
                                                  'x_rel_tol': x_rel_tol}))

            if fit_result[0] is None:
                return 1e12
            else:
                return -self.objective(fit_result)

        param_bounds = {
            'sigma_epsilon': (1e-6, 1. - 1e-6),
            'sigma_beta': (1e-12, 1.),
            'pi': (-np.floor(np.log10(self.gdl.M)), -.001)
        }

        res = gp_minimize(opt_func,  # the function to minimize
                          [param_bounds[p] for p in self.opt_params],  # the bounds on each dimension of x
                          acq_func=acq_func,  # the acquisition function
                          n_calls=n_calls,  # the number of evaluations of f
                          n_random_starts=n_random_starts)  # the random seed

        # Store validation result
        self.validation_result = []
        for obj, x in zip(res.func_vals, res.x_iters):
            v_res = dict(zip(self.opt_params, x))
            if 'pi' in v_res:
                v_res['pi'] = 10**v_res['pi']

            if self._opt_objective == 'ELBO':
                v_res['ELBO'] = -obj
            else:
                v_res['Validation R2'] = -obj

            self.validation_result.append(v_res)

        # Extract the best performing hyperparameters:
        final_best_params = dict(zip(self.opt_params, res.x))
        if 'pi' in final_best_params:
            final_best_params['pi'] = 10 ** final_best_params['pi']

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
                 localized_grid=True,
                 verbose=False,
                 opt_params=('sigma_epsilon', 'pi'),
                 n_jobs=1):

        super().__init__(gdl, viprs=viprs, objective=objective,
                         validation_gdl=validation_gdl,
                         verbose=verbose,
                         opt_params=opt_params,
                         n_jobs=n_jobs)

        self.localized_grid = localized_grid
        self.viprs.threads = 1

    def fit(self, max_iter=50, f_abs_tol=1e-3, x_rel_tol=1e-3, **grid_kwargs):

        if self.localized_grid:
            h2g = np.clip(self.gdl.estimate_snp_heritability(), a_min=1e-3, a_max=1. - 1e-3)
            steps = generate_grid(self.gdl.M, h2g_estimate=h2g, **grid_kwargs)
        else:
            steps = generate_grid(self.gdl.M, **grid_kwargs)

        print("> Performing Grid Search over the following grid:")
        pprint({k: v for k, v in steps.items() if k in self.opt_params})

        opts = [(self.viprs, dict(zip(self.opt_params, p)), {'max_iter': max_iter,
                                                             'f_abs_tol': f_abs_tol,
                                                             'x_rel_tol': x_rel_tol})
                for p in itertools.product(*[steps[k] for k in self.opt_params])]

        assert len(opts) > 1

        self.validation_result = []
        fit_results = []
        params = []

        ctx = multiprocessing.get_context("spawn")

        with ctx.Pool(self.n_jobs, maxtasksperchild=1) as pool:

            for idx, fit_result in tqdm(enumerate(pool.imap(fit_model_fixed_params, opts)), total=len(opts)):

                if fit_result[0] is None:
                    continue

                fit_results.append(fit_result)
                params.append(copy.copy(opts[idx][1]))
                self.validation_result.append(copy.copy(opts[idx][1]))
                self.validation_result[-1]['ELBO'] = fit_result[0]

        if len(fit_results) > 1:
            res_objectives = self.multi_objective(fit_results)
        else:
            raise Exception("Error: Convergence was achieved for less than 2 models.")

        if self._opt_objective == 'validation':
            for i in range(len(self.validation_result)):
                self.validation_result[i]['Validation R2'] = res_objectives[i]

        best_idx = np.argmax(res_objectives)
        best_params = params[best_idx]

        print("> Grid search identified the best hyperparameters as:")
        pprint(best_params)

        print("> Refitting the model with the best hyperparameters...")

        self.viprs.fix_params = best_params
        return self.viprs.fit()


class BMA(PRSModel):
    """
    Bayesian Model Averaging fitting procedure
    """

    def __init__(self, gdl,
                 viprs=None,
                 opt_params=('sigma_epsilon', 'pi'),
                 verbose=False,
                 localized_grid=True,
                 n_jobs=1):
        """

        :param gdl:
        :param viprs:
        :param opt_params:
        :param verbose:
        :param obj_transform:
        :param n_jobs:
        """

        super().__init__(gdl)

        self.n_jobs = n_jobs
        self.verbose = verbose

        if viprs is None:
            self.viprs = VIPRS(gdl)
        else:
            self.viprs = viprs

        self.viprs.verbose = verbose
        self.viprs.threads = 1
        self.localized_grid = localized_grid

        self.shapes = self.viprs.shapes
        self.opt_params = opt_params

        self.var_gamma = None
        self.var_mu_beta = None
        self.var_sigma_beta = None

        self.pip = None
        self.inf_beta = None

    def initialize(self):

        self.var_gamma = {c: np.zeros(c_size) for c, c_size in self.shapes.items()}
        self.var_mu_beta = {c: np.zeros(c_size) for c, c_size in self.shapes.items()}
        self.var_sigma_beta = {c: np.zeros(c_size) for c, c_size in self.shapes.items()}

    def fit(self, max_iter=100, f_abs_tol=1e-3, x_rel_tol=1e-3, **grid_kwargs):

        self.initialize()

        if self.localized_grid:
            h2g = np.clip(self.gdl.estimate_snp_heritability(), a_min=1e-3, a_max=1. - 1e-3)
            steps = generate_grid(self.gdl.M, h2g_estimate=h2g, **grid_kwargs)
        else:
            steps = generate_grid(self.gdl.M, **grid_kwargs)

        print("> Performing Bayesian Model Averaging with the following grid:")
        pprint({k: v for k, v in steps.items() if k in self.opt_params})

        opts = [(self.viprs, dict(zip(self.opt_params, p)), {'max_iter': max_iter,
                                                             'f_abs_tol': f_abs_tol,
                                                             'x_rel_tol': x_rel_tol})
                for p in itertools.product(*[steps[k] for k in self.opt_params])]

        elbos = []
        var_gammas = []
        var_mu_betas = []
        var_sigma_betas = []

        ctx = multiprocessing.get_context("spawn")

        with ctx.Pool(self.n_jobs, maxtasksperchild=1) as pool:
            for elbo, n_var_gamma, n_var_mu_beta, n_var_sigma_beta, _ in \
                    tqdm(pool.imap_unordered(fit_model_fixed_params, opts), total=len(opts)):

                if elbo is None:
                    continue

                elbos.append(elbo)
                var_gammas.append(n_var_gamma)
                var_mu_betas.append(n_var_mu_beta)
                var_sigma_betas.append(n_var_sigma_beta)

        elbos = np.array(elbos)
        # Correction for negative ELBOs:
        elbos = elbos - elbos.min() + 1.
        elbos /= elbos.sum()

        for idx in range(len(elbos)):
            for c in self.shapes:
                self.var_gamma[c] += var_gammas[idx][c]*elbos[idx]
                self.var_mu_beta[c] += var_mu_betas[idx][c]*elbos[idx]
                self.var_sigma_beta[c] += var_sigma_betas[idx][c]*elbos[idx]

        self.pip = {}
        self.inf_beta = {}

        for c, v_gamma in self.var_gamma.items():

            if len(v_gamma.shape) > 1:
                self.pip[c] = v_gamma.sum(axis=1)
                self.inf_beta[c] = (v_gamma*self.var_mu_beta[c]).sum(axis=1)
            else:
                self.pip[c] = v_gamma
                self.inf_beta[c] = v_gamma * self.var_mu_beta[c]

        return self

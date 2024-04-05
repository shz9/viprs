import pandas as pd
import numpy as np
import copy
from tqdm import tqdm
import multiprocessing
from pprint import pprint

from viprs.eval.continuous_metrics import r2
from viprs.eval.binary_metrics import roc_auc
from viprs.model.BayesPRSModel import BayesPRSModel
from viprs.model.VIPRS import VIPRS


def fit_model_fixed_params(params):
    """
    Perform model fitting using a set of fixed parameters.
    This is a helper function to allow us to use the `multiprocessing` module
    to fit PRS models in parallel.
    :param params: A tuple of (BayesPRSModel, fixed parameters dictionary, and kwargs for the .fit() method).
    """

    # vi_model, fixed_params, **fit_kwargs
    vi_model, fixed_params, fit_kwargs = params
    vi_model.fix_params = fixed_params

    try:
        vi_model.fit(**fit_kwargs)
    except Exception as e:
        return None

    return vi_model


class HyperparameterSearch(object):
    """
    A generic class for performing hyperparameter search on the
    `VIPRS` model. This interface is old and will likely be deprecated
    in future releases. It is recommended to use the `VIPRSGrid` class
    and its derivatives for performing grid search instead.
    """

    def __init__(self,
                 gdl,
                 model=None,
                 criterion='ELBO',
                 validation_gdl=None,
                 verbose=False,
                 n_jobs=1):
        """
        A generic hyperparameter search class that implements common functionalities
        that may be required by hyperparameter search strategies.
        :param gdl: A GWADataLoader object
        :param model: A `PRSModel`-derived object (e.g. VIPRS).
        :param criterion: The objective function for the hyperparameter search.
        Options are: `ELBO`, `pseudo_validation` or `validation`.
        :param validation_gdl: If the objective is validation, provide the GWADataLoader object for the validation
        dataset.
        :param verbose: Detailed messages and print statements.
        :param n_jobs: The number of processes to use for the hyperparameters search.
        """

        # Sanity checking:
        assert criterion in ('ELBO', 'validation', 'pseudo_validation')

        self.gdl = gdl
        self.n_jobs = n_jobs

        if model is None:
            self.model = VIPRS(gdl)
        else:
            import inspect
            if inspect.isclass(model):
                self.model = model(gdl)
            else:
                self.model = model

        self.validation_result = None

        self.criterion = criterion
        self._validation_gdl = validation_gdl

        self.verbose = verbose
        self.model.verbose = verbose

        if self._validation_gdl is not None:
            self._validation_gdl.verbose = verbose

        if self.criterion == 'ELBO':
            assert hasattr(self.model, 'elbo')
        elif self.criterion == 'pseudo_validation':
            assert self._validation_gdl is not None
            assert self._validation_gdl.sumstats_table is not None
        if self.criterion == 'validation':
            assert self._validation_gdl is not None
            assert self._validation_gdl.genotype is not None
            assert self._validation_gdl.sample_table.phenotype is not None

    def to_validation_table(self):
        """
        Summarize the validation results in a pandas table.
        """
        if self.validation_result is None:
            raise Exception("Validation result is not set!")
        elif len(self.validation_result) < 1:
            raise Exception("Validation result is not set!")

        return pd.DataFrame(self.validation_result)

    def write_validation_result(self, v_filename, sep="\t"):
        """
        After performing hyperparameter search, write a table
        that records that value of the objective for each combination
        of hyperparameters.
        :param v_filename: The filename for the validation table.
        :param sep: The separator for the validation table
        """

        v_df = self.to_validation_table()
        v_df.to_csv(v_filename, index=False, sep=sep)

    def multi_objective(self, models):
        """
        This method evaluates multiple PRS models simultaneously. This can be faster for
        some evaluation criteria, such as the validation R^2, because we only need to
        multiply the inferred effect sizes with the genotype matrix only once.

        :param models: A list of PRS models that we wish to evaluate.
        """

        if len(models) == 1:
            return self.objective(models[0])

        if self.criterion == 'ELBO':
            return [m.elbo() for m in models]

        elif self.criterion == 'pseudo_validation':
            return [m.pseudo_validate(validation_gdl=self._validation_gdl) for m in models]
        else:

            prs_m = BayesPRSModel(self._validation_gdl)

            eff_table = models[0].to_table(per_chromosome=False)
            eff_table = eff_table[['CHR', 'SNP', 'A1', 'A2', 'BETA']]
            eff_table.rename(columns={'BETA': 'BETA_0'}, inplace=True)

            eff_table[[f'BETA_{i}' for i in range(1, len(models))]] = np.array(
                [models[i].to_table(per_chromosome=False)['BETA'].values for i in range(1, len(models))]
            ).T

            prs_m.set_model_parameters(eff_table)

            prs = prs_m.predict(test_gdl=self._validation_gdl)

            if self._validation_gdl.phenotype_likelihood == 'binomial':
                eval_func = roc_auc
            else:
                eval_func = r2

            metrics = [eval_func(prs[:, i].flatten(), self._validation_gdl.sample_table.phenotype)
                       for i in range(len(models))]

            return metrics

    def objective(self, model):
        """
        A method that takes the result of fitting the model
        and returns the desired objective (either `ELBO`, `pseudo_validation`, or `validation`).
        :param model: The PRS model to evaluate
        """

        if self.criterion == 'ELBO':
            return model.elbo()
        elif self.criterion == 'pseudo_validation':
            return model.pseudo_validate(validation_gdl=self._validation_gdl)
        else:

            # Predict:
            prs = model.predict(test_gdl=self._validation_gdl)

            if self._validation_gdl.phenotype_likelihood == 'binomial':
                eval_func = roc_auc
            else:
                eval_func = r2

            return eval_func(prs, self._validation_gdl.sample_table.phenotype)

    def fit(self):
        raise NotImplementedError


class BayesOpt(HyperparameterSearch):
    """
    Hyperparameter search using Bayesian optimization
    """

    def __init__(self,
                 gdl,
                 opt_params,
                 param_bounds=None,
                 model=None,
                 criterion='ELBO',
                 validation_gdl=None,
                 verbose=False,
                 n_jobs=1):
        """
        Perform hyperparameter search using Bayesian optimization
        :param gdl: A GWADataLoader object
        :param opt_params: A list of the hyperparameters to optimize over (e.g. 'pi', 'sigma_epsilon', 'sigma_beta').
        :param param_bounds: The bounds for each hyperparameter included in the optimization. A list of tuples,
        where each tuples records the (min, max) values for each hyperparameter.
        :param model: A `PRSModel`-derived object (e.g. VIPRS).
        :param criterion: The objective function for the hyperparameter search (ELBO or validation).
        :param validation_gdl: If the objective is validation, provide the GWADataLoader object for the validation
        dataset.
        :param verbose: Detailed messages and print statements.
        :param n_jobs: The number of processes to use for the hyperparameters search (not applicable here).
        """
        
        super().__init__(gdl,
                         model=model,
                         criterion=criterion,
                         validation_gdl=validation_gdl,
                         verbose=verbose,
                         n_jobs=n_jobs)

        self._opt_params = opt_params
        self._param_bounds = param_bounds or {
            'sigma_epsilon': (1e-6, 1. - 1e-6),
            'tau_beta': (1e-3, None),
            'pi': (1e-6, 1. - 1e-6)
        }

        # Convert the `pi` limits to log-scale:
        if 'pi' in self._param_bounds:
            self._param_bounds['pi'] = tuple(np.log10(list(self._param_bounds['pi'])))

        assert all([opp in self._param_bounds for opp in self._opt_params])

    def fit(self,
            max_iter=50,
            f_abs_tol=1e-4,
            n_calls=30,
            n_random_starts=5,
            acq_func="gp_hedge"):
        """
        Perform model fitting and hyperparameter search using Bayesian optimization.

        :param n_calls: The number of model runs with different hyperparameter settings.
        :param n_random_starts: The number of random starts to initialize the optimizer.
        :param acq_func: The acquisition function (default: `gp_hedge`)
        :param max_iter: The maximum number of iterations within the search (default: 50).
        :param f_abs_tol: The absolute tolerance for the objective (ELBO) within the search
        """

        from skopt import gp_minimize

        def opt_func(p):

            fix_params = dict(zip(self._opt_params, p))
            if 'pi' in fix_params:
                fix_params['pi'] = 10**fix_params['pi']

            fitted_model = fit_model_fixed_params((self.model, fix_params,
                                                   {'max_iter': max_iter,
                                                    'f_abs_tol': f_abs_tol}))

            if fitted_model is None:
                return np.inf
            else:
                return -self.objective(fitted_model)

        res = gp_minimize(opt_func,  # the function to minimize
                          [self._param_bounds[op] for op in self._opt_params],  # the bounds on each dimension of x
                          acq_func=acq_func,  # the acquisition function
                          n_calls=n_calls,  # the number of evaluations of f
                          n_random_starts=n_random_starts)  # the random seed

        # Store validation result
        self.validation_result = []
        for obj, x in zip(res.func_vals, res.x_iters):
            v_res = dict(zip(self._opt_params, x))
            if 'pi' in v_res:
                v_res['pi'] = 10**v_res['pi']

            if self.criterion == 'ELBO':
                v_res['ELBO'] = -obj
            elif self.criterion == 'pseudo_validation':
                v_res['Pseudo_Validation_Corr'] = -obj
            else:
                v_res['Validation_R2'] = -obj

            self.validation_result.append(v_res)

        # Extract the best performing hyperparameters:
        final_best_params = dict(zip(self._opt_params, res.x))
        if 'pi' in final_best_params:
            final_best_params['pi'] = 10 ** final_best_params['pi']

        print("> Bayesian Optimization identified the best hyperparameters as:")
        pprint(final_best_params)

        print("> Refitting the model with the best hyperparameters...")

        self.model.fix_params = final_best_params
        return self.model.fit()


class GridSearch(HyperparameterSearch):
    """
    Hyperparameter search using Grid Search
    """

    def __init__(self,
                 gdl,
                 grid,
                 model=None,
                 criterion='ELBO',
                 validation_gdl=None,
                 verbose=False,
                 n_jobs=1):

        """
        Perform hyperparameter search using grid search
        :param gdl: A GWADataLoader object
        :param grid: A HyperParameterGrid object
        :param model: A `PRSModel`-derived object (e.g. VIPRS).
        :param criterion: The objective function for the grid search (ELBO or validation).
        :param validation_gdl: If the objective is validation, provide the GWADataLoader object for the validation
        dataset.
        :param verbose: Detailed messages and print statements.
        :param n_jobs: The number of processes to use for the grid search
        """

        super().__init__(gdl, model=model, criterion=criterion,
                         validation_gdl=validation_gdl,
                         verbose=verbose,
                         n_jobs=n_jobs)

        self.grid = grid
        self.model.threads = 1

    def fit(self, max_iter=50, f_abs_tol=1e-3, x_abs_tol=1e-8):

        print("> Performing Grid Search over the following grid:")
        print(self.grid.to_table())

        opts = [(self.model, g, {'max_iter': max_iter,
                                 'f_abs_tol': f_abs_tol,
                                 'x_abs_tol': x_abs_tol})
                for g in self.grid.combine_grids()]

        assert len(opts) > 1

        self.validation_result = []
        fit_results = []
        params = []

        ctx = multiprocessing.get_context("spawn")

        with ctx.Pool(self.n_jobs, maxtasksperchild=1) as pool:

            for idx, fitted_model in tqdm(enumerate(pool.imap(fit_model_fixed_params, opts)), total=len(opts)):

                if fitted_model is None:
                    continue

                fit_results.append(fitted_model)
                params.append(copy.copy(opts[idx][1]))
                self.validation_result.append(copy.copy(opts[idx][1]))
                self.validation_result[-1]['ELBO'] = fitted_model.elbo()

        if len(fit_results) > 1:
            res_objectives = self.multi_objective(fit_results)
        else:
            raise Exception("Error: Convergence was achieved for less than 2 models.")

        if self.criterion == 'validation':
            for i in range(len(self.validation_result)):
                self.validation_result[i]['Validation_R2'] = res_objectives[i]
        elif self.criterion == 'pseudo_validation':
            for i in range(len(self.validation_result)):
                self.validation_result[i]['Pseudo_Validation_Corr'] = res_objectives[i]

        best_idx = np.argmax(res_objectives)
        best_params = params[best_idx]

        print("> Grid search identified the best hyperparameters as:")
        pprint(best_params)

        print("> Refitting the model with the best hyperparameters...")

        self.model.fix_params = best_params
        return self.model.fit()


class BMA(BayesPRSModel):
    """
    Bayesian Model Averaging fitting procedure
    """

    def __init__(self,
                 gdl,
                 grid,
                 model=None,
                 normalization='softmax',
                 verbose=False,
                 n_jobs=1):
        """
        Integrate out hyperparameters using Bayesian Model Averaging
        :param gdl: A GWADataLoader object
        :param grid: A HyperParameterGrid object
        :param model: A `PRSModel`-derived object (e.g. VIPRS).
        :param normalization: The normalization scheme for the final ELBOs. Options are (`softmax`, `sum`).
        :param verbose: Detailed messages and print statements.
        :param n_jobs: The number of processes to use for the BMA
        """

        super().__init__(gdl)

        assert normalization in ('softmax', 'sum')

        self.grid = grid
        self.n_jobs = n_jobs
        self.verbose = verbose

        if model is None:
            self.model = VIPRS(gdl)
        else:
            self.model = model

        self.model.verbose = verbose
        self.model.threads = 1

        self.normalization = normalization

        self.var_gamma = None
        self.var_mu = None
        self.var_sigma = None

    def initialize(self):

        self.var_gamma = {c: np.zeros(c_size) for c, c_size in self.shapes.items()}
        self.var_mu = {c: np.zeros(c_size) for c, c_size in self.shapes.items()}
        self.var_sigma = {c: np.zeros(c_size) for c, c_size in self.shapes.items()}

    def fit(self, max_iter=100, f_abs_tol=1e-3, x_abs_tol=1e-8, **grid_kwargs):

        self.initialize()

        print("> Performing Bayesian Model Averaging with the following grid:")
        print(self.grid.to_table())

        opts = [(self.model, g, {'max_iter': max_iter,
                                 'f_abs_tol': f_abs_tol,
                                 'x_abs_tol': x_abs_tol})
                for g in self.grid.combine_grids()]

        elbos = []
        var_gammas = []
        var_mus = []
        var_sigmas = []

        ctx = multiprocessing.get_context("spawn")

        with ctx.Pool(self.n_jobs, maxtasksperchild=1) as pool:
            for fitted_model in tqdm(pool.imap_unordered(fit_model_fixed_params, opts), total=len(opts)):

                if fitted_model is None:
                    continue

                elbos.append(fitted_model.elbo())
                var_gammas.append(fitted_model.var_gamma)
                var_mus.append(fitted_model.var_mu)
                var_sigmas.append(fitted_model.var_sigma)

        elbos = np.array(elbos)

        if self.normalization == 'softmax':
            from scipy.special import softmax
            elbos = softmax(elbos)
        elif self.normalization == 'sum':
            # Correction for negative ELBOs:
            elbos = elbos - elbos.min() + 1.
            elbos /= elbos.sum()

        for idx in range(len(elbos)):
            for c in self.shapes:
                self.var_gamma[c] += var_gammas[idx][c]*elbos[idx]
                self.var_mu[c] += var_mus[idx][c]*elbos[idx]
                self.var_sigma[c] += var_sigmas[idx][c]*elbos[idx]

        self.pip = {}
        self.post_mean_beta = {}
        self.post_var_beta = {}

        for c, v_gamma in self.var_gamma.items():

            if len(v_gamma.shape) > 1:
                self.pip[c] = v_gamma.sum(axis=1)
                self.post_mean_beta[c] = (v_gamma*self.var_mu[c]).sum(axis=1)
                self.post_var_beta[c] = ((v_gamma * (self.var_mu[c] ** 2 + self.var_sigma[c])).sum(axis=1) -
                                         self.post_mean_beta[c]**2)
            else:
                self.pip[c] = v_gamma
                self.post_mean_beta[c] = v_gamma * self.var_mu[c]
                self.post_var_beta[c] = (v_gamma * (self.var_mu[c] ** 2 + self.var_sigma[c]) -
                                         self.post_mean_beta[c]**2)

        return self

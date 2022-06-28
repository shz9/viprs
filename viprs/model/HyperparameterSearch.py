import pandas as pd
import numpy as np
import copy
from tqdm import tqdm
import multiprocessing
from pprint import pprint

from viprs.eval.metrics import r2, auc
from .PRSModel import PRSModel
from .VIPRS import VIPRS


def fit_model_fixed_params(params):
    """
    Perform model fitting using a set of fixed parameters.
    This is a helper function to allow us to use the `multiprocessing` module
    to fit PRS models in parallel.
    :param params: A tuple of (PRSModel, fixed parameters dictionary, and kwargs for the .fit() method).
    """

    # TODO: Figure out how to streamline this for VIPRSMix implementations
    # vi_model, fixed_params, **fit_kwargs
    vi_model, fixed_params, fit_kwargs = params
    vi_model.fix_params = fixed_params

    try:
        vi_model.fit(**fit_kwargs)
    except Exception as e:
        return None, None, None, None, None

    return (
        vi_model.objective(),
        vi_model.var_gamma,
        vi_model.var_mu,
        vi_model.var_sigma,
        vi_model.get_posterior_mean_beta()
    )


class HyperparameterSearch(object):

    def __init__(self,
                 gdl,
                 model=None,
                 objective='ELBO',
                 validation_gdl=None,
                 verbose=False,
                 n_jobs=1):
        """
        A generic hyperparameter search class that implements common functionalities
        that may be required by hyperparameter search strategies.
        :param gdl: A GWADataLoader object
        :param model: A `PRSModel`-derived object (e.g. VIPRS).
        :param objective: The objective function for the hyperparameter search (ELBO or validation).
        :param validation_gdl: If the objective is validation, provide the GWADataLoader object for the validation
        dataset.
        :param verbose: Detailed messages and print statements.
        :param n_jobs: The number of processes to use for the hyperparameters search.
        """

        self.gdl = gdl
        self.n_jobs = n_jobs

        if model is None:
            self.model = VIPRS(gdl)
        else:
            self.model = model

        self.validation_result = None

        self._opt_objective = objective
        self._validation_gdl = validation_gdl
        self._validation_to_train_map = None

        self.verbose = verbose
        self.model.verbose = verbose

        if self._validation_gdl is not None:
            self._validation_gdl.verbose = verbose

            # Match the index of SNPs in the training dataset
            # and SNPs in the validation dataset:
            self._validation_to_train_map = {}

            for c in self.gdl.chromosomes:
                valid_snps = self._validation_gdl.genotype[c].snps

                train_df = pd.DataFrame({'SNP': self.gdl.sumstats_table[c].snps}).reset_index()
                train_df.columns = ['train_index', 'SNP']

                valid_df = pd.DataFrame({'SNP': valid_snps}).reset_index()
                valid_df.columns = ['validation_index', 'SNP']

                self._validation_to_train_map[c] = train_df.merge(valid_df)[['train_index', 'validation_index']]

        assert self._opt_objective in ['ELBO', 'validation']

        if self._opt_objective == 'ELBO':
            assert hasattr(self.model, 'objective')
        if self._opt_objective == 'validation':
            assert self._validation_gdl is not None
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

    def multi_objective(self, fit_results):
        """
        This method evaluates multiple models simultaneously. This is generally
        faster than evaluating each model separately. See also `.objective()`.
        :param fit_results: A list of fit results (output from `fit_model_fixed_params`).
        """

        if len(fit_results) == 1:
            return self.objective(fit_results[0])

        if self._opt_objective == 'ELBO':
            return [fr[0] for fr in fit_results]
        else:

            v_post_beta = {}
            for c, val_map in self._validation_to_train_map.items():
                valid_beta = np.zeros(shape=(self._validation_gdl.shapes[c], len(fit_results)))
                valid_beta[val_map['validation_index'].values, :] = np.array(
                    [post_beta[c] for _, _, _, _, post_beta in fit_results]).T[
                    val_map['train_index'].values, :
                ]
                v_post_beta[c] = valid_beta

            prs = self._validation_gdl.predict(v_post_beta)

            if self._validation_gdl.phenotype_likelihood == 'binomial':
                eval_func = auc
            else:
                eval_func = r2

            metrics = [eval_func(prs[:, i].flatten(), self._validation_gdl.sample_table.phenotype)
                       for i in range(len(fit_results))]

            return metrics

    def objective(self, fit_result):
        """
        A method that takes the result of fitting the model
        and returns the desired objective (either ELBO, e.g. or the validation R^2).
        :param fit_result: fit result (output from `fit_model_fixed_params`)
        """

        if self._opt_objective == 'ELBO':
            return fit_result[0]
        else:
            _, _, _, _, post_beta = fit_result

            # Match inferred betas with the SNPs in the validation GDL:
            v_post_beta = {}

            for c, val_map in self._validation_to_train_map.items():
                valid_beta = np.zeros(self._validation_gdl.shapes[c])
                valid_beta[val_map['validation_index'].values] = (post_beta[c])[
                    val_map['train_index'].values
                ]
                v_post_beta[c] = valid_beta

            # Predict:
            prs = self._validation_gdl.predict(v_post_beta)

            if self._validation_gdl.phenotype_likelihood == 'binomial':
                eval_func = auc
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
                 objective='ELBO',
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
        :param objective: The objective function for the hyperparameter search (ELBO or validation).
        :param validation_gdl: If the objective is validation, provide the GWADataLoader object for the validation
        dataset.
        :param verbose: Detailed messages and print statements.
        :param n_jobs: The number of processes to use for the hyperparameters search (not applicable here).
        """
        
        super().__init__(gdl,
                         model=model,
                         objective=objective,
                         validation_gdl=validation_gdl,
                         verbose=verbose,
                         n_jobs=n_jobs)

        self._opt_params = opt_params
        self._param_bounds = param_bounds or {
            'sigma_epsilon': (1e-6, 1. - 1e-6),
            'sigma_beta': (1e-9, 1. - 1e-9),
            'pi': (1e-6, 1. - 1e-6)
        }

        # Convert the `pi` limits to log-scale:
        if 'pi' in self._param_bounds:
            self._param_bounds['pi'] = tuple(np.log10(self._param_bounds['pi']))

        assert all([opp in self._param_bounds for opp in self._opt_params])

    def fit(self, max_iter=50, f_abs_tol=1e-3, x_rel_tol=1e-3,
            n_calls=20, n_random_starts=5, acq_func="gp_hedge"):
        """
        Perform model fitting and hyperparameter search using Bayesian optimization.

        :param n_calls: The number of model runs with different hyperparameter settings.
        :param n_random_starts: The number of random starts to initialize the optimizer.
        :param acq_func: The acquisition function (default: `gp_hedge`)
        :param max_iter: The maximum number of iterations within the search (default: 50).
        :param f_abs_tol: The absolute tolerance for the objective (ELBO) within the search
        :param x_rel_tol: The relative tolerance for the parameters within the search
        """

        from skopt import gp_minimize

        def opt_func(p):

            fix_params = dict(zip(self._opt_params, p))
            if 'pi' in fix_params:
                fix_params['pi'] = 10**fix_params['pi']

            fit_result = fit_model_fixed_params((self.model, fix_params,
                                                 {'max_iter': max_iter,
                                                  'f_abs_tol': f_abs_tol,
                                                  'x_rel_tol': x_rel_tol}))

            if fit_result[0] is None:
                return 1e12
            else:
                return -self.objective(fit_result)

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

            if self._opt_objective == 'ELBO':
                v_res['ELBO'] = -obj
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
                 objective='ELBO',
                 validation_gdl=None,
                 verbose=False,
                 n_jobs=1):

        """
        Perform hyperparameter search using grid search
        :param gdl: A GWADataLoader object
        :param grid: A HyperParameterGrid object (e.g. VIPRSGrid).
        :param model: A `PRSModel`-derived object (e.g. VIPRS).
        :param objective: The objective function for the grid search (ELBO or validation).
        :param validation_gdl: If the objective is validation, provide the GWADataLoader object for the validation
        dataset.
        :param verbose: Detailed messages and print statements.
        :param n_jobs: The number of processes to use for the grid search
        """

        super().__init__(gdl, model=model, objective=objective,
                         validation_gdl=validation_gdl,
                         verbose=verbose,
                         n_jobs=n_jobs)

        self.grid = grid
        self.model.threads = 1

    def fit(self, max_iter=50, f_abs_tol=1e-3, x_rel_tol=1e-3):

        print("> Performing Grid Search over the following grid:")
        print(self.grid.to_table())

        opts = [(self.model, g, {'max_iter': max_iter,
                                 'f_abs_tol': f_abs_tol,
                                 'x_rel_tol': x_rel_tol})
                for g in self.grid.combine_grids()]

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
                self.validation_result[i]['Validation_R2'] = res_objectives[i]

        best_idx = np.argmax(res_objectives)
        best_params = params[best_idx]

        print("> Grid search identified the best hyperparameters as:")
        pprint(best_params)

        print("> Refitting the model with the best hyperparameters...")

        self.model.fix_params = best_params
        return self.model.fit()


class BMA(PRSModel):
    """
    Bayesian Model Averaging fitting procedure
    """

    def __init__(self,
                 gdl,
                 grid,
                 model=None,
                 verbose=False,
                 n_jobs=1):
        """
        Integrate out hyperparameters using Bayesian Model Averaging
        :param gdl: A GWADataLoader object
        :param grid: A HyperParameterGrid object (e.g. VIPRSGrid).
        :param model: A `PRSModel`-derived object (e.g. VIPRS).
        :param verbose: Detailed messages and print statements.
        :param n_jobs: The number of processes to use for the BMA
        """

        super().__init__(gdl)

        self.grid = grid
        self.n_jobs = n_jobs
        self.verbose = verbose

        if model is None:
            self.model = VIPRS(gdl)
        else:
            self.model = model

        self.model.verbose = verbose
        self.model.threads = 1

        self.var_gamma = None
        self.var_mu = None
        self.var_sigma = None

    def initialize(self):

        self.var_gamma = {c: np.zeros(c_size) for c, c_size in self.shapes.items()}
        self.var_mu = {c: np.zeros(c_size) for c, c_size in self.shapes.items()}
        self.var_sigma = {c: np.zeros(c_size) for c, c_size in self.shapes.items()}

    def fit(self, max_iter=100, f_abs_tol=1e-3, x_rel_tol=1e-3, **grid_kwargs):

        self.initialize()

        print("> Performing Bayesian Model Averaging with the following grid:")
        print(self.grid.to_table())

        opts = [(self.model, g, {'max_iter': max_iter,
                                 'f_abs_tol': f_abs_tol,
                                 'x_rel_tol': x_rel_tol})
                for g in self.grid.combine_grids()]

        elbos = []
        var_gammas = []
        var_mus = []
        var_sigmas = []

        ctx = multiprocessing.get_context("spawn")

        with ctx.Pool(self.n_jobs, maxtasksperchild=1) as pool:
            for elbo, n_var_gamma, n_var_mu, n_var_sigma, _ in \
                    tqdm(pool.imap_unordered(fit_model_fixed_params, opts), total=len(opts)):

                if elbo is None:
                    continue

                elbos.append(elbo)
                var_gammas.append(n_var_gamma)
                var_mus.append(n_var_mu)
                var_sigmas.append(n_var_sigma)

        elbos = np.array(elbos)
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

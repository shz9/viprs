import pandas as pd
import numpy as np
from multiprocessing import shared_memory
from joblib import Parallel, delayed

from viprs.eval.continuous_metrics import r2
from viprs.eval.binary_metrics import roc_auc
from viprs.model.BayesPRSModel import BayesPRSModel
from viprs.model.VIPRS import VIPRS

# Set up the logger:
import logging
logger = logging.getLogger(__name__)


def fit_model_fixed_params(model, fixed_params, shm_data=None, **fit_kwargs):
    """

    Perform model fitting using a set of fixed set of hyperparameters.
    This is a helper function to allow users to use the `multiprocessing` module
    to fit PRS models in parallel.

    :param model: A PRS model object that implements a `.fit()` method and takes `fix_params` as an attribute.
    :param fixed_params: A dictionary of fixed parameters to use for the model fitting.
    :param shm_data: A dictionary of shared memory data to use for the model fitting. This is primarily used to
    share LD data across multiple processes.
    :param fit_kwargs: Key-word arguments to pass to the `.fit()` method of the PRS model.

    :return: A dictionary containing the coefficient table, hyperparameter table and the training objective.
    If the model did not converge successfully, return `None`.
    """

    model.fix_params = fixed_params

    if shm_data is not None:

        model.ld_data = {}

        try:
            ld_data_shm = shared_memory.SharedMemory(name=shm_data['shm_name'])
            model.ld_data[shm_data['chromosome']] = np.ndarray(
                shape=shm_data['shm_shape'],
                dtype=shm_data['shm_dtype'],
                buffer=ld_data_shm.buf
            )
        except FileNotFoundError:
            raise Exception("LD data not found in shared memory.")

    try:
        model.fit(**fit_kwargs)
    except Exception as e:
        logger.warning("Exception encountered when fitting model:", e)
        model = None
    finally:
        if shm_data is not None:
            ld_data_shm.close()
            model.ld_data = None

    if model is not None:
        return {
            'coef_table': model.to_table()[['CHR', 'SNP', 'POS', 'A1', 'A2', 'BETA']],
            'hyp_table': model.to_theta_table(),
            'training_objective': model.objective()
        }


class BaseHyperparamSearch(object):
    """
    A generic class for performing hyperparameter search on any genetic PRS model.
    This API is under active development and some of the components may change in the near future.

    TODO: Allow users to choose different metrics under each criterion.

    """

    def __init__(self,
                 gdl,
                 model=None,
                 criterion='training_objective',
                 validation_gdl=None,
                 n_jobs=1):
        """
        A generic hyperparameter search class that implements common functionalities
        that may be required by hyperparameter search strategies.
        :param gdl: A GWADataLoader object containing the GWAS summary statistics for inference.
        :param model: An instance of the PRS model to use for the hyperparameter search. By default,
        we use `VIPRS`.
        :param criterion: The objective function for the hyperparameter search.
        Options are: `training_objective`, `pseudo_validation` or `validation`. In the case of `VIPRS`, the training
        objective is the ELBO.
        :param validation_gdl: If the objective is validation or pseudo-validation, provide the GWADataLoader
        object for the validation dataset. If the criterion is pseudo-validation, the `validation_gdl` should
        contain summary statistics from a held-out test set. If the criterion is validation, `validation_gdl` should
        contain individual-level data from a held-out test set.
        :param n_jobs: The number of processes to use for the hyperparameters search.
        """

        # Sanity checking:
        assert criterion in ('training_objective', 'validation', 'pseudo_validation')

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

        self._model_coefs = None
        self._model_hyperparams = None
        self._training_objective = None

        # Sanity checks:
        if self.criterion == 'training_objective':
            assert hasattr(self.model, 'objective')
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
        :return: A pandas DataFrame with the validation results.
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

    def _evaluate_models(self):
        """
        This method evaluates multiple PRS models to determine their relative performance based on the
        criterion set by the user. The criterion can be the training objective (e.g. ELBO in the case of VIPRS),
        pseudo-validation or validation using held-out test data.

        :return: The metrics associated with each model setup.
        """

        assert self._training_objective is not None
        assert self._model_coefs is not None

        if self.criterion == 'training_objective':
            metrics = self._training_objective
        elif self.criterion == 'pseudo_validation':
            from viprs.eval.pseudo_metrics import pseudo_r2

            metrics = pseudo_r2(self._validation_gdl, self._model_coefs)

        else:

            prs_m = BayesPRSModel(self._validation_gdl)
            prs_m.set_model_parameters(self._model_coefs)

            prs = prs_m.predict(test_gdl=self._validation_gdl)

            if self._validation_gdl.phenotype_likelihood == 'binomial':
                eval_func = roc_auc
            else:
                eval_func = r2

            metrics = [eval_func(prs[:, i].flatten(), self._validation_gdl.sample_table.phenotype)
                       for i in range(prs.shape[1])]

        return metrics

    def fit(self):
        raise NotImplementedError


class GridSearch(BaseHyperparamSearch):
    """
    Hyperparameter search using Grid Search
    """

    def __init__(self,
                 gdl,
                 grid,
                 model=None,
                 criterion='training_objective',
                 validation_gdl=None,
                 n_jobs=1):

        """
        Perform hyperparameter search using grid search
        :param gdl: A GWADataLoader object containing the GWAS summary statistics for inference.
        :param model: An instance of the PRS model to use for the hyperparameter search. By default,
        we use `VIPRS`.
        :param criterion: The objective function for the hyperparameter search.
        Options are: `training_objective`, `pseudo_validation` or `validation`. In the case of `VIPRS`, the training
        objective is the ELBO.
        :param validation_gdl: If the objective is validation or pseudo-validation, provide the GWADataLoader
        object for the validation dataset. If the criterion is pseudo-validation, the `validation_gdl` should
        contain summary statistics from a held-out test set. If the criterion is validation, `validation_gdl` should
        contain individual-level data from a held-out test set.
        :param n_jobs: The number of processes to use for the hyperparameters search.
        """

        super().__init__(gdl,
                         model=model,
                         criterion=criterion,
                         validation_gdl=validation_gdl,
                         n_jobs=n_jobs)

        self.grid = grid
        self.model.threads = 1

    def fit(self, max_iter=1000, f_abs_tol=1e-6, x_abs_tol=1e-6):
        """
        Perform grid search over the hyperparameters to determine the
        best model based on the criterion set by the user. This utility method
        performs model fitting across the grid of hyperparameters, potentially in parallel
        if `n_jobs` is greater than 1.

        :param max_iter: The maximum number of iterations to run for each model fit.
        :param f_abs_tol: The absolute tolerance for the function convergence criterion.
        :param x_abs_tol: The absolute tolerance for the parameter convergence criterion.

        :return: The best model based on the criterion set by the user.
        """

        logger.info("> Performing Grid Search over the following grid:")
        logger.info(self.grid.to_table())

        if self.n_jobs > 1:
            # Only create the shared memory object if the number of processes is more than 1.
            # Otherwise, this would be a waste of resources.

            # ----------------- Copy the LD data to shared memory -----------------
            ld_data_arr = self.model.ld_data[self.model.chromosomes[0]]
            # Create a shared memory block for the array
            shm = shared_memory.SharedMemory(create=True, size=ld_data_arr.nbytes)

            # Create a NumPy array backed by the shared memory block
            shared_array = np.ndarray(ld_data_arr.shape, dtype=ld_data_arr.dtype, buffer=shm.buf)

            np.copyto(shared_array, ld_data_arr)

            del ld_data_arr
            self.model.ld_data = None

            shm_args = {
                'shm_name': shm.name,
                'chromosome': self.model.chromosomes[0],
                'shm_shape': shared_array.shape,
                'shm_dtype': shared_array.dtype
            }

        else:
            shm_args = None

        # --------------------------------------------------------------------
        # Perform grid search:

        grid = self.grid.combine_grids()

        parallel = Parallel(n_jobs=self.n_jobs, backend='multiprocessing')

        with parallel:

            fitted_models = parallel(
                delayed(fit_model_fixed_params)(self.model, g, shm_args,
                                                max_iter=max_iter,
                                                f_abs_tol=f_abs_tol,
                                                x_abs_tol=x_abs_tol)
                for g in grid
            )

        # Clean up after performing model fit:
        self.model.ld_data = None  # To minimize memory usage with validation/pseudo-validation

        # Close and unlink shared memory objects:
        if shm_args is not None:
            shm.close()
            shm.unlink()

        # --------------------------------------------------------------------
        # Post-process the results and determine the best model:

        assert not all([fm is None for fm in fitted_models]), "None of the models converged successfully."

        # 1) Extract the data from the trained models:
        from viprs.utils.compute_utils import combine_coefficient_tables

        self._model_coefs = combine_coefficient_tables([fm['coef_table'] for fm in fitted_models if fm is not None])
        self._model_hyperparams = [fm['hyp_table'] for fm in fitted_models if fm is not None]
        self._training_objective = [fm['training_objective'] for fm in fitted_models if fm is not None]

        # 2) Perform evaluation on the models that converged:
        eval_metrics = self._evaluate_models()

        # 3) Combine all the results together into a single table (populate records in
        # self.validation_result):

        self.validation_result = []
        success_counter = 0

        for i, vr in enumerate(grid):
            if fitted_models[i] is not None:
                vr['Converged'] = True
                vr['training_objective'] = self._training_objective[success_counter]
                if self.criterion != 'training_objective':
                    vr[self.criterion] = eval_metrics[success_counter]
                success_counter += 1
            else:
                vr['Converged'] = False
                vr['training_objective'] = np.NaN
                if self.criterion != 'training_objective':
                    vr[self.criterion] = np.NaN

            self.validation_result.append(vr)

        # --------------------------------------------------------------------
        # Determine and return the best model:

        best_idx = np.argmax(self.to_validation_table()[self.criterion].values)

        logger.info("> Grid search identified the best hyperparameters as:")
        logger.info(grid[best_idx])

        self.model.fix_params = grid[best_idx]
        self.model.initialize()
        self.model.set_model_parameters(fitted_models[best_idx]['coef_table'])

        return self.model

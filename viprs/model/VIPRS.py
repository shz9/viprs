import numpy as np
import pandas as pd
import logging
from tqdm.auto import tqdm

from .BayesPRSModel import BayesPRSModel
from ..utils.exceptions import OptimizationDivergence
from .vi.e_step import e_step
from .vi.e_step_cpp import cpp_e_step
from ..utils.OptimizeResult import OptimizeResult
from ..utils.compute_utils import dict_mean, dict_sum, dict_concat
from magenpy.utils.compute_utils import is_numeric


class VIPRS(BayesPRSModel):
    """
    The base class for performing Variational Inference of Polygenic Risk Scores (VIPRS).

    This class implements the Variational EM algorithm for estimating the posterior distribution
    of the effect sizes using GWAS summary statistics. The model assumes a spike-and-slab mixture
    prior on the effect size distribution, with the spike component representing the null effects
    and the slab component representing the non-null effects.

    Details for the algorithm can be found in the Supplementary Material of the following paper:

    > Zabad S, Gravel S, Li Y. Fast and accurate Bayesian polygenic risk modeling with variational inference.
    Am J Hum Genet. 2023 May 4;110(5):741-761. doi: 10.1016/j.ajhg.2023.03.009.
    Epub 2023 Apr 7. PMID: 37030289; PMCID: PMC10183379.

    :ivar gdl: An instance of GWADataLoader containing harmonized GWAS summary statistics and LD matrices.
    :ivar var_gamma: A dictionary of the variational gamma parameter, denoting the probability that the
    variant comes from the slab component.
    :ivar var_mu: A dictionary of the variational mu parameter, denoting the mean of the
    effect size for each variant.
    :ivar var_tau: A dictionary of the variational tau parameter, denoting the precision of
    the effect size for each variant.
    :ivar eta: A dictionary of the posterior mean of the effect size, E[B] = gamma*mu.
    :ivar zeta: A dictionary of the expectation of B^2 under the posterior, E[B^2] = gamma*(mu^2 + 1./tau).
    :ivar eta_diff: A dictionary of the difference between the etas in two consecutive iterations.
    :ivar q: A dictionary of the q-factor, which keeps track of the multiplication of eta with the LD matrix.
    :ivar sigma_epsilon: The global residual variance parameter.
    :ivar tau_beta: The prior precision (inverse variance) for the effect size.
    :ivar pi: The proportion of causal variants.
    :ivar _sigma_g: A pseudo-estimate of the additive genotypic variance.
    :ivar lambda_min: The minimum eigenvalue for the LD matrix (or an approximation of it that will serve
    as a regularizer).
    :ivar ld_data: A dictionary of the `data` arrays of the sparse LD matrices.
    :ivar ld_indptr: A dictionary of the `indptr` arrays of the sparse LD matrices.
    :ivar ld_left_bound: A dictionary of the left boundaries of the LD matrices.
    :ivar std_beta: A dictionary of the standardized marginal effect sizes from GWAS.
    :ivar Nj: A dictionary of the sample size per SNP from the GWAS study.
    :ivar threads: The number of threads to use when fitting the model.
    :ivar fix_params: A dictionary of hyperparameters with their fixed values.
    :ivar float_precision: The precision of the floating point variables. Options are: 'float32' or 'float64'.
    :ivar order: The order of the arrays in memory. Options are: 'C' or 'F'.
    :ivar low_memory: A boolean flag to indicate whether to use low memory mode.
    :ivar dequantize_on_the_fly: A boolean flag to indicate whether to dequantize the LD matrix on the fly.
    :ivar use_cpp: A boolean flag to indicate whether to use the C++ backend.
    :ivar use_blas: A boolean flag to indicate whether to use BLAS for linear algebra operations.
    :ivar optim_result: An instance of OptimizeResult tracking the progress of the optimization algorithm.
    :ivar verbose: Verbosity of the information printed to standard output. Can be boolean or an integer.
    :ivar history: A dictionary to store the history of the optimization procedure (e.g. the objective as a function
    of iteration number).
    :ivar tracked_params: A list of hyperparameters to track throughout the optimization procedure. Useful for
    debugging/model checking.

    """

    def __init__(self,
                 gdl,
                 fix_params=None,
                 tracked_params=None,
                 verbose=True,
                 lambda_min=None,
                 float_precision='float32',
                 order='F',
                 low_memory=True,
                 use_blas=True,
                 use_cpp=True,
                 dequantize_on_the_fly=False,
                 threads=1):

        """

        Initialize the VIPRS model.

        .. note::
            The initialization of the model involves loading the LD matrix to memory.

        :param gdl: An instance of GWADataLoader containing harmonized GWAS summary statistics and LD matrices.
        :param fix_params: A dictionary of hyperparameters with their fixed values.
        :param tracked_params: A list of hyperparameters/quantities to track throughout the optimization
        procedure. Useful for debugging/model checking. Currently, we allow the user to track the following:

            * The proportion of causal variants (`pi`).
            * The heritability ('heritability').
            * The residual variance (`sigma_epsilon`).
            * The prior precision for the effect size (`tau_beta`).
            * The additive genotypic variance (`sigma_g`).
            * The maximum difference in the posterior mean between iterations (`max_eta_diff`).
            * User may also provide arbitrary functions that takes the `VIPRS` object as input and
            compute any quantity of interest from it.

        :param verbose: Verbosity of the information printed to standard output. Can be boolean or an integer.
        Provide a number greater than 1 for more detailed output.
        :param lambda_min: The minimum eigenvalue for the LD matrix (or an approximation of it that will serve
        as a regularizer). If set to 'infer', the minimum eigenvalue will be computed or retrieved from the LD matrix.
        :param float_precision: The precision of the floating point variables. Options are: 'float32' or 'float64'.
        :param order: The order of the arrays in memory. Options are: 'C' or 'F'.
        :param low_memory: A boolean flag to indicate whether to use low memory mode.
        :param use_blas: A boolean flag to indicate whether to use BLAS for linear algebra operations.
        :param use_cpp: A boolean flag to indicate whether to use the C++ backend.
        :param dequantize_on_the_fly: A boolean flag to indicate whether to dequantize the LD matrix on the fly.
        :param threads: The number of threads to use when fitting the model.
        """

        super().__init__(gdl)

        # ------------------- Sanity checks -------------------

        assert gdl.ld is not None, "The LD matrices must be initialized in the GWADataLoader object."
        assert gdl.sumstats_table is not None, ("The summary statistics must be "
                                                "initialized in the GWADataLoader object.")

        if dequantize_on_the_fly and not use_cpp:
            raise Exception("Dequantization on the fly is only supported when using the C++ backend.")

        # ------------------- Initialize the model -------------------

        # Variational parameters:
        self.var_gamma = {}
        self.var_mu = {}
        self.var_tau = {}

        # Cache this quantity:
        self._log_var_tau = {}

        # Properties of proposed variational distribution:
        self.eta = {}  # The posterior mean, E[B] = \gamma*\mu_beta
        self.zeta = {}  # The expectation of B^2 under the posterior, E[B^2] = \gamma*(\mu_beta^2 + 1./\tau_beta)

        # The difference between the etas in two consecutive iterations (can be used for checking convergence,
        # or implementing optimized updates in the E-Step).
        self.eta_diff = {}

        # q-factor (keeps track of LD-related terms)
        self.q = {}

        # ---------- Model hyperparameters ----------

        self.sigma_epsilon = None
        self.tau_beta = None
        self.pi = None
        self._sigma_g = None  # A proxy for the additive genotypic variance
        self.lambda_min = None

        # ---------- Inputs to the model: ----------

        # NOTE: Here, we typecast the inputs to the model to the specified float precision.
        # This also needs to be done in the initialization methods.

        # LD-related quantities:

        self.ld_data = {}
        self.ld_indptr = {}
        self.ld_left_bound = {}

        for c, ld_mat in self.gdl.get_ld_matrices().items():

            # Determine how to load the LD data:
            if dequantize_on_the_fly and np.issubdtype(ld_mat.stored_dtype, np.integer):
                dtype = ld_mat.stored_dtype
            else:

                if dequantize_on_the_fly:
                    logging.warning("Dequantization on the fly is only supported for "
                                    "integer data types. Ignoring this flag.")

                dtype = float_precision
                dequantize_on_the_fly = False

            # Load the LD data:
            self.ld_data[c], self.ld_indptr[c], self.ld_left_bound[c] = ld_mat.load_data(
                return_symmetric=not low_memory,
                dtype=dtype
            )

            # Obtain / infer lambda_min:
            # TODO: Handle cases where we do inference over multiple chromosomes.
            # In this case, `lambda_min` should ideally be a dictionary.
            if lambda_min is None:
                self.lambda_min = 0.
            elif is_numeric(lambda_min):

                self.lambda_min = lambda_min

                if not np.isscalar(self.lambda_min):
                    assert self.lambda_min.shape == self.ld_indptr[c].shape[0] - 1, \
                        "Vector-valued lambda_min must have the same shape as the LD matrix."
            else:

                # If lambda min is set to `infer`, we try to retrieve information about the
                # spectral properties of the LD matrix from the LDMatrix object.
                # If this is not available, we set the minimum eigenvalue to 0.
                self.lambda_min = ld_mat.get_lambda_min(min_max_ratio=1e-3)

        # Standardized betas:
        self.std_beta = {c: ss.get_snp_pseudo_corr().astype(float_precision)
                         for c, ss in self.gdl.sumstats_table.items()}

        # Make sure that the data type for the sample size-per-SNP has the correct format:

        self.Nj = {c: nj.astype(float_precision, order=order)
                   for c, nj in self.Nj.items()}

        # ---------- General properties: ----------

        self.threads = threads
        self.fix_params = fix_params or {}

        self.float_precision = float_precision
        self.float_resolution = np.finfo(self.float_precision).resolution
        self.order = order
        self.low_memory = low_memory

        self.dequantize_on_the_fly = dequantize_on_the_fly

        if self.dequantize_on_the_fly:
            info = np.iinfo(self.ld_data[c].dtype)
            self.dequantize_scale = 1. / info.max
        else:
            self.dequantize_scale = 1.

        self.use_cpp = use_cpp
        self.use_blas = use_blas

        self.optim_result = OptimizeResult()
        self.verbose = verbose
        self.history = {}
        self.tracked_params = tracked_params or []

    def initialize(self, theta_0=None, param_0=None):
        """
        A convenience method to initialize all the objects associated with the model.
        :param theta_0: A dictionary of initial values for the hyperparameters theta
        :param param_0: A dictionary of initial values for the variational parameters
        """

        logging.debug("> Initializing model parameters")

        self.initialize_theta(theta_0)
        self.initialize_variational_parameters(param_0)
        self.init_optim_meta()

    def init_optim_meta(self):
        """
        Initialize the various quantities/objects to keep track of the optimization process.
         This method initializes the "history" object (which keeps track of the objective + other
         hyperparameters requested by the user), in addition to the OptimizeResult objects.
        """

        self.history = {
            'ELBO': [],
        }

        for tt in self.tracked_params:
            if isinstance(tt, str):
                self.history[tt] = []
            elif callable(tt):
                self.history[tt.__name__] = []

        self.optim_result.reset()

    def initialize_theta(self, theta_0=None):
        """
        Initialize the global hyperparameters of the model.
        :param theta_0: A dictionary of initial values for the hyperparameters theta
        """

        if theta_0 is not None and self.fix_params is not None:
            theta_0.update(self.fix_params)
        elif self.fix_params is not None:
            theta_0 = self.fix_params
        elif theta_0 is None:
            theta_0 = {}

        # ----------------------------------------------
        # (1) If 'pi' is not set, initialize from a uniform
        if 'pi' not in theta_0:
            self.pi = np.clip(np.random.uniform(low=0.005, high=0.1),
                              a_min=1./self.n_snps,
                              a_max=0.2*self.n/self.n_snps)
        else:
            self.pi = theta_0['pi']

        # ----------------------------------------------
        # (2) Initialize sigma_epsilon and tau_beta
        # Assuming that the genotype and phenotype are normalized,
        # these two quantities are conceptually linked.
        # The initialization routine here assumes that:
        # Var(y) = h2 + sigma_epsilon
        # Where, by assumption, Var(y) = 1,
        # And h2 ~= pi*M/tau_beta

        if 'sigma_epsilon' not in theta_0:
            if 'tau_beta' not in theta_0:

                # If neither tau_beta nor sigma_epsilon are given,
                # then initialize using the SNP heritability estimate

                try:
                    from magenpy.stats.h2.ldsc import simple_ldsc
                    naive_h2g = np.clip(simple_ldsc(self.gdl), a_min=.05, a_max=.95)
                except Exception as e:
                    print(e)
                    naive_h2g = np.random.uniform(low=.01, high=.1)

                self.sigma_epsilon = 1. - naive_h2g
                self.tau_beta = self.pi * self.n_snps / max(naive_h2g, 0.05)
            else:

                # If tau_beta is given, use it to initialize sigma_epsilon

                self.tau_beta = theta_0['tau_beta']
                self.sigma_epsilon = np.clip(1. - (self.pi * self.n_snps / self.tau_beta),
                                             a_min=self.float_resolution,
                                             a_max=1. - self.float_resolution)
        else:

            # If sigma_epsilon is given, use it in the initialization

            self.sigma_epsilon = theta_0['sigma_epsilon']

            if 'tau_beta' in theta_0:
                self.tau_beta = theta_0['tau_beta']
            else:
                self.tau_beta = (self.pi * self.n_snps) / np.maximum(0.01, 1. - self.sigma_epsilon)

        # Cast all the hyperparameters to conform to the precision set by the user:
        self.sigma_epsilon = np.dtype(self.float_precision).type(self.sigma_epsilon)
        self.tau_beta = np.dtype(self.float_precision).type(self.tau_beta)
        self.pi = np.dtype(self.float_precision).type(self.pi)
        self.lambda_min = np.dtype(self.float_precision).type(self.lambda_min)
        self._sigma_g = np.dtype(self.float_precision).type(0.)

    def initialize_variational_parameters(self, param_0=None):
        """
        Initialize the variational parameters of the model.
        :param param_0: A dictionary of initial values for the variational parameters
        """

        param_0 = param_0 or {}

        self.var_mu = {}
        self.var_tau = {}
        self.var_gamma = {}

        for c, shapes in self.shapes.items():

            # Initialize the variational parameters according to the derived update equations,
            # ignoring correlations between SNPs.
            if 'tau' in param_0:
                self.var_tau[c] = param_0['tau'][c]
            else:
                self.var_tau[c] = (self.Nj[c] / self.sigma_epsilon) + self.tau_beta

            self.var_tau[c] = self.var_tau[c].astype(self.float_precision, order=self.order, copy=False)

            if 'mu' in param_0:
                self.var_mu[c] = param_0['mu'][c].astype(self.float_precision, order=self.order)
            else:
                self.var_mu[c] = np.zeros(shapes, dtype=self.float_precision, order=self.order)

            if 'gamma' in param_0:
                self.var_gamma[c] = param_0['gamma'][c].astype(self.float_precision, order=self.order)
            else:
                pi = self.get_pi(c)
                if isinstance(self.pi, dict):
                    self.var_gamma[c] = pi.astype(self.float_precision, order=self.order)
                else:
                    self.var_gamma[c] = pi*np.ones(shapes, dtype=self.float_precision, order=self.order)

        self.eta = self.compute_eta()
        self.zeta = self.compute_zeta()
        self.eta_diff = {c: np.zeros_like(eta, dtype=self.float_precision) for c, eta in self.eta.items()}
        self.q = {c: np.zeros_like(eta, dtype=self.float_precision) for c, eta in self.eta.items()}
        self._log_var_tau = {c: np.log(self.var_tau[c]) for c in self.var_tau}

    def e_step(self):
        """
        Run the E-Step of the Variational EM algorithm.
        Here, we update the variational parameters for each variant using coordinate
        ascent optimization techniques. The update equations are outlined in
        the Supplementary Material of the following paper:

        > Zabad S, Gravel S, Li Y. Fast and accurate Bayesian polygenic risk modeling with variational inference.
        Am J Hum Genet. 2023 May 4;110(5):741-761. doi: 10.1016/j.ajhg.2023.03.009.
        Epub 2023 Apr 7. PMID: 37030289; PMCID: PMC10183379.
        """

        for c, c_size in self.shapes.items():

            # Get the priors:
            tau_beta = self.get_tau_beta(c)
            pi = self.get_pi(c)

            # Updates for tau variational parameters:
            self.var_tau[c] = (self.Nj[c]*(1. + self.lambda_min) / self.sigma_epsilon) + tau_beta
            np.log(self.var_tau[c], out=self._log_var_tau[c])

            # Compute some quantities that are needed for the per-SNP updates:
            mu_mult = self.Nj[c]/(self.var_tau[c]*self.sigma_epsilon)
            u_logs = np.log(pi) - np.log(1. - pi) + .5*(np.log(tau_beta) - self._log_var_tau[c])

            if self.use_cpp:
                cpp_e_step(self.ld_left_bound[c],
                           self.ld_indptr[c],
                           self.ld_data[c],
                           self.std_beta[c],
                           self.var_gamma[c],
                           self.var_mu[c],
                           self.eta[c],
                           self.q[c],
                           self.eta_diff[c],
                           u_logs,
                           0.5*self.var_tau[c],
                           mu_mult,
                           self.dequantize_scale,
                           self.threads,
                           self.use_blas,
                           self.low_memory)
            else:

                e_step(self.ld_left_bound[c],
                       self.ld_indptr[c],
                       self.ld_data[c],
                       self.std_beta[c],
                       self.var_gamma[c],
                       self.var_mu[c],
                       self.eta[c],
                       self.q[c],
                       self.eta_diff[c],
                       u_logs,
                       0.5*self.var_tau[c],
                       mu_mult,
                       self.threads,
                       self.use_blas,
                       self.low_memory)

        self.zeta = self.compute_zeta()

    def update_pi(self):
        """
        Update the prior probability of a variant being causal, or the proportion of causal variants, `pi`.
        """

        if 'pi' not in self.fix_params:

            # Get the average of the gammas:
            self.pi = dict_mean(self.var_gamma, axis=0)

    def update_tau_beta(self):
        """
        Update the prior precision (inverse variance) for the effect size, `tau_beta`.
        """

        if 'tau_beta' not in self.fix_params:

            # tau_beta estimate:
            self.tau_beta = (self.pi * self.m / dict_sum(self.zeta, axis=0)).astype(self.float_precision)

    def _update_sigma_g(self):
        """
        Update the expectation of the additive genotypic variance, `sigma_g`, under the variational distribution.
        This quantity is equivalent to E_q[B'RB], where B is the vector of effect sizes and R is the LD matrix.
        This quantity is used in the update of the residual variance, `sigma_epsilon` and
        in computing the pseudo-heritability.
        """

        self._sigma_g = np.sum([
            np.sum((1. + self.lambda_min)*self.zeta[c] + np.multiply(self.q[c], self.eta[c]), axis=0)
            for c in self.shapes.keys()
        ], axis=0)

    def update_sigma_epsilon(self):
        """
        Update the global residual variance parameter, `sigma_epsilon`.
        """

        if 'sigma_epsilon' not in self.fix_params:

            sig_eps = 0.

            for c, _ in self.shapes.items():
                sig_eps -= 2.*self.std_beta[c].dot(self.eta[c])

            self.sigma_epsilon = 1. + sig_eps + self._sigma_g

    def m_step(self):
        """
        Run the M-Step of the Variational EM algorithm.
        Here, we update the hyperparameters of the model, by simply calling
        the update functions for each hyperparameter separately.

        """

        self.update_pi()
        self.update_tau_beta()
        self._update_sigma_g()
        self.update_sigma_epsilon()

    def objective(self):
        """
        The optimization objective for the variational inference problem. The objective
        for the VIPRS method is the Evidence Lower-Bound (ELBO) in this case.

        !!! seealso "See Also"
            * [elbo][viprs.model.VIPRS.VIPRS.elbo]


        """
        return self.elbo()

    def elbo(self, sum_axis=None):
        """
        Compute the variational objective, the Evidence Lower-BOund (ELBO),
        from GWAS summary statistics and the reference LD data. This implementation assumes
        that the product of the LD matrix with the current estimate of the effect sizes
        is already computed and stored in the `q` dictionary. If this is not the case,
        we recommend computing q first and then calling this method.

        :param sum_axis: The axis along which to sum the ELBO. If None, the ELBO is returned as a scalar.
        :return: The ELBO of the model.
        """

        # Concatenate the dictionary items for easy computation:
        var_gamma = np.clip(dict_concat(self.var_gamma),
                            a_min=self.float_resolution,
                            a_max=1. - self.float_resolution)
        # The gamma for the null component
        null_gamma = np.clip(1. - dict_concat(self.compute_pip()),
                             a_min=self.float_resolution,
                             a_max=1. - self.float_resolution)
        log_var_tau = dict_concat(self._log_var_tau)

        if isinstance(self.pi, dict):
            pi = dict_concat(self.pi)
            null_pi = dict_concat(self.get_null_pi())
        else:
            pi = self.pi
            null_pi = self.get_null_pi()

        if isinstance(self.tau_beta, dict):
            tau_beta = dict_concat(self.tau_beta)
        else:
            tau_beta = self.tau_beta

        zeta = dict_concat(self.zeta)

        # Initialize the ELBO:
        elbo = 0.

        # -----------------------------------------------
        # (1) Compute the log of the joint density:

        #
        # (1.1) The following terms are an expansion of ||Y - X\beta||^2
        #
        # -N/2log(2pi*sigma_epsilon)
        elbo -= np.log(2 * np.pi * self.sigma_epsilon)

        # -Y'Y/(2*sigma_epsilon), where we assume Y'Y = N
        # + (1./sigma_epsilon)*\beta*(XY), where we assume XY = N\hat{\beta}
        if 'sigma_epsilon' not in self.fix_params:
            # If sigma_epsilon was updated in the M-Step, then this expression would
            # simply evaluate to 1. and there's no point in re-computing it again:
            elbo -= 1.
        else:

            eta = dict_concat(self.eta)
            std_beta = dict_concat(self.std_beta)

            elbo -= (1. / self.sigma_epsilon) * (1. - 2.*std_beta.dot(eta) + self._sigma_g)

        elbo *= 0.5*self.n

        elbo -= np.multiply(var_gamma, np.log(var_gamma) - np.log(pi)).sum(axis=sum_axis)
        elbo -= np.multiply(null_gamma, np.log(null_gamma) - np.log(null_pi)).sum(axis=sum_axis)

        elbo += .5 * np.multiply(var_gamma, 1. - log_var_tau + np.log(tau_beta)).sum(axis=sum_axis)

        if np.isscalar(tau_beta) or len(zeta.shape) > 1:
            elbo -= .5*(tau_beta*zeta).sum(axis=sum_axis)
        else:
            var_mu = dict_concat(self.var_mu)
            var_tau = dict_concat(self.var_tau)

            elbo -= .5*(np.multiply(var_gamma, tau_beta) * (var_mu**2 + 1./var_tau)).sum(axis=sum_axis)

        try:
            if len(elbo) == 1:
                return elbo[0]
            else:
                return elbo
        except TypeError:
            return elbo

    def entropy(self, sum_axis=None):
        """
        Compute the entropy of the variational distribution given the current parameter values.

        :param sum_axis: The axis along which to sum the ELBO. If None, the ELBO is returned as a scalar.
        :return: The entropy of the variational distribution.
        """

        # Concatenate the dictionary items for easy computation:
        var_gamma = np.clip(dict_concat(self.var_gamma),
                            a_min=self.float_resolution,
                            a_max=1. - self.float_resolution)
        # The gamma for the null component
        null_gamma = np.clip(1. - dict_concat(self.compute_pip()),
                             a_min=self.float_resolution,
                             a_max=1. - self.float_resolution)

        log_var_tau = dict_concat(self._log_var_tau)

        entropy = 0.

        # Bernoulli entropy terms:
        entropy -= np.multiply(var_gamma, np.log(var_gamma)).sum(axis=sum_axis)
        entropy -= np.multiply(null_gamma, np.log(null_gamma)).sum(axis=sum_axis)
        # Gaussian entropy terms:
        entropy -= .5 * np.multiply(var_gamma, log_var_tau).sum(axis=sum_axis)

        return .5 * self.n_snps * (np.log(2. * np.pi) + 1.) + entropy

    def loglikelihood(self):
        """
        Compute the expectation of the loglikelihood of the data given the current model parameter values.
        The expectation is taken with respect to the variational distribution.

        :return: The loglikelihood of the data.
        """

        eta = dict_concat(self.eta)
        std_beta = dict_concat(self.std_beta)

        return -0.5*self.n*(np.log(2.*np.pi*self.sigma_epsilon) +
                            (1./self.sigma_epsilon)*(1. - 2.*std_beta.dot(eta) + self._sigma_g))

    def mse(self, sum_axis=None):
        """
        Compute a summary statistics-based estimate of the mean squared error on the training set.

        :param sum_axis: The axis along which to sum the MSE.
        If None, the MSE is returned as a scalar.
        :return: The mean squared error.
        """

        eta = dict_concat(self.eta)
        std_beta = dict_concat(self.std_beta)
        zeta = dict_concat(self.zeta)

        return 1. - 2.*std_beta.dot(eta) + (
                self._sigma_g - zeta.sum(axis=sum_axis) + (eta**2).sum(axis=sum_axis)
        )

    def log_prior(self, sum_axis=None):
        """
        Compute the expectation of the log prior of the model parameters given the current hyperparameter values.
        The expectation is taken with respect to the variational distribution.

        :param sum_axis: The axis along which to sum the log prior.
        :return: The expectation of the log prior according to the variational density.
        """

        var_gamma = np.clip(dict_concat(self.var_gamma),
                            a_min=self.float_resolution,
                            a_max=1. - self.float_resolution)
        # The gamma for the null component
        null_gamma = np.clip(1. - dict_concat(self.compute_pip()),
                             a_min=self.float_resolution,
                             a_max=1. - self.float_resolution)

        if isinstance(self.pi, dict):
            pi = dict_concat(self.pi)
            null_pi = dict_concat(self.get_null_pi())
        else:
            pi = self.pi
            null_pi = self.get_null_pi()

        if isinstance(self.tau_beta, dict):
            tau_beta = dict_concat(self.tau_beta)
        else:
            tau_beta = self.tau_beta

        zeta = dict_concat(self.zeta)

        log_prior = 0.

        log_prior += .5*(np.multiply(var_gamma, np.log(tau_beta))).sum(axis=sum_axis)
        log_prior += np.multiply(var_gamma, np.log(pi)).sum(axis=sum_axis)
        log_prior += np.multiply(null_gamma, np.log(null_pi)).sum(axis=sum_axis)

        if np.isscalar(tau_beta) or len(zeta.shape) > 1:
            log_prior -= (.5*tau_beta * zeta).sum(axis=sum_axis)
        else:
            var_mu = dict_concat(self.var_mu)
            var_tau = dict_concat(self.var_tau)

            log_prior -= .5 * (np.multiply(var_gamma, tau_beta) * (var_mu ** 2 + 1. / var_tau)).sum(axis=sum_axis)

        return log_prior - .5*self.n_snps*np.log(2.*np.pi)

    def get_sigma_epsilon(self):
        """
        :return: The value of the residual variance, `sigma_epsilon`.
        """
        return self.sigma_epsilon

    def get_tau_beta(self, chrom=None):
        """
        :param chrom: Get the value of `tau_beta` for a given chromosome.

        :return: The value of the prior precision on the effect size(s), `tau_beta`
        """
        if chrom is None:
            return self.tau_beta
        else:
            if isinstance(self.tau_beta, dict):
                return self.tau_beta[chrom]
            else:
                return self.tau_beta

    def get_pi(self, chrom=None):
        """
        :param chrom: Get the value of `pi` for a given chromosome.

        :return: The value of the prior probability of a variant being causal, `pi`.
        """

        if chrom is None:
            return self.pi
        else:
            if isinstance(self.pi, dict):
                return self.pi[chrom]
            else:
                return self.pi

    def get_null_pi(self, chrom=None):
        """
        :param chrom: If provided, get the mixing proportion for the null component on a given chromosome.

        :return: The value of the prior probability of a variant being null, `1 - pi`.
        """

        pi = self.get_pi(chrom=chrom)

        if isinstance(pi, dict):
            return {c: 1. - c_pi for c, c_pi in pi.items()}
        else:
            return 1. - pi

    def get_proportion_causal(self):
        """
        :return: The proportion of causal variants in the model.
        """
        if isinstance(self.pi, dict):
            return dict_mean(self.pi, axis=0)
        else:
            return self.pi

    def get_average_effect_size_variance(self):
        """
        :return: The average per-SNP variance for the prior mixture components
        """
        if isinstance(self.pi, dict):
            pi = dict_concat(self.pi, axis=0)
        else:
            pi = self.pi

        if isinstance(self.tau_beta, dict):
            tau_beta = dict_concat(self.tau_beta, axis=0)
        else:
            tau_beta = self.tau_beta

        return np.sum(pi / tau_beta, axis=0)

    def get_heritability(self):
        """
        :return: An estimate of the SNP heritability, or proportion of variance explained by SNPs.
        """

        return self._sigma_g / (self._sigma_g + self.sigma_epsilon)

    def to_theta_table(self):
        """
        :return: A `pandas` DataFrame containing information about the estimated hyperparameters of the model.
        """

        theta_table = [
            {'Parameter': 'Residual_variance', 'Value': self.sigma_epsilon},
            {'Parameter': 'Heritability', 'Value': self.get_heritability()},
            {'Parameter': 'Proportion_causal', 'Value': self.get_proportion_causal()},
            {'Parameter': 'Average_effect_variance', 'Value': self.get_average_effect_size_variance()},
        ]

        if np.isscalar(self.lambda_min):
            theta_table += [
                {'Parameter': 'Lambda_min', 'Value': self.lambda_min}
            ]

        if isinstance(self.tau_beta, dict):
            taus = dict_mean(self.tau_beta, axis=0)
        else:
            taus = self.tau_beta

        try:
            taus = list(taus)
            for i in range(len(taus)):
                theta_table.append({'Parameter': f'tau_beta_{i+1}', 'Value': taus[i]})
        except TypeError:
            theta_table.append({'Parameter': 'tau_beta', 'Value': taus})

        return pd.DataFrame(theta_table)

    def to_history_table(self):
        """
        :return: A `pandas` DataFrame containing the history of tracked parameters as a function of
        the number of iterations.
        """
        return pd.DataFrame(self.history)

    def write_inferred_theta(self, f_name, sep="\t"):
        """
        A convenience method to write the inferred (and fixed) hyperparameters of the model to file.
        :param f_name: The file name
        :param sep: The separator for the hyperparameter file.
        """

        # Write the table to file:
        try:
            self.to_theta_table().to_csv(f_name, sep=sep, index=False)
        except Exception as e:
            raise e

    def update_theta_history(self):
        """
        A convenience method to update the history of the hyperparameters/objectives/other summary statistics
        of the model, if the user requested that they should be tracked.
        """

        self.history['ELBO'].append(self.elbo())

        for tt in self.tracked_params:
            if tt == 'pi':
                self.history['pi'].append(self.get_proportion_causal())
            elif tt == 'pis':
                self.history['pis'].append(self.pi)
            if tt == 'heritability':
                self.history['heritability'].append(self.get_heritability())
            if tt == 'sigma_epsilon':
                self.history['sigma_epsilon'].append(self.sigma_epsilon)
            elif tt == 'tau_beta':
                self.history['tau_beta'].append(self.tau_beta)
            elif tt == 'sigma_g':
                self.history['sigma_g'].append(self._sigma_g)
            elif tt == 'entropy':
                self.history['entropy'].append(self.entropy())
            elif tt == 'loglikelihood':
                self.history['loglikelihood'].append(self.loglikelihood())
            elif tt == 'mse':
                self.history['mse'].append(self.mse())
            elif tt == 'max_eta_diff':
                self.history['max_eta_diff'].append(np.max([
                    np.max(np.abs(diff)) for diff in self.eta_diff.values()
                ]))
            elif callable(tt):
                self.history[tt.__name__].append(tt(self))

    def compute_pip(self):
        """
        :return: The posterior inclusion probability
        """
        return self.var_gamma.copy()

    def compute_eta(self):
        """
        :return: The mean for the effect size under the variational posterior.
        """
        return {c: v*self.var_mu[c] for c, v in self.var_gamma.items()}

    def compute_zeta(self):
        """
        :return: The expectation of the squared effect size under the variational posterior.
        """
        return {c: np.multiply(v, self.var_mu[c]**2 + 1./self.var_tau[c])
                for c, v in self.var_gamma.items()}

    def update_posterior_moments(self):
        """
        A convenience method to update the dictionaries containing the posterior moments,
        including the PIP and posterior mean and variance for the effect size.
        """

        self.pip = {c: pip.copy() for c, pip in self.compute_pip().items()}
        self.post_mean_beta = {c: eta.copy() for c, eta in self.eta.items()}
        self.post_var_beta = {c: zeta - self.eta[c]**2 for c, zeta in self.zeta.items()}

    def fit(self,
            max_iter=1000,
            theta_0=None,
            param_0=None,
            continued=False,
            min_iter=5,
            f_abs_tol=1e-6,
            x_abs_tol=1e-6,
            drop_r_tol=0.01,
            patience=5):
        """
        A convenience method to fit the model using the Variational EM algorithm.

        :param max_iter: Maximum number of iterations. 
        :param theta_0: A dictionary of values to initialize the hyperparameters
        :param param_0: A dictionary of values to initialize the variational parameters
        :param continued: If true, continue the model fitting for more iterations from current parameters
        instead of starting over.
        :param min_iter: The minimum number of iterations to run before checking for convergence.
        :param f_abs_tol: The absolute tolerance threshold for the objective (ELBO).
        :param x_abs_tol: The absolute tolerance threshold for the variational parameters.
        :param drop_r_tol: The relative tolerance for the drop in the ELBO to be considered as a red flag. It usually
        happens around convergence that the objective fluctuates due to numerical errors. This is a way to
        differentiate such random fluctuations from actual drops in the objective.
        :param patience: The maximum number of times the objective is allowed to drop before termination.
        """

        if not continued:
            self.initialize(theta_0, param_0)
            start_idx = 1
            self.update_theta_history()
        else:
            start_idx = len(self.history['ELBO']) + 1
            # Update OptimizeResult object to enable continuation of the optimization:
            self.optim_result.update(self.history['ELBO'][-1], increment=False)

        if int(self.verbose) > 1:
            logging.debug("> Performing model fit...")
            if self.threads > 1:
                logging.debug(f"> Using up to {self.threads} threads.")

        # If the model is fit over a single chromosome, append this information to the
        # tqdm progress bar:
        if len(self.shapes) == 1:
            chrom, num_snps = list(self.shapes.items())[0]
            desc = f"Chromosome {chrom} ({num_snps} variants)"
        else:
            desc = None

        # Progress bar:
        pbar = tqdm(range(start_idx, start_idx + max_iter),
                    disable=not self.verbose,
                    desc=desc)

        if continued:
            prev_elbo = self.history['ELBO'][-1]
        else:
            prev_elbo = -np.inf

        # -------------------------- Main optimization loop (EM Algorithm) --------------------------

        for i in pbar:

            if self.optim_result.stop_iteration:
                pbar.set_postfix({'Final ELBO': f"{self.optim_result.objective:.4f}"})
                pbar.n = i - 1
                pbar.total = i - 1
                pbar.refresh()
                pbar.close()
                break

            # Perform parameter updates (E-Step + M-Step):
            self.e_step()
            self.m_step()

            # Update the tracked parameters (including objectives):
            self.update_theta_history()

            # Update the progress bar:
            pbar.set_postfix({'ELBO': f"{self.history['ELBO'][-1]:.4f}"})

            # ------------------------------------------------------------
            # Sanity checking / convergence criteria:

            curr_elbo = self.history['ELBO'][-1]

            # Check if the objective / model parameters behave in unexpected/pathological ways:
            if self.mse() < 0.:
                if 'sigma_epsilon' not in self.fix_params:
                    logging.warning(f"Iteration {i} | MSE is negative; Restarting optimization "
                                    f"and fixing residual variance hyperparameter (sigma_epsilon).")
                    import copy
                    hist = copy.deepcopy(self.history)
                    self.initialize(theta_0, param_0)
                    self.history = hist
                    prev_elbo = -np.inf
                    self.fix_params['sigma_epsilon'] = self.sigma_epsilon = .95
                    continue
                else:
                    raise OptimizationDivergence(f"Stopping at iteration {i}: "
                                                 f"The optimization algorithm is not converging!\n"
                                                 f"The MSE is negative ({self.mse():.6f}). "
                                                 f"This usually indicates poor agreement between "
                                                 f"summary statistics and LD reference panel.")

            elif not np.isfinite(curr_elbo):
                raise OptimizationDivergence(f"Stopping at iteration {i}: "
                                             f"The optimization algorithm is not converging!\n"
                                             f"The objective (ELBO) is undefined ({curr_elbo}).")
            elif self.sigma_epsilon < 0.:
                raise OptimizationDivergence(f"Stopping at iteration {i}: "
                                             f"The optimization algorithm is not converging!\n"
                                             f"The residual variance estimate is negative.")
            elif (i > min_iter) and self.optim_result.oscillation_counter >= 3:

                if self.threads > 1:
                    logging.warning(f"Iteration {i} | Reducing the number of "
                                    f"threads for better parameter synchronization.")
                    self.threads -= 1
                    self.optim_result._reset_oscillation_counter()
                else:
                    self.optim_result.update(curr_elbo,
                                             stop_iteration=True,
                                             success=True,
                                             message='Objective converged successfully '
                                                     '(with some minor oscillations).')

            elif self.get_heritability() > 1. or self.get_heritability() < 0.:
                raise OptimizationDivergence(f"Stopping at iteration {i}: "
                                             f"The optimization algorithm is not converging!\n"
                                             f"Value of estimated heritability is out of bounds.")

            # Check for convergence in the objective + parameters:
            elif (i > min_iter) and np.isclose(prev_elbo, curr_elbo, atol=f_abs_tol, rtol=0.):
                self.optim_result.update(curr_elbo,
                                         stop_iteration=True,
                                         success=True,
                                         message='Objective (ELBO) converged successfully.')
            elif (i > min_iter) and max([np.max(np.abs(diff)) for diff in self.eta_diff.values()]) < x_abs_tol:
                self.optim_result.update(curr_elbo,
                                         stop_iteration=True,
                                         success=True,
                                         message='Variational parameters converged successfully.')

            # Check to see if the objective drops significantly due to numerical instabilities:
            elif curr_elbo < prev_elbo and not np.isclose(curr_elbo, prev_elbo, atol=0., rtol=drop_r_tol):
                patience -= 1

                if patience == 0:
                    raise OptimizationDivergence(f"Stopping at iteration {i}: "
                                                 f"The optimization algorithm is not converging!\n"
                                                 f"Objective (ELBO) is decreasing.")
                else:
                    self.optim_result.update(curr_elbo)

            else:
                self.optim_result.update(curr_elbo)

            prev_elbo = curr_elbo

        # -------------------------- Post processing / cleaning up / model checking --------------------------

        # Update the posterior moments:
        self.update_posterior_moments()

        # Inspect the optim result:
        if not self.optim_result.stop_iteration:
            self.optim_result.update(self.history['ELBO'][-1],
                                     stop_iteration=True,
                                     success=False,
                                     message="Maximum iterations reached without convergence.\n"
                                             "You may need to run the model for more iterations.",
                                     increment=False)

        # Inform the user about potential issues:
        if not self.optim_result.success:
            logging.warning("\t" + self.optim_result.message)

        logging.debug(f"> Final ELBO: {self.history['ELBO'][-1]:.6f}")
        logging.debug(f"> Estimated heritability: {self.get_heritability():.6f}")
        logging.debug(f"> Estimated proportion of causal variants: {self.get_proportion_causal():.6f}")

        return self

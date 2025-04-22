import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from .BayesPRSModel import BayesPRSModel
from .vi.e_step_cpp import cpp_e_step
from ..utils.OptimizeResult import OptimizeResult, IterationConditionCounter
from ..utils.compute_utils import dict_mean, dict_sum, dict_concat
from magenpy.utils.compute_utils import is_numeric

# Set up the logger:
import logging
logger = logging.getLogger(__name__)


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
    :ivar n_per_snp: A dictionary of the sample size per SNP from the GWAS study.
    :ivar threads: The number of threads to use when fitting the model.
    :ivar fix_params: A dictionary of hyperparameters with their fixed values.
    :ivar float_precision: The precision of the floating point variables. Options are: 'float32' or 'float64'.
    :ivar order: The order of the arrays in memory. Options are: 'C' or 'F'.
    :ivar low_memory: A boolean flag to indicate whether to use low memory mode.
    :ivar dequantize_on_the_fly: A boolean flag to indicate whether to dequantize the LD matrix on the fly.
    :ivar optim_result: An instance of OptimizeResult tracking the progress of the optimization algorithm.
    :ivar history: A dictionary to store the history of the optimization procedure (e.g. the objective as a function
    of iteration number).
    :ivar tracked_params: A list of hyperparameters to track throughout the optimization procedure. Useful for
    debugging/model checking.

    """

    def __init__(self,
                 gdl,
                 fix_params=None,
                 tracked_params=None,
                 lambda_min=None,
                 float_precision='float32',
                 order='F',
                 low_memory=True,
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
            * User may also provide arbitrary functions that take the `VIPRS` object as input and
            compute any quantity of interest from it.

        :param lambda_min: The minimum eigenvalue for the LD matrix (or an approximation of it that will serve
        as a regularizer). If set to 'infer', the minimum eigenvalue will be computed or retrieved from the LD matrix.
        :param float_precision: The precision of the floating point variables. Options are: 'float32' or 'float64'.
        :param order: The order of the arrays in memory. Options are: 'C' or 'F'.
        :param low_memory: A boolean flag to indicate whether to use low memory mode.
        :param dequantize_on_the_fly: A boolean flag to indicate whether to dequantize the LD matrix on the fly.
        :param threads: The number of threads to use when fitting the model.
        """

        super().__init__(gdl, float_precision=float_precision)

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

        logger.debug("> Loading LD matrices to memory")

        for c, ld_mat in self.gdl.get_ld_matrices().items():

            # Determine how to load the LD data:
            if dequantize_on_the_fly and np.issubdtype(ld_mat.stored_dtype, np.integer):
                dtype = ld_mat.stored_dtype
            else:

                if dequantize_on_the_fly:
                    logger.debug("Dequantization on the fly is only supported for "
                                 "integer data types. Ignoring this flag.")

                dtype = float_precision
                dequantize_on_the_fly = False

            ld_lop = ld_mat.load(return_symmetric=not low_memory, dtype=dtype)

            # Load the LD data:
            self.ld_data[c] = ld_lop.ld_data
            self.ld_indptr[c] = ld_lop.ld_indptr
            self.ld_left_bound[c] = ld_lop.leftmost_idx

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

        # ---------- General properties: ----------

        self.threads = threads
        self.fix_params = fix_params or {}

        self.order = order
        self.low_memory = low_memory

        self.dequantize_on_the_fly = dequantize_on_the_fly

        if self.dequantize_on_the_fly:
            info = np.iinfo(self.ld_data[self.chromosomes[0]].dtype)
            self.dequantize_scale = 1. / info.max
        else:
            self.dequantize_scale = 1.

        self.optim_result = OptimizeResult()
        self.history = {}
        self.tracked_params = tracked_params or []

    def initialize(self, theta_0=None, param_0=None):
        """
        A convenience method to initialize all the objects associated with the model.
        :param theta_0: A dictionary of initial values for the hyperparameters theta
        :param param_0: A dictionary of initial values for the variational parameters
        """

        logger.debug("> Initializing model parameters")

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

            min_pi = max(10./self.n_snps, 1e-5)
            max_pi = min(0.2, 1e4/self.n_snps)

            self.pi = np.random.uniform(low=min_pi, high=max_pi)
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
                    naive_h2g = np.clip(simple_ldsc(self.gdl), a_min=.01, a_max=.99)
                except Exception as e:
                    logger.debug(e)
                    naive_h2g = np.random.uniform(low=.01, high=.1)

                self.sigma_epsilon = 1. - naive_h2g
                self.tau_beta = self.pi * self.n_snps / max(naive_h2g, 0.01)
            else:

                # If tau_beta is given, use it to initialize sigma_epsilon

                self.tau_beta = theta_0['tau_beta']
                self.sigma_epsilon = np.clip(1. - (self.pi * self.n_snps / self.tau_beta),
                                             a_min=1e-4,
                                             a_max=1. - 1e-4)
        else:

            # If sigma_epsilon is given, use it in the initialization

            self.sigma_epsilon = theta_0['sigma_epsilon']

            if 'tau_beta' in theta_0:
                self.tau_beta = theta_0['tau_beta']
            else:
                self.tau_beta = (self.pi * self.n_snps) / np.maximum(0.01, 1. - self.sigma_epsilon)

        # Cast all the hyperparameters to conform to the precision set by the user:
        self.sigma_epsilon = np.dtype(self.float_precision).type(self.sigma_epsilon)
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
                self.var_tau[c] = (self.n_per_snp[c] / self.sigma_epsilon) + self.tau_beta

            self.var_tau[c] = self.var_tau[c]

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

    def set_fixed_params(self, fix_params):
        """
        Set the fixed hyperparameters of the model.
        :param fix_params: A dictionary of hyperparameters with their fixed values.
        """

        assert isinstance(fix_params, dict), "The fixed parameters must be provided as a dictionary."

        self.fix_params.update(fix_params)

        for key, val in fix_params.items():
            if key == 'sigma_epsilon':
                self.sigma_epsilon = np.dtype(self.float_precision).type(val)
            elif key == 'tau_beta':
                self.tau_beta = np.dtype(self.float_precision).type(val)
            elif key == 'pi':
                self.pi = np.dtype(self.float_precision).type(val)
            elif key == 'lambda_min':
                self.lambda_min = np.dtype(self.float_precision).type(val)

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
            self.var_tau[c] = (self.n_per_snp[c]*(1. + self.lambda_min) / self.sigma_epsilon) + tau_beta
            np.log(self.var_tau[c], out=self._log_var_tau[c])

            # Compute some quantities that are needed for the per-SNP updates:
            mu_mult = (self.n_per_snp[c]/(self.var_tau[c]*self.sigma_epsilon)).astype(self.float_precision)
            u_logs = (np.log(pi) - np.log(1. - pi) + .5*(np.log(tau_beta) -
                                                         self._log_var_tau[c])).astype(self.float_precision)

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
                       np.sqrt(0.5*self.var_tau[c]).astype(self.float_precision),
                       mu_mult,
                       self.dequantize_scale,
                       self.threads,
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
            self.tau_beta = (self.pi * self.n_snps / dict_sum(self.zeta, axis=0))

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

        double_resolution = np.finfo(np.float64).resolution

        # Concatenate the dictionary items for easy computation:
        var_gamma = np.clip(dict_concat(self.var_gamma).astype(np.float64),
                            a_min=double_resolution,
                            a_max=1. - double_resolution)
        # The gamma for the null component
        null_gamma = np.clip(1. - dict_concat(self.compute_pip()).astype(np.float64),
                             a_min=double_resolution,
                             a_max=1. - double_resolution)
        log_var_tau = dict_concat(self._log_var_tau)

        if isinstance(self.pi, dict):
            pi = dict_concat(self.pi)
            null_pi = dict_concat(self.get_null_pi())
        else:
            pi = self.pi
            null_pi = self.get_null_pi()

        if isinstance(self.tau_beta, dict):
            tau_beta = dict_concat(self.tau_beta).astype(np.float64)
        else:
            tau_beta = self.tau_beta

        zeta = dict_concat(self.zeta).astype(np.float64)

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

            eta = dict_concat(self.eta).astype(np.float64)
            std_beta = dict_concat(self.std_beta).astype(np.float64)

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

        double_resolution = np.finfo(np.float64).resolution

        # Concatenate the dictionary items for easy computation:
        var_gamma = np.clip(dict_concat(self.var_gamma),
                            a_min=double_resolution,
                            a_max=1. - double_resolution)
        # The gamma for the null component
        null_gamma = np.clip(1. - dict_concat(self.compute_pip()),
                             a_min=double_resolution,
                             a_max=1. - double_resolution)

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

        return -0.5*self.n*(
                np.log(2.*np.pi*self.sigma_epsilon) +
                (1./self.sigma_epsilon)*(1. - 2.*std_beta.dot(eta) + self._sigma_g)
        )

    def log_prior(self, sum_axis=None):
        """
        Compute the expectation of the log prior of the model parameters given the current hyperparameter values.
        The expectation is taken with respect to the variational distribution.

        :param sum_axis: The axis along which to sum the log prior.
        :return: The expectation of the log prior according to the variational density.
        """

        double_resolution = np.finfo(np.float64).resolution

        var_gamma = np.clip(dict_concat(self.var_gamma),
                            a_min=double_resolution,
                            a_max=1. - double_resolution)
        # The gamma for the null component
        null_gamma = np.clip(1. - dict_concat(self.compute_pip()),
                             a_min=double_resolution,
                             a_max=1. - double_resolution)

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

    def complete_loglikelihood(self):
        """
        Compute the complete loglikelihood of the data given the current model parameter values.
        The complete loglikelihood is the sum of the loglikelihood and the log prior.

        :return: The complete loglikelihood of the data.
        """

        return self.loglikelihood() + self.log_prior()

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
            {'Parameter': 'ELBO', 'Value': self.elbo()},
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
            elif tt == 'log_prior':
                self.history['log_prior'].append(self.log_prior())
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
        :return: The posterior inclusion probability (PIP) of
        each variant under the variational posterior.
        """
        return self.var_gamma.copy()

    def compute_eta(self):
        """
        :return: The mean for the effect size under the variational posterior.
        """
        return {c: v*self.var_mu[c] for c, v in self.var_gamma.items()}

    def compute_zeta(self):
        """

        .. note:: Due to the small magnitude of the variational parameters, we only store
        zeta using double precision.

        :return: The expectation of the squared effect size under the variational posterior.
        """
        return {c: np.multiply(v, self.var_mu[c].astype(np.float64)**2 + 1./self.var_tau[c].astype(np.float64))
                for c, v in self.var_gamma.items()}

    def update_posterior_moments(self):
        """
        A convenience method to update the dictionaries containing the posterior moments,
        including the PIP and posterior mean and variance for the effect size.
        """

        self.pip = self.compute_pip()
        self.post_mean_beta = {c: eta.copy() for c, eta in self.eta.items()}
        self.post_var_beta = {c: zeta - self.eta[c]**2 for c, zeta in self.zeta.items()}

    def fit(self,
            max_iter=1000,
            theta_0=None,
            param_0=None,
            continued=False,
            disable_pbar=False,
            min_iter=3,
            f_abs_tol=1e-6,
            x_abs_tol=1e-6,
            patience=10,
            **kwargs):
        """
        A convenience method to fit the model using the Variational EM algorithm.

        :param max_iter: Maximum number of iterations. 
        :param theta_0: A dictionary of values to initialize the hyperparameters
        :param param_0: A dictionary of values to initialize the variational parameters
        :param continued: If true, continue the model fitting for more iterations from current parameters
        instead of starting over.
        :param disable_pbar: If True, disable the progress bar.
        :param min_iter: The minimum number of iterations to run before checking for convergence.
        :param f_abs_tol: The absolute tolerance threshold for the objective (ELBO).
        :param x_abs_tol: The absolute tolerance threshold for the variational parameters.
        :param patience: The maximum number of consecutive iterations with no improvement in the ELBO or change
        in model parameters.
        :param kwargs: Additional keyword arguments to pass to the optimization routine.

        :return: The VIPRS object with the fitted model parameters.
        """

        if not continued:
            self.initialize(theta_0, param_0)
            start_idx = 1
            self.update_theta_history()
        else:
            start_idx = len(self.history['ELBO']) + 1
            # Update OptimizeResult object to enable continuation of the optimization:
            self.optim_result.update(self.elbo(), increment=False)

        logger.info("> Performing model fit...")
        if self.threads > 1:
            logger.info(f"> Using up to {self.threads} threads.")

        # If the model is fit over a single chromosome, append this information to the
        # tqdm progress bar:
        if len(self.shapes) == 1:
            desc = f"Chromosome {self.chromosomes[0]} ({self.n_snps} variants)"
        else:
            desc = None

        if continued:
            prev_elbo = self.elbo()
        else:
            prev_elbo = -np.inf

        # The following is used to track LD-weighted effect sizes.
        # This is useful for tracking oscillations in ultra high-dimensions due to high LD.
        prev_sigma_g = self._sigma_g
        sigma_g_icc = IterationConditionCounter()
        divergence_icc = IterationConditionCounter()

        # -------------------------- Main optimization loop (EM Algorithm) --------------------------

        with (logging_redirect_tqdm(loggers=[logger])):

            # Progress bar:
            pbar = tqdm(range(start_idx, start_idx + max_iter),
                        disable=disable_pbar,
                        desc=desc)

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

                # Compute maximum absolute difference in effect sizes:
                max_eta_diff = max([np.max(np.abs(diff)) for diff in self.eta_diff.values()])

                # Update the current ELBO:
                curr_elbo = self.history['ELBO'][-1]

                # Update the sigma_g condition counter:
                sigma_g_icc.update(
                    (i > min_iter) and
                    np.isclose(self._sigma_g, prev_sigma_g, atol=x_abs_tol, rtol=0.) and
                    max_eta_diff < x_abs_tol * 10,
                    i
                )

                # Update the ELBO drop condition counter:
                # TODO: Find other ways to determine if the optimization is diverging.
                divergence_icc.update(
                    (curr_elbo < prev_elbo) and not
                    np.isclose(curr_elbo, prev_elbo, atol=1e3*f_abs_tol, rtol=1e-4),
                    i
                )

                # Update the progress bar:
                pbar.set_postfix({'ELBO': f"{curr_elbo:.4f}"})

                # --------------------------------------------------------------------------------------
                # Sanity checking / convergence criteria:

                # Check if the objective / model parameters behave in unexpected/pathological ways:
                if self.mse() < 0.:

                    if 'sigma_epsilon' not in self.fix_params:

                        logger.info(f"Iteration {i} | MSE is negative; Restarting optimization "
                                    f"and fixing residual variance hyperparameter (sigma_epsilon).")

                        self.initialize_theta(theta_0)
                        self.initialize_variational_parameters(param_0)

                        # Set the residual variance to a fixed value for now:
                        self.fix_params['sigma_epsilon'] = self.sigma_epsilon = .95

                        continue

                    else:
                        self.optim_result.update(curr_elbo,
                                                 stop_iteration=True,
                                                 success=False,
                                                 message=f'The MSE is negative ({self.mse():.6f}).')

                elif not np.isfinite(curr_elbo):
                    self.optim_result.update(curr_elbo,
                                             stop_iteration=True,
                                             success=False,
                                             message='Objective (ELBO) is undefined.')
                elif self.sigma_epsilon < 0.:
                    self.optim_result.update(curr_elbo,
                                             stop_iteration=True,
                                             success=False,
                                             message='Residual variance estimate is negative.')
                elif self.threads > 1 and self.optim_result.oscillation_counter > 5:

                    logger.info(f"Iteration {i} | Reducing the number of "
                                f"threads for better parameter synchronization.")
                    self.threads -= 1
                    self.optim_result._reset_oscillation_counter()

                elif self.get_heritability() > 1. or self.get_heritability() < 0.:

                    self.optim_result.update(curr_elbo,
                                             stop_iteration=True,
                                             success=False,
                                             message='Estimated heritability is out of bounds.')

                # Check for convergence in the objective + parameters:
                elif (i > min_iter) and np.isclose(prev_elbo, curr_elbo, atol=f_abs_tol, rtol=0.):
                    self.optim_result.update(curr_elbo,
                                             stop_iteration=True,
                                             success=True,
                                             message='Objective (ELBO) converged successfully.')

                elif (i > min_iter) and max_eta_diff < x_abs_tol:
                    self.optim_result.update(curr_elbo,
                                             stop_iteration=True,
                                             success=True,
                                             message='Variational parameters converged successfully.')
                # Check for convergence based on the LD-weighted effect sizes:
                elif sigma_g_icc.counter > patience:
                    self.optim_result.update(curr_elbo,
                                             stop_iteration=True,
                                             success=True,
                                             message='LD-weighted variational parameters converged successfully.')
                # Check if the ELBO has been consistently dropping:
                elif divergence_icc.counter > patience:

                    self.optim_result.update(curr_elbo,
                                             stop_iteration=True,
                                             success=False,
                                             message='The objective (ELBO) is decreasing.')

                else:
                    self.optim_result.update(curr_elbo)

                prev_elbo = curr_elbo
                prev_sigma_g = self._sigma_g

        # -------------------------- Post processing / cleaning up / model checking --------------------------

        # Update the posterior moments:
        self.update_posterior_moments()

        # Inspect the optim result:
        if not self.optim_result.stop_iteration:
            self.optim_result.update(self.elbo(),
                                     stop_iteration=True,
                                     success=False,
                                     message="Maximum iterations reached without convergence.\n"
                                             "You may need to run the model for more iterations.",
                                     increment=False)

        # Inform the user about potential issues:
        if not self.optim_result.success:
            logger.warning("\t" + self.optim_result.message)

        logger.info(f"> Final ELBO: {self.history['ELBO'][-1]:.6f}")
        logger.info(f"> Estimated heritability: {self.get_heritability():.6f}")
        logger.info(f"> Estimated proportion of causal variants: {self.get_proportion_causal():.6f}")

        return self

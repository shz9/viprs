import numpy as np
import pandas as pd
import os.path as osp

from magenpy import GWADataLoader
from ..utils.compute_utils import expand_column_names, dict_max

# Set up the logger:
import logging
logger = logging.getLogger(__name__)


class BayesPRSModel:
    """
    A base class for Bayesian PRS models. This class defines the basic structure and methods
    that are common to most Bayesian PRS models. Specifically, this class provides methods and interfaces
    for initialization, harmonization, prediction, and fitting of Bayesian PRS models.

    The class is generic is designed to be inherited and extended by
    specific Bayesian PRS models, such as `LDPred` and `VIPRS`.

    :ivar gdl: A GWADataLoader object containing harmonized GWAS summary statistics and
    Linkage-Disequilibrium (LD) matrices.
    :ivar float_precision: The precision of the floating point variables. Options are: 'float32' or 'float64'.
    :ivar shapes: A dictionary where keys are chromosomes and values are the shapes of the variant arrays
    (e.g. the number of variants per chromosome).
    :ivar n_per_snp: A dictionary where keys are chromosomes and values are the sample sizes per variant.
    :ivar std_beta: A dictionary of the standardized marginal effect sizes from GWAS.
    :ivar validation_std_beta: A dictionary of the standardized marginal effect sizes
    from an independent validation set.
    :ivar _sample_size: The maximum per-SNP sample size.
    :ivar pip: The posterior inclusion probability.
    :ivar post_mean_beta: The posterior mean for the effect sizes.
    :ivar post_var_beta: The posterior variance for the effect sizes.
    """

    def __init__(self,
                 gdl: GWADataLoader,
                 float_precision='float32'):
        """
        Initialize the Bayesian PRS model.
        :param gdl: An instance of `GWADataLoader`. Must contain either GWAS summary statistics
        or genotype data.
        :param float_precision: The precision for the floating point numbers.
        """

        # ------------------- Sanity checks -------------------

        assert isinstance(gdl, GWADataLoader), "The `gdl` object must be an instance of GWASDataLoader."

        assert gdl.genotype is not None or (gdl.ld is not None and gdl.sumstats_table is not None), (
            "The GWADataLoader object must contain either genotype data or summary statistics and LD matrices."
        )

        # -----------------------------------------------------

        self.gdl = gdl
        self.float_precision = float_precision
        self.float_eps = np.finfo(self.float_precision).eps
        self.shapes = self.gdl.shapes.copy()

        # Placeholder for sample size per SNP:
        self.n_per_snp = None
        # Placeholder for standardized marginal betas:
        self.std_beta = None

        # Placeholder for standardized marginal betas from an independent validation set:
        self.validation_std_beta = None

        if gdl.sumstats_table is not None:
            # Initialize the input data arrays:
            self.initialize_input_data_arrays()

            # Determine the overall sample size:
            self._sample_size = dict_max(self.n_per_snp)

        # Placeholder for inferred model parameters:
        self.pip = None  # Posterior inclusion probability
        self.post_mean_beta = None  # The posterior mean for the effect sizes
        self.post_var_beta = None  # The posterior variance for the effect sizes

    @property
    def chromosomes(self):
        """
        :return: The list of chromosomes that are included in the BayesPRSModel
        """
        return sorted(list(self.shapes.keys()))

    @property
    def m(self) -> int:
        """

        !!! seealso "See Also"
            * [n_snps][viprs.model.BayesPRSModel.BayesPRSModel.n_snps]

        :return: The number of variants in the model.
        """
        return self.gdl.m

    @property
    def n(self) -> int:
        """
        :return: The number of samples in the model. If not available, average the per-SNP
        sample sizes.
        """
        return self._sample_size

    @property
    def n_snps(self) -> int:
        """
        !!! seealso "See Also"
            * [m][viprs.model.BayesPRSModel.BayesPRSModel.m]

        :return: The number of SNPs in the model.
        """
        return self.m

    def initialize_input_data_arrays(self):
        """
        Initialize the input data arrays for the Bayesian PRS model.
        The input data for summary statistics-based PRS models typically include the following:
            * The sample size per variant (n_per_snp)
            * The standardized marginal betas (std_beta)
            * LD matrices (LD)

        This convenience method initializes the first two inputs, primarily the sample size per variant
        and the standardized marginal betas.
        """

        logger.debug("> Initializing the input data arrays (marginal statistics).")

        try:
            self.n_per_snp = {c: ss.n_per_snp
                              for c, ss in self.gdl.sumstats_table.items()}
            self.std_beta = {c: ss.get_snp_pseudo_corr().astype(self.float_precision)
                             for c, ss in self.gdl.sumstats_table.items()}
        except AttributeError:
            # If not provided, use the overall sample size:
            self.n_per_snp = {c: np.repeat(self.gdl.n, c_size)
                              for c, c_size in self.shapes.items()}

        self.validation_std_beta = None

    def set_validation_sumstats(self):
        """
        Set the validation summary statistics.
        TODO: Allow the user to set the validation sumstats as a property of the model.
        """
        raise NotImplementedError

    def split_gwas_sumstats(self,
                            prop_train=0.8,
                            seed=None,
                            **kwargs):
        """
        Split the GWAS summary statistics into training and validation sets, using the
        PUMAS procedure outlined in Zhao et al. (2021).

        :param prop_train: The proportion of samples to include in the training set.
        :param seed: The random seed for reproducibility.
        :param kwargs: Additional keyword arguments to pass to the `sumstats_train_test_split` function.
        """

        from magenpy.utils.model_utils import sumstats_train_test_split

        logger.debug("> Splitting the GWAS summary statistics into training and validation sets. "
                     f"Training proportion: {prop_train}")

        split_sumstats = sumstats_train_test_split(self.gdl,
                                                   prop_train=prop_train,
                                                   seed=seed,
                                                   **kwargs)

        self.std_beta = {
            c: split_sumstats[c]['train_beta'].astype(self.float_precision)
            for c in self.chromosomes
        }

        self.n_per_snp = {
            c: self.n_per_snp[c]*prop_train
            for c in self.chromosomes
        }

        self.validation_std_beta = {
            c: split_sumstats[c]['test_beta'].astype(self.float_precision)
            for c in self.chromosomes
        }

    def fit(self, *args, **kwargs):
        """
        A genetic method to fit the Bayesian PRS model. This method should be implemented by the
        specific Bayesian PRS model.
        :raises NotImplementedError: If the method is not implemented in the child class.
        """
        raise NotImplementedError

    def get_proportion_causal(self):
        """
        A generic method to get an estimate of the proportion of causal variants.
        :raises NotImplementedError: If the method is not implemented in the child class.
        """
        raise NotImplementedError

    def get_heritability(self):
        """
        A generic method to get an estimate of the heritability, or proportion of variance explained by SNPs.
        :raises NotImplementedError: If the method is not implemented in the child class.
        """
        raise NotImplementedError

    def get_pip(self):
        """
        :return: The posterior inclusion probability for each variant in the model.
        """
        return self.pip

    def get_posterior_mean_beta(self):
        """
        :return: The posterior mean of the effect sizes (BETA) for each variant in the model.
        """
        return self.post_mean_beta

    def get_posterior_variance_beta(self):
        """
        :return: The posterior variance of the effect sizes (BETA) for each variant in the model.
        """
        return self.post_var_beta

    def predict(self, test_gdl=None):
        """
        Given the inferred effect sizes, predict the phenotype for the training samples in
        the GWADataLoader object or new test samples. If `test_gdl` is not provided, genotypes
        from training samples will be used (if available).

        :param test_gdl: A GWADataLoader object containing genotype data for new test samples.
        :raises ValueError: If the posterior means for BETA are not set. AssertionError if the GWADataLoader object
        does not contain genotype data.
        """

        if self.post_mean_beta is None:
            raise ValueError("The posterior means for BETA are not set. Call `.fit()` first.")

        if test_gdl is None:
            assert self.gdl.genotype is not None, "The GWADataLoader object must contain genotype data."
            test_gdl = self.gdl
            post_mean_beta = self.post_mean_beta
        else:
            _, post_mean_beta, _ = self.harmonize_data(gdl=test_gdl)

        return test_gdl.predict(post_mean_beta)

    def harmonize_data(self, gdl=None, parameter_table=None):
        """
        Harmonize the inferred effect sizes with a new GWADataLoader object. This method is useful
        when the user wants to predict on new samples or when the effect sizes are inferred from a
        different set of samples. The method aligns the effect sizes with the SNP table in the
        GWADataLoader object.

        :param gdl: An instance of `GWADataLoader` object.
        :param parameter_table: A `pandas` DataFrame of variant effect sizes.

        :return: A tuple of the harmonized posterior inclusion probability, posterior mean for the effect sizes,
        and posterior variance for the effect sizes.

        """

        if gdl is None and parameter_table is None:
            return

        if gdl is None:
            gdl = self.gdl

        if parameter_table is None:
            parameter_table = self.to_table(per_chromosome=True)
        else:
            parameter_table = {c: parameter_table.loc[parameter_table['CHR'] == c, ]
                               for c in parameter_table['CHR'].unique()}

        snp_tables = gdl.to_snp_table(col_subset=['SNP', 'A1', 'A2'],
                                      per_chromosome=True)

        pip = {}
        post_mean_beta = {}
        post_var_beta = {}

        common_chroms = sorted(list(set(snp_tables.keys()).intersection(set(parameter_table.keys()))))

        from magenpy.utils.model_utils import merge_snp_tables

        for c in common_chroms:

            try:
                post_mean_cols = expand_column_names('BETA', self.post_mean_beta[c].shape)
                pip_cols = expand_column_names('PIP', self.post_mean_beta[c].shape)
                post_var_cols = expand_column_names('VAR_BETA', self.post_mean_beta[c].shape)

            except (TypeError, KeyError):
                pip_cols = [col for col in parameter_table[c].columns if 'PIP' in col]
                post_var_cols = [col for col in parameter_table[c].columns if 'VAR_BETA' in col]
                post_mean_cols = [col for col in parameter_table[c].columns
                                  if 'BETA' in col and col not in post_var_cols]

            # Merge the effect table with the GDL SNP table:
            c_df = merge_snp_tables(snp_tables[c], parameter_table[c], how='left',
                                    signed_statistics=post_mean_cols)

            if len(c_df) < len(snp_tables[c]):
                raise ValueError("The parameter table could not aligned with the reference SNP table. This may due to "
                                 "conflicts/errors in use of reference vs. alternative alleles.")

            # Obtain the values for the posterior mean:
            c_df[post_mean_cols] = c_df[post_mean_cols].fillna(0.)
            post_mean_beta[c] = c_df[post_mean_cols].values

            # Obtain the values for the posterior inclusion probability:
            if len(set(pip_cols).intersection(set(c_df.columns))) > 0:
                c_df[pip_cols] = c_df[pip_cols].fillna(0.)
                pip[c] = c_df[pip_cols].values

            # Obtain the values for the posterior variance:
            if len(set(post_var_cols).intersection(set(c_df.columns))) > 0:
                c_df[post_var_cols] = c_df[post_var_cols].fillna(0.)
                post_var_beta[c] = c_df[post_var_cols].values

        if len(pip) < 1:
            pip = None

        if len(post_var_beta) < 1:
            post_var_beta = None

        return pip, post_mean_beta, post_var_beta

    def to_table(self, col_subset=('CHR', 'SNP', 'POS', 'A1', 'A2'), per_chromosome=False):
        """
        Output the posterior estimates for the effect sizes to a pandas dataframe.
        :param col_subset: The subset of columns to include in the tables (in addition to the effect sizes).
        :param per_chromosome: If True, return a separate table for each chromosome.

        :return: A pandas Dataframe with the posterior estimates for the effect sizes.
        """

        if self.post_mean_beta is None:
            raise Exception("The posterior means for BETA are not set. Call `.fit()` first.")

        tables = self.gdl.to_snp_table(col_subset=col_subset, per_chromosome=True)

        for c in self.chromosomes:

            cols_to_add = []

            mean_beta_df = pd.DataFrame(self.post_mean_beta[c],
                                        columns=expand_column_names('BETA', self.post_mean_beta[c].shape),
                                        index=tables[c].index)
            cols_to_add.append(mean_beta_df)

            if self.pip is not None:
                pip_df = pd.DataFrame(self.pip[c],
                                      columns=expand_column_names('PIP', self.pip[c].shape),
                                      index=tables[c].index)
                cols_to_add.append(pip_df)

            if self.post_var_beta is not None:
                var_beta_df = pd.DataFrame(self.post_var_beta[c],
                                           columns=expand_column_names('VAR_BETA', self.post_var_beta[c].shape),
                                           index=tables[c].index)
                cols_to_add.append(var_beta_df)

            tables[c] = pd.concat([tables[c]] + cols_to_add, axis=1)

        if per_chromosome:
            return tables
        else:
            return pd.concat([tables[c] for c in self.chromosomes])

    def pseudo_validate(self, test_gdl=None):
        """
        Evaluate the prediction accuracy of the inferred PRS using external GWAS summary statistics.

        :param test_gdl: A `GWADataLoader` object with the external GWAS summary statistics and LD matrix information.

        :return: The pseudo-R^2 metric.
        """

        from ..eval.pseudo_metrics import pseudo_r2, _streamlined_pseudo_r2
        from ..utils.compute_utils import dict_concat

        assert self.post_mean_beta is not None, "The posterior means for BETA are not set. Call `.fit()` first."
        assert self.validation_std_beta is not None or test_gdl is not None, (
            "Must provide a GWADataLoader object with validation sumstats or initialize the standardized "
            "betas inside the model."
        )

        if test_gdl is not None:
            return pseudo_r2(test_gdl, self.to_table(per_chromosome=False))
        else:

            # Check if q is an attribute of the model:
            if hasattr(self, 'q'):
                ldw_prs = {c: self.q[c] + self.post_mean_beta[c] for c in self.shapes}
            else:
                # Compute LD-weighted PRS weights first:
                ldw_prs = {}
                for c in self.shapes:
                    ldw_prs[c] = self.gdl.ld[c].dot(self.post_mean_beta[c])

            return _streamlined_pseudo_r2(
                dict_concat(self.validation_std_beta),
                dict_concat(self.post_mean_beta),
                dict_concat(ldw_prs)
            )

    def set_model_parameters(self, parameter_table):
        """
        Parses a pandas dataframe with model parameters and assigns them 
        to the corresponding class attributes. 
        
        For example: 
            * Columns with `BETA`, will be assigned to `self.post_mean_beta`.
            * Columns with `PIP` will be assigned to `self.pip`.
            * Columns with `VAR_BETA`, will be assigned to `self.post_var_beta`.
        
        :param parameter_table: A pandas table or dataframe.
        """

        self.pip, self.post_mean_beta, self.post_var_beta = self.harmonize_data(parameter_table=parameter_table)

    def read_inferred_parameters(self, f_names, sep=r"\s+"):
        """
        Read a file with the inferred parameters.
        :param f_names: A path (or list of paths) to the file with the effect sizes.
        :param sep: The delimiter for the file(s).
        """

        if isinstance(f_names, str):
            f_names = [f_names]

        param_table = []

        for f_name in f_names:
            param_table.append(pd.read_csv(f_name, sep=sep))

        if len(param_table) > 0:
            param_table = pd.concat(param_table)
            self.set_model_parameters(param_table)
        else:
            raise FileNotFoundError

    def write_inferred_parameters(self, f_name, per_chromosome=False, sep="\t"):
        """
        A convenience method to write the inferred posterior for the effect sizes to file.

        TODO:
            * Support outputting scoring files compatible with PGS catalog format:
            https://www.pgscatalog.org/downloads/#dl_scoring_files

        :param f_name: The filename (or directory) where to write the effect sizes
        :param per_chromosome: If True, write a file for each chromosome separately.
        :param sep: The delimiter for the file (tab by default).
        """

        tables = self.to_table(per_chromosome=per_chromosome)

        if '.fit' not in f_name:
            ext = '.fit'
        else:
            ext = ''

        if per_chromosome:
            for c, tab in tables.items():
                try:
                    tab.to_csv(osp.join(f_name, f'chr_{c}.fit'), sep=sep, index=False)
                except Exception as e:
                    raise e
        else:
            try:
                tables.to_csv(f_name + ext, sep=sep, index=False)
            except Exception as e:
                raise e

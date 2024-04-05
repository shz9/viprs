import numpy as np
import pandas as pd
import os.path as osp

from ..utils.compute_utils import expand_column_names, dict_mean
from magenpy.utils.model_utils import merge_snp_tables


class BayesPRSModel:
    """
    A base class for Bayesian PRS models. This class defines the basic structure and methods
    that are common to most Bayesian PRS models. Specifically, this class provides methods and interfaces
    for initialization, harmonization, prediction, and fitting of Bayesian PRS models.

    The class is generic is designed to be inherited and extended by
    specific Bayesian PRS models, such as `LDPred` and `VIPRS`.

    :ivar gdl: A GWADataLoader object containing harmonized GWAS summary statistics and
    Linkage-Disequilibrium (LD) matrices.
    :ivar Nj: A dictionary where keys are chromosomes and values are the sample sizes per variant.
    :ivar shapes: A dictionary where keys are chromosomes and values are the shapes of the variant arrays
    (e.g. the number of variants per chromosome).
    :ivar _sample_size: The average per-SNP sample size.
    :ivar pip: The posterior inclusion probability.
    :ivar post_mean_beta: The posterior mean for the effect sizes.
    :ivar post_var_beta: The posterior variance for the effect sizes.
    """

    def __init__(self, gdl):
        """
        Initialize the Bayesian PRS model.
        :param gdl: An instance of `GWADataLoader`.
        """

        self.gdl = gdl

        # Sample size per SNP:
        try:
            self.Nj = {c: ss.n_per_snp.astype(float) for c, ss in gdl.sumstats_table.items()}
        except AttributeError:
            # If not provided, use the overall sample size:
            self.Nj = {c: np.repeat(gdl.n, c_size).astype(float) for c, c_size in gdl.shapes.items()}

        self.shapes = self.gdl.shapes.copy()

        # Determine the overall sample size:
        self._sample_size = dict_mean(self.Nj)

        # Inferred model parameters:
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

        for c in common_chroms:

            try:
                post_mean_cols = expand_column_names('BETA', self.post_mean_beta[c].shape)
                if isinstance(post_mean_cols, str):
                    post_mean_cols = [post_mean_cols]

                pip_cols = expand_column_names('PIP', self.post_mean_beta[c].shape)
                if isinstance(pip_cols, str):
                    pip_cols = [pip_cols]

                post_var_cols = expand_column_names('VAR_BETA', self.post_mean_beta[c].shape)
                if isinstance(post_var_cols, str):
                    post_var_cols = [post_var_cols]

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

    def to_table(self, col_subset=('CHR', 'SNP', 'A1', 'A2'), per_chromosome=False):
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

            tables[c][expand_column_names('BETA', self.post_mean_beta[c].shape)] = self.post_mean_beta[c]

            if self.pip is not None:
                tables[c][expand_column_names('PIP', self.pip[c].shape)] = self.pip[c]

            if self.post_var_beta is not None:
                tables[c][expand_column_names('VAR_BETA', self.post_var_beta[c].shape)] = self.post_var_beta[c]

        if per_chromosome:
            return tables
        else:
            return pd.concat([tables[c] for c in self.chromosomes])

    def pseudo_validate(self, test_gdl, metric='pearson_correlation'):
        """
        Evaluate the prediction accuracy of the inferred PRS using external GWAS summary statistics.

        :param test_gdl: A `GWADataLoader` object with the external GWAS summary statistics and LD matrix information.
        :param metric: The metric to use for evaluation. Options: 'r2' or 'pearson_correlation'.

        :return: The pseudo-validation metric.
        """

        from ..eval.pseudo_metrics import pseudo_r2, pseudo_pearson_r

        metric = metric.lower()

        assert self.post_mean_beta is not None, "The posterior means for BETA are not set. Call `.fit()` first."

        if metric in ('pearson_correlation', 'corr', 'r'):
            return pseudo_pearson_r(test_gdl, self.to_table(per_chromosome=False))
        elif metric == 'r2':
            return pseudo_r2(test_gdl, self.to_table(per_chromosome=False))
        else:
            raise KeyError(f"Pseudo validation metric ({metric}) not recognized. "
                           f"Options are: 'r2' or 'pearson_correlation'.")

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

    def read_inferred_parameters(self, f_names, sep="\t"):
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

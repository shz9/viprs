# cython: linetrace=False
# cython: profile=False
# cython: binding=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: language_level=3
# cython: infer_types=True

import numpy as np
import pandas as pd
import os.path as osp

from viprs.utils.compute_utils import expand_column_names
from magenpy.utils.model_utils import merge_snp_tables


cdef class PRSModel:

    def __init__(self, gdl):

        self.N = gdl.sample_size

        # Sample size per SNP:
        try:
            self.Nj = {c: ss.n_per_snp.astype(float) for c, ss in gdl.sumstats_table.items()}
        except AttributeError:
            # If not provided, use the overall sample size:
            self.Nj = {c: np.repeat(self.N, c_size).astype(float) for c, c_size in gdl.shapes.items()}

        self.gdl = gdl  # An instance of GWADataLoader
        self.shapes = self.gdl.shapes.copy()

        # Inferred model parameters:
        self.pip = None  # Posterior inclusion probability
        self.post_mean_beta = None  # The posterior mean for the effect sizes
        self.post_var_beta = None  # The posterior variance for the effect sizes

    @property
    def chromosomes(self):
        """
        Return the list of chromosomes that are part of PRSModel
        """
        return sorted(list(self.shapes.keys()))

    @property
    def m(self) -> int:
        return self.gdl.m

    @property
    def n_snps(self) -> int:
        return self.m

    cpdef fit(self):
        raise NotImplementedError

    cpdef get_proportion_causal(self):
        raise NotImplementedError

    cpdef get_heritability(self):
        raise NotImplementedError

    cpdef get_pip(self):
        """
        Get the posterior inclusion probability
        """
        return self.pip

    cpdef get_posterior_mean_beta(self):
        """
        Get the posterior mean for the effect sizes BETA.
        """
        return self.post_mean_beta

    cpdef get_posterior_variance_beta(self):
        """
        Get the posterior variance for the effect sizes BETA. 
        """
        return self.post_var_beta

    cpdef predict(self, gdl=None):
        """
        Given the inferred effect sizes, predict the phenotype for samples in 
        the GWADataLoader object passed to PRSModel or a new GWADataLoader 
        object.
        :param gdl: A GWADataLoader object containing genotype data for new samples.
        """

        if self.post_mean_beta is None:
            raise Exception("The posterior means for BETA are not set. Call `.fit()` first.")

        if gdl is None:
            gdl = self.gdl
            post_mean_beta = self.post_mean_beta
        else:
            _, post_mean_beta, _ = self.harmonize_data(gdl=gdl)

        return gdl.predict(post_mean_beta)

    cpdef harmonize_data(self, gdl=None, parameter_table=None):
        """
        Harmonize the inferred effect sizes with a GWAS Data Loader object
        The user must provide at least one object to harmonize with existing information.
        :param gdl: A `GWADataLoader` object
        :param parameter_table: The table of effect sizes
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

    cpdef to_table(self, col_subset=('CHR', 'SNP', 'A1', 'A2'), per_chromosome=False):
        """
        Output the posterior estimates for the effect sizes to a pandas dataframe.
        :param col_subset: The subset of columns to include in the tables (in addition to the effect sizes).
        :param per_chromosome: If True, return a separate table for each chromosome.
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

    cpdef set_model_parameters(self, parameter_table):
        """
        Parses a pandas dataframe with model parameters and assigns them 
        to the corresponding class attributes. 
        
        For example: 
            - Columns with `BETA`, will be assigned to `self.post_mean_beta`.
            - Columns with `PIP` will be assigned to `self.pip`.
            - Columns with `VAR_BETA`, will be assigned to `self.post_var_beta`.
        
        :param parameter_table: A pandas table or dataframe.
        """

        self.pip, self.post_mean_beta, self.post_var_beta = self.harmonize_data(parameter_table=parameter_table)
    cpdef read_inferred_parameters(self, f_names, sep="\t"):
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

    cpdef write_inferred_parameters(self, f_name, per_chromosome=False, sep="\t"):
        """
        Write the inferred posterior for the effect sizes to file.
        :param f_name: The filename (or directory) where to write the effect sizes
        :param per_chromosome: If True, write a file for each chromosome separately.
        :param sep: The delimiter for the file (tab by default).
        """

        tables = self.to_table(per_chromosome=per_chromosome)

        if per_chromosome:
            for c, tab in tables.items():
                try:
                    tab.to_csv(osp.join(f_name, f'chr_{c}.fit'), sep=sep, index=False)
                except Exception as e:
                    raise e
        else:
            try:
                tables.to_csv(f_name, sep=sep, index=False)
            except Exception as e:
                raise e

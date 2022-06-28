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
        self.M = gdl.m
        self.shapes = self.gdl.shapes

        # Inferred model parameters:
        self.pip = None  # Posterior inclusion probability
        self.post_mean_beta = None  # The posterior mean for the effect sizes
        self.post_var_beta = None  # The posterior variance for the effect sizes

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

    cpdef harmonize_data(self, gdl=None, eff_table=None):
        """
        Harmonize the inferred effect sizes with a GWAS Data Loader object
        The user must provide at least one object to harmonize with existing information.
        :param gdl: The GWAS Data Loader object
        :param eff_table: The table of effect sizes
        """

        if gdl is None and eff_table is None:
            return

        if gdl is None:
            gdl = self.gdl

        if eff_table is None:
            eff_table = self.to_table(per_chromosome=True)
        else:
            eff_table = {c: eff_table.loc[eff_table['CHR'] == c, ]
                         for c in eff_table['CHR'].unique()}

        snp_tables = gdl.to_snp_table(col_subset=['SNP', 'A1'],
                                      per_chromosome=True)

        pip = {}
        post_mean_beta = {}
        post_var_beta = {}

        for c, snp_table in snp_tables.items():

            # Merge the effect table with the GDL SNP table:
            c_df = merge_snp_tables(snp_table, eff_table[c], how='left')

            # Obtain the values for the posterior mean:
            c_df['BETA'] = c_df['BETA'].fillna(0.)
            post_mean_beta[c] = c_df['BETA'].values

            # Obtain the values for the posterior inclusion probability:
            if 'PIP' in c_df.columns:
                c_df['PIP'] = c_df['PIP'].fillna(0.)
                pip[c] = c_df['PIP'].values

            # Obtain the values for the posterior variance:
            if 'VAR_BETA' in c_df.columns:
                post_var_beta[c] = c_df['VAR_BETA'].values


        return pip, post_mean_beta, post_var_beta

    cpdef to_table(self, per_chromosome=False, col_subset=('CHR', 'SNP', 'A1', 'A2')):
        """
        Output the posterior estimates for the effect sizes to a pandas table.
        :param per_chromosome: If True, return a separate table for each chromosome.
        :param col_subset: The subset of columns to include in the tables (in addition to the effect sizes).
        """

        if self.post_mean_beta is None:
            raise Exception("The posterior means for BETA are not set. Call `.fit()` first.")

        tables = self.gdl.to_snp_table(col_subset=col_subset, per_chromosome=True)

        for c in self.shapes:
            tables[c]['PIP'] = self.pip[c]
            tables[c]['BETA'] = self.post_mean_beta[c]
            tables[c]['VAR_BETA'] = self.post_var_beta[c]

        if per_chromosome:
            return tables
        else:
            return pd.concat(tables.values())

    cpdef read_inferred_params(self, f_names, sep="\t"):
        """
        Read a file with the inferred parameters.
        :param f_names: A path (or list of paths) to the file with the effect sizes.
        :param sep: The delimiter for the file(s).
        """

        if isinstance(f_names, str):
            f_names = [f_names]

        eff_table = []

        for f_name in f_names:
            eff_table.append(pd.read_csv(f_name, sep=sep))

        eff_table = pd.concat(eff_table)

        self.pip, self.post_mean_beta, self.post_var_beta = self.harmonize_data(eff_table=eff_table)

    cpdef write_inferred_params(self, f_name, per_chromosome=False, sep="\t"):
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

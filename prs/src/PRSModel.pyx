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

from ..gwasimulator.model_utils import merge_snp_tables

cdef class PRSModel:

    def __init__(self, gdl):

        self.N = gdl.sample_size

        # Sample size per SNP:
        try:
            self.Nj = {c: n.astype(float) for c, n in gdl.n_per_snp.items()}
        except AttributeError:
            # If not provided, use the overall sample size:
            self.Nj = {c: np.repeat(self.N, c_size).astype(float) for c, c_size in gdl.shapes.items()}

        self.M = gdl.M
        self.gdl = gdl  # An instance of GWASDataLoader
        self.shapes = self.gdl.shapes

        # Inferred model parameters:
        self.pip = None  # Posterior inclusion probability
        self.inf_beta = None  # Inferred beta

    cpdef fit(self):
        raise NotImplementedError

    cpdef get_proportion_causal(self):
        return None

    cpdef get_heritability(self):
        return None

    cpdef get_pip(self):
        return self.pip

    cpdef get_inf_beta(self):
        return self.inf_beta

    cpdef predict(self, gdl=None):

        if self.inf_beta is None:
            raise Exception("Inferred betas are not set. Call `.fit()` first.")

        if gdl is None:
            gdl = self.gdl
            inf_beta = self.inf_beta
        else:
            _, inf_beta = self.harmonize_data(gdl=gdl)

        return gdl.predict(inf_beta)

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
            eff_table = self.to_table(per_chromosome=False)

        snp_tables = gdl.to_snp_table(col_subset=('CHR', 'SNP', 'A1'),
                                      per_chromosome=True)

        pip = {}
        inf_beta = {}

        for c, snp_table in snp_tables.items():

            # Merge the effect table with the GDL SNP table:
            c_df = merge_snp_tables(snp_table, eff_table, how='left')

            # Fill in missing values:
            c_df['PIP'] = c_df['PIP'].fillna(0.)
            c_df['BETA'] = c_df['BETA'].fillna(0.)

            pip[c] = c_df['PIP'].values
            inf_beta[c] = c_df['BETA'].values

        return pip, inf_beta

    cpdef to_table(self, per_chromosome=False, col_subset=('CHR', 'SNP', 'A1', 'A2')):

        if self.inf_beta is None:
            raise Exception("Inferred betas are not set. Call `.fit()` first.")

        tables = self.gdl.to_snp_table(per_chromosome=True, col_subset=col_subset)

        for c in self.shapes:
            tables[c]['PIP'] = self.pip[c]
            tables[c]['BETA'] = self.inf_beta[c]

        if per_chromosome:
            return tables
        else:
            return pd.concat(tables.values())

    cpdef read_inferred_params(self, f_names):

        if isinstance(f_names, str):
            f_names = [f_names]

        eff_table = []

        for f_name in f_names:
            eff_table.append(pd.read_csv(f_name, sep="\t"))

        eff_table = pd.concat(eff_table)

        self.pip, self.inf_beta = self.harmonize_data(eff_table=eff_table)

    cpdef write_inferred_params(self, f_name, per_chromosome=False):

        tables = self.to_table(per_chromosome=per_chromosome)

        if per_chromosome:
            for c, tab in tables.items():
                try:
                    tab.to_csv(osp.join(f_name, f'chr_{c}.fit'), sep="\t", index=False)
                except Exception as e:
                    raise e
        else:
            try:
                tables.to_csv(f_name, sep="\t", index=False)
            except Exception as e:
                raise e

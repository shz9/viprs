from types import MethodType

import numpy as np
from scipy.sparse import csr_matrix

from viprs.model import LDPredInf, LDPredInfGrid


class FakeLD:
    def __init__(self, matrix, maf=None):
        self.matrix = csr_matrix(matrix)
        self.maf = maf

    def load(self, dtype=None):
        matrix = self.matrix.astype(dtype) if dtype else self.matrix
        return FakeLD(matrix, self.maf)

    def to_csr(self):
        return self.matrix

    def dot(self, value):
        return self.matrix.dot(value)


class FakeGDL:
    def __init__(self, matrix, maf=None):
        self.chromosomes = [1]
        self.ld = {1: FakeLD(matrix, maf)}
        self.sumstats_table = {1: FakeSumstats(maf)}
        self.m = len(matrix)


class FakeSumstats:
    def __init__(self, maf):
        self.maf = maf


def make_model(cls=LDPredInf, **kwargs):
    model = object.__new__(cls)
    model.gdl = FakeGDL(
        [[1.0, 0.8], [0.8, 1.0]],
        maf=np.array([0.1, 0.5])
    )
    model.float_precision = "float64"
    model.shapes = {1: 2}
    model.n_per_snp = {1: np.array([100.0, 100.0])}
    model.std_beta = {1: np.array([1.0, 0.0])}
    model.validation_std_beta = None
    model.h2 = kwargs.get("h2", 0.2)
    model.alpha = kwargs.get("alpha", 0.)
    model.penalty = None
    return model


def test_ldpred_inf_matches_direct_solve():
    model = make_model()
    model.gdl.sumstats_table[1].maf = None
    model.gdl.ld[1].maf = None
    model.fit(rtol=1e-12)

    penalty = 2 / (100 * 0.2)
    expected = np.linalg.solve(
        np.array([[1.0, 0.8], [0.8, 1.0]]) + penalty * np.eye(2),
        np.array([1.0, 0.0]),
    )
    np.testing.assert_allclose(model.post_mean_beta[1], expected)


def test_chromosome_heritability_sets_local_penalty():
    model = make_model(h2={1: 0.1})
    model._validate_prior_parameters()
    assert model.get_penalties()[1] == 2 / (100 * 0.1)


def test_alpha_sets_frequency_dependent_penalties():
    model = make_model(alpha=-1.)
    heterozygosity = np.array([0.18, 0.5])
    variance_weights = heterozygosity ** model.alpha
    expected = variance_weights.sum() / (
        100 * model.h2 * variance_weights
    )

    model.fit(rtol=1e-12)

    expected_beta = np.linalg.solve(
        np.array([[1.0, 0.8], [0.8, 1.0]]) + np.diag(expected),
        np.array([1.0, 0.0]),
    )

    np.testing.assert_allclose(model.penalty[1], expected)
    np.testing.assert_allclose(model.post_mean_beta[1], expected_beta)


def test_grid_search_splits_selects_and_refits():
    model = make_model(LDPredInfGrid)
    model.h2 = 1.0
    model.penalty_factors = np.array([0.1, 1.0, 10.0])
    model.alpha_grid = np.array([-1.0, 0.0, 1.0])
    model.validation_result = None
    model.selected_penalty_factor = None
    model.selected_alpha = None
    full_beta = model.std_beta[1].copy()

    def split(self, prop_train=0.8, seed=None):
        self.std_beta = {1: np.array([1.0, 0.0])}
        self.validation_std_beta = {1: np.array([1.0, 0.5])}
        self.n_per_snp = {1: np.array([2.0, 2.0])}

    def reset(self):
        self.std_beta = {1: full_beta.copy()}
        self.n_per_snp = {1: np.array([2.0, 2.0])}
        self.validation_std_beta = None

    model.split_gwas_sumstats = MethodType(split, model)
    model.initialize_input_data_arrays = MethodType(reset, model)
    model.fit(rtol=1e-12)

    assert model.selected_penalty_factor == 1.0
    assert model.selected_alpha == 1.0
    assert model.alpha == 1.0
    assert model.validation_result["Selected"].sum() == 1
    assert model.validation_std_beta is None

import numpy as np

from viprs.model import BayesPRSModel


class FakeSumstats:
    def __init__(self, n_per_snp, std_beta):
        self.n_per_snp = np.asarray(n_per_snp)
        self._std_beta = np.asarray(std_beta)

    def get_snp_pseudo_corr(self):
        return self._std_beta


class FakeGDL:
    def __init__(self):
        self.sumstats_table = {
            1: FakeSumstats([100.0, 80.0], [0.1, 0.2]),
            2: FakeSumstats([90.0], [0.3]),
        }
        self.n = 100.0


def make_model():
    model = object.__new__(BayesPRSModel)
    model.gdl = FakeGDL()
    model.float_precision = "float64"
    model.shapes = {1: 2, 2: 1}
    model.initialize_input_data_arrays()
    return model


def test_initialize_input_data_arrays_restores_sample_size():
    model = make_model()
    model.n_per_snp = {1: np.array([20.0, 16.0]), 2: np.array([18.0])}
    model._sample_size = 20.0

    model.initialize_input_data_arrays()

    assert model.n == 100.0
    np.testing.assert_array_equal(model.n_per_snp[1], [100.0, 80.0])


def test_split_uses_returned_sample_sizes_without_compounding(monkeypatch):
    model = make_model()

    split_result = {
        1: {
            "train_beta": np.array([0.11, 0.19]),
            "train_n": np.array([75.0, 55.0]),
            "test_beta": np.array([0.06, 0.24]),
            "test_n": np.array([25.0, 25.0]),
        },
        2: {
            "train_beta": np.array([0.29]),
            "train_n": np.array([65.0]),
            "test_beta": np.array([0.34]),
            "test_n": np.array([25.0]),
        },
    }

    monkeypatch.setattr(
        "magenpy.utils.model_utils.sumstats_train_test_split",
        lambda *args, **kwargs: split_result,
    )

    model.split_gwas_sumstats(prop_train=0.75, seed=1)
    model.split_gwas_sumstats(prop_train=0.75, seed=1)

    assert model.n == 75.0
    np.testing.assert_array_equal(model.n_per_snp[1], [75.0, 55.0])
    np.testing.assert_array_equal(model.n_per_snp[2], [65.0])


def test_split_preserves_trailing_sample_size_dimensions(monkeypatch):
    model = make_model()
    model.n_per_snp = {c: n[:, None] for c, n in model.n_per_snp.items()}

    split_result = {
        c: {
            "train_beta": ss.get_snp_pseudo_corr(),
            "train_n": ss.n_per_snp * 0.8,
            "test_beta": ss.get_snp_pseudo_corr(),
            "test_n": ss.n_per_snp * 0.2,
        }
        for c, ss in model.gdl.sumstats_table.items()
    }
    monkeypatch.setattr(
        "magenpy.utils.model_utils.sumstats_train_test_split",
        lambda *args, **kwargs: split_result,
    )

    model.split_gwas_sumstats()

    assert model.n_per_snp[1].shape == (2, 1)
    assert model.n == 80.0

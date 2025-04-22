
import magenpy as mgp
import viprs as vp
from viprs.model.vi.e_step_cpp import check_blas_support, check_omp_support
import numpy as np
from viprs.model import VIPRS, VIPRSMix, VIPRSGrid
from viprs.model.gridsearch import (
    HyperparameterGrid,
    select_best_model,
    bayesian_model_average
)
from functools import partial
import shutil
import pytest


@pytest.fixture(scope='module')
def gdl_object():
    """
    Initialize a GWADataLoader using data pre-packaged with magenpy.
    Make this data loader available to all tests.
    """
    gdl = mgp.GWADataLoader(mgp.tgp_eur_data_path(),
                            sumstats_files=mgp.ukb_height_sumstats_path(),
                            sumstats_format='fastgwa',
                            backend='xarray')

    ld_block_url = ("https://bitbucket.org/nygcresearch/ldetect-data/raw/"
                    "ac125e47bf7ff3e90be31f278a7b6a61daaba0dc/EUR/fourier_ls-all.bed")
    gdl.compute_ld('block', gdl.output_dir, ld_blocks_file=ld_block_url)

    gdl.harmonize_data()

    yield gdl

    # Clean up after tests are done:
    gdl.cleanup()
    shutil.rmtree(gdl.temp_dir)
    shutil.rmtree(gdl.output_dir)


@pytest.fixture(scope='module')
def viprs_model(gdl_object):
    """
    Initialize a basic VIPRS model using GWAS sumstats data pre-packaged with magenpy.
    Make this data loader available to all tests.
    """
    return vp.VIPRS(gdl_object)


@pytest.fixture(scope='module')
def viprsmix_model(gdl_object):
    """
    Initialize a VIPRS model (Mixture prior) using GWAS sumstats data pre-packaged with magenpy.
    Make this data loader available to all tests.
    """
    return VIPRSMix(gdl_object, K=10)


@pytest.fixture(scope='module')
def grid_obj():
    """
    Initialize a grid object.
    """
    grid = HyperparameterGrid()
    grid.generate_pi_grid(steps=10)
    return grid


@pytest.fixture(scope='module')
def viprs_grid_model(gdl_object, grid_obj):
    """
    Initialize a VIPRSGrid model using GWAS sumstats data pre-packaged with magenpy,
    as well as a grid object.
    """

    return VIPRSGrid(gdl_object, grid_obj)


class TestVIPRS(object):

    def test_init(self,
                  viprs_model: VIPRS,
                  gdl_object: mgp.GWADataLoader):

        assert viprs_model.m == gdl_object.m

        viprs_model.initialize()

        # Check the input data:
        for p in (viprs_model.std_beta, viprs_model.n_per_snp):
            assert p[22].shape == (viprs_model.m, )

        # Check the LD data:
        assert viprs_model.ld_indptr[22].shape == (viprs_model.m + 1, )
        assert viprs_model.ld_left_bound[22].shape == (viprs_model.m, )
        assert viprs_model.ld_data[22].shape == (viprs_model.ld_indptr[22][-1], )

        # Check hyperparameters:
        assert 0. < viprs_model.pi < 1.
        assert 0. < viprs_model.sigma_epsilon < 1.
        assert viprs_model.tau_beta > 0.

        # Check the model parameters:
        for p in (viprs_model.var_gamma, viprs_model.var_mu, viprs_model.var_tau,
                  viprs_model.q, viprs_model.eta):
            assert p[22].shape == (viprs_model.m, )

        # Other checks here?

    def test_fit(self, viprs_model: VIPRS):

        viprs_model.fit(max_iter=10)

        # Check the posterior moments:
        for p in (viprs_model.pip, viprs_model.post_mean_beta, viprs_model.post_var_beta):
            assert p[22].shape == (viprs_model.m,)

        # Test that the following methods are working properly:
        viprs_model.to_table()
        viprs_model.to_theta_table()
        viprs_model.to_history_table()
        viprs_model.mse()
        viprs_model.log_prior()
        viprs_model.loglikelihood()
        viprs_model.entropy()


class TestVIPRSMix(TestVIPRS):

    def test_init(self,
                  viprsmix_model: VIPRSMix,
                  gdl_object: mgp.GWADataLoader):

        assert viprsmix_model.m == gdl_object.m

        viprsmix_model.initialize()

        # Check the input data:
        assert viprsmix_model.std_beta[22].shape == (viprsmix_model.m,)
        assert viprsmix_model.n_per_snp[22].shape == (viprsmix_model.m, 1)

        # Check the LD data:
        assert viprsmix_model.ld_indptr[22].shape == (viprsmix_model.m + 1,)
        assert viprsmix_model.ld_left_bound[22].shape == (viprsmix_model.m,)
        assert viprsmix_model.ld_data[22].shape == (viprsmix_model.ld_indptr[22][-1],)

        # Check the hyperparameters:
        assert np.all((0. < viprsmix_model.pi) & (viprsmix_model.pi < 1.))
        assert 0. < np.sum(viprsmix_model.pi) < 1.
        assert 0. < viprsmix_model.sigma_epsilon < 1.
        assert np.all(viprsmix_model.tau_beta > 0.)

        # Check the variational parameters:
        for p in (viprsmix_model.var_gamma, viprsmix_model.var_mu, viprsmix_model.var_tau):
            assert p[22].shape == viprsmix_model.shapes[22]

        # Check the aggregation parameters:
        for p in (viprsmix_model.q, viprsmix_model.eta):
            assert p[22].shape == (viprsmix_model.m, )

    def test_fit(self, viprsmix_model):

        viprsmix_model.fit(max_iter=10)

        for p in (viprsmix_model.var_gamma, viprsmix_model.var_mu, viprsmix_model.var_tau):
            assert p[22].shape == viprsmix_model.shapes[22]

        for p in (viprsmix_model.pip, viprsmix_model.post_mean_beta, viprsmix_model.post_var_beta):
            assert p[22].shape == (viprsmix_model.m,)

        # Test that the following methods don't fail:
        viprsmix_model.to_table()
        viprsmix_model.to_theta_table()
        viprsmix_model.to_history_table()
        viprsmix_model.mse()
        viprsmix_model.log_prior()
        viprsmix_model.loglikelihood()
        viprsmix_model.entropy()


class TestVIPRSGrid(TestVIPRS):

    """
    Not testing the initialization because it should be the same as
    the standard VIPRS.
    """

    def test_fit(self,
                 viprs_grid_model: VIPRSGrid):

        # Test splitting the sumstats data (PUMAS):
        viprs_grid_model.split_gwas_sumstats()

        # Check the split sumstats:
        for p in (viprs_grid_model.std_beta, viprs_grid_model.validation_std_beta, viprs_grid_model.n_per_snp):
            assert p[22].shape == (viprs_grid_model.m,)

        # -----------------------------------------------------------------------

        model_selection_criteria = [
            partial(select_best_model, criterion='ELBO'),
            partial(select_best_model, criterion='pseudo_validation'),
            bayesian_model_average
        ]

        for criterion in model_selection_criteria:
            # Reset the search:
            viprs_grid_model._reset_search()

            # Perform model fit:
            viprs_grid_model.fit(max_iter=10)

            for p in (viprs_grid_model.pip, viprs_grid_model.post_mean_beta, viprs_grid_model.post_var_beta):
                assert p[22].shape == (viprs_grid_model.m, viprs_grid_model.n_models)

            # Test that the following methods don't fail:
            viprs_grid_model.to_table()
            viprs_grid_model.to_theta_table()
            viprs_grid_model.to_history_table()
            viprs_grid_model.mse()
            viprs_grid_model.log_prior()
            viprs_grid_model.loglikelihood()
            viprs_grid_model.entropy()
            viprs_grid_model.pseudo_validate()

            # Perform model selection:
            criterion(viprs_grid_model)

            viprs_grid_model.fit(max_iter=10)

            # Check that the other methods work fine still:
            for p in (viprs_grid_model.pip, viprs_grid_model.post_mean_beta, viprs_grid_model.post_var_beta):
                assert p[22].shape == (viprs_grid_model.m,)

            viprs_grid_model.to_table()
            viprs_grid_model.to_theta_table()
            viprs_grid_model.to_history_table()
            viprs_grid_model.mse()
            viprs_grid_model.log_prior()
            viprs_grid_model.loglikelihood()
            viprs_grid_model.entropy()
            viprs_grid_model.pseudo_validate()


@pytest.mark.xfail(not check_blas_support(), reason="BLAS library not found!")
def test_check_blas_support():
    assert check_blas_support()


@pytest.mark.xfail(not check_omp_support(), reason="OpenMP library not found!")
def test_check_omp_support():
    assert check_omp_support()

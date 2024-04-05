
import magenpy as mgp
import viprs as vp
from viprs.model.vi.e_step_cpp import check_blas_support, check_omp_support
import numpy as np
from viprs.model.gridsearch.HyperparameterGrid import HyperparameterGrid
from viprs.model.gridsearch.VIPRSGrid import VIPRSGrid
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

    # Compute LD matrix:
    gdl.compute_ld('shrinkage',
                   genetic_map_ne=11400,
                   genetic_map_sample_size=183,
                   output_dir='output/ld/')

    yield gdl

    # Clean up after tests are done:
    gdl.cleanup()
    shutil.rmtree(gdl.temp_dir)
    shutil.rmtree(gdl.output_dir)


@pytest.fixture(scope='module')
def viprs_model(gdl_object):
    """
    Initialize a VIPRS model using data pre-packaged with magenpy.
    Make this data loader available to all tests.
    """
    return vp.VIPRS(gdl_object, verbose=False)


@pytest.fixture(scope='module')
def viprsmix_model(gdl_object):
    """
    Initialize a VIPRS model using data pre-packaged with magenpy.
    Make this data loader available to all tests.
    """
    return vp.VIPRSMix(gdl_object, K=10, verbose=False)


@pytest.fixture(scope='module')
def viprsgrid_model(gdl_object):
    """
    Initialize a VIPRS model using data pre-packaged with magenpy.
    Make this data loader available to all tests.
    """
    # Create a grid:
    grid = HyperparameterGrid()
    # Generate a grid for pi using 5 equidistant grid points:
    grid.generate_pi_grid(steps=5)
    # Generate a grid for sigma epsilon using 5 equidistant grid points:
    grid.generate_sigma_epsilon_grid(steps=5)

    return VIPRSGrid(gdl_object, grid, verbose=False)


class TestVIPRS(object):

    def test_viprs_init(self, viprs_model, gdl_object):

        assert viprs_model.m == gdl_object.m

        viprs_model.initialize()

        assert 0. < viprs_model.pi < 1.
        assert 0. < viprs_model.sigma_epsilon < 1.
        assert viprs_model.tau_beta > 0.

        for p in (viprs_model.var_gamma, viprs_model.var_mu, viprs_model.var_tau,
                  viprs_model.q, viprs_model.eta, viprs_model.zeta):
            assert p[22].shape == (viprs_model.m, )
            assert p[22].dtype == viprs_model.float_precision

        # ...

    def test_viprs_fit(self, viprs_model):

        viprs_model.fit(max_iter=5)

        for p in (viprs_model.pip, viprs_model.post_mean_beta, viprs_model.post_var_beta):
            assert p[22].shape == (viprs_model.m,)
            assert p[22].dtype == viprs_model.float_precision


class TestVIPRSMix(TestVIPRS):

    def test_viprs_init(self, viprsmix_model, gdl_object):

        assert viprsmix_model.m == gdl_object.m

        viprsmix_model.initialize()

        assert np.all((0. < viprsmix_model.pi) & (viprsmix_model.pi < 1.))
        assert 0. < np.sum(viprsmix_model.pi) < 1.
        assert np.all((0. < viprsmix_model.sigma_epsilon) & (viprsmix_model.sigma_epsilon < 1.))
        assert np.all(viprsmix_model.tau_beta > 0.)

        for p in (viprsmix_model.var_gamma, viprsmix_model.var_mu, viprsmix_model.var_tau):
            assert p[22].shape == (viprsmix_model.m, 10)
            assert p[22].dtype == viprsmix_model.float_precision

        for p in (viprsmix_model.q, viprsmix_model.eta, viprsmix_model.zeta):
            assert p[22].shape == (viprsmix_model.m, )
            assert p[22].dtype == viprsmix_model.float_precision


class TestVIPRSGrid(TestVIPRS):

    def test_viprs_init(self, viprsgrid_model, gdl_object):

        assert viprsgrid_model.m == gdl_object.m

        viprsgrid_model.initialize()

        assert np.all((0. < viprsgrid_model.pi) & (viprsgrid_model.pi < 1.))
        assert np.all((0. < viprsgrid_model.sigma_epsilon) & (viprsgrid_model.sigma_epsilon < 1.))
        assert np.all(viprsgrid_model.tau_beta > 0.)

        for p in (viprsgrid_model.var_gamma, viprsgrid_model.var_mu, viprsgrid_model.var_tau,
                  viprsgrid_model.q, viprsgrid_model.eta, viprsgrid_model.zeta):
            assert p[22].shape == (viprsgrid_model.m, 25)
            assert p[22].dtype == viprsgrid_model.float_precision

        # ...

    def test_viprs_fit(self, viprsgrid_model):

        viprsgrid_model.fit(max_iter=5)

        for p in (viprsgrid_model.pip, viprsgrid_model.post_mean_beta, viprsgrid_model.post_var_beta):
            assert p[22].shape == (viprsgrid_model.m, 25)
            assert p[22].dtype == viprsgrid_model.float_precision


@pytest.mark.xfail(not check_blas_support(), reason="BLAS library not found!")
def test_check_blas_support():
    assert check_blas_support()


@pytest.mark.xfail(not check_omp_support(), reason="OpenMP library not found!")
def test_check_omp_support():
    assert check_omp_support()

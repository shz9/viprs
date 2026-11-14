"""Check covariate-adjusted metrics against a direct least-squares projection."""
import numpy as np
import pandas as pd
import pytest

from viprs.eval.continuous_metrics import partial_correlation, r2_residualized_target


@pytest.mark.parametrize("as_dataframe", [False, True])
def test_covariate_adjusted_metrics(as_dataframe):
    rng = np.random.default_rng(15)
    covariates = rng.normal(size=(100, 2))
    true_val = 3 * covariates[:, 0] + rng.normal(size=100)
    pred_val = 2 * covariates[:, 1] + .4 * true_val + rng.normal(size=100)
    design = np.column_stack((np.ones(100), covariates))
    true_residuals = true_val - design @ np.linalg.lstsq(design, true_val, rcond=None)[0]
    pred_residuals = pred_val - design @ np.linalg.lstsq(design, pred_val, rcond=None)[0]
    if as_dataframe:
        covariates = pd.DataFrame(covariates, columns=["age", "PC1"])

    assert r2_residualized_target(true_val, pred_val, covariates) == pytest.approx(
        np.corrcoef(true_residuals, pred_val)[0, 1] ** 2
    )
    assert partial_correlation(true_val, pred_val, covariates) == pytest.approx(
        np.corrcoef(true_residuals, pred_residuals)[0, 1]
    )

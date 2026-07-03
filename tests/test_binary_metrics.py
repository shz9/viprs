import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from viprs.eval.binary_metrics import f1
from viprs.eval.eval_utils import fit_linear_model


def test_f1_fits_logistic_model_before_thresholding():
    true_val = np.array([0, 1, 0, 1, 0, 1])
    pred_val = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    logit_result = fit_linear_model(
        true_val,
        pd.DataFrame({'pred_val': pred_val}),
        family='binomial',
        add_intercept=True
    )
    expected_labels = (logit_result.predict() >= 0.5).astype(int)

    assert f1(true_val, pred_val) == f1_score(true_val, expected_labels)


@pytest.mark.parametrize('threshold', [-0.1, 1.1])
def test_f1_rejects_invalid_threshold(threshold):
    true_val = np.array([0, 1, 0, 1, 0, 1])
    pred_val = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    with pytest.raises(AssertionError, match='threshold must be between 0 and 1'):
        f1(true_val, pred_val, threshold=threshold)

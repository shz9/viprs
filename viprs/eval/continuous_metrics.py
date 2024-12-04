import numpy as np
import pandas as pd
from .eval_utils import fit_linear_model


def r2(true_val, pred_val):
    """
    Compute the R^2 (proportion of variance explained) between
    the predictions or PRS `pred_val` and the phenotype `true_val`
    
    :param true_val: The response value or phenotype (a numpy vector)
    :param pred_val: The predicted value or PRS (a numpy vector)

    :return: The R^2 value
    """
    from scipy import stats

    _, _, r_val, _, _ = stats.linregress(pred_val, true_val)
    return r_val ** 2


def mse(true_val, pred_val):
    """
    Compute the mean squared error (MSE) between
    the predictions or PRS `pred_val` and the phenotype `true_val`
    
    :param true_val: The response value or phenotype (a numpy vector)
    :param pred_val: The predicted value or PRS (a numpy vector)

    :return: The mean squared error
    """

    return np.mean((pred_val - true_val)**2)


def spearman_r(true_val, pred_val):
    """
    Compute the spearman correlation between the predictions or PRS `pred_val` and the phenotype `true_val`

    :param true_val: The response value or phenotype (a numpy vector)
    :param pred_val: The predicted value or PRS (a numpy vector)
    :return: The spearman correlation
    """

    from scipy import stats
    return stats.spearmanr(true_val, pred_val).statistic


def pearson_r(true_val, pred_val):
    """
    Compute the pearson correlation coefficient between
    the predictions or PRS `pred_val` and the phenotype `true_val`
    
    :param true_val: The response value or phenotype (a numpy vector)
    :param pred_val: The predicted value or PRS (a numpy vector)

    :return: The pearson correlation coefficient
    """
    return np.corrcoef(true_val, pred_val)[0, 1]


def r2_residualized_target(true_val, pred_val, covariates):
    """
    Compute the R^2 (proportion of variance explained) between
    the predictions or PRS `pred_val` and the phenotype `true_val`
    after residualizing the phenotype on a set of covariates.

    :param true_val: The response value or phenotype (a numpy vector)
    :param pred_val: The predicted value or PRS (a numpy vector)
    :param covariates: A pandas table of covariates where the rows are ordered
    the same way as the predictions and response.

    :return: The residualized R^2 value
    """

    resid_true_val = fit_linear_model(true_val, covariates, add_intercept=True)

    return r2(resid_true_val.resid, pred_val)


def incremental_r2(true_val, pred_val, covariates=None, return_all_r2=False):
    """
    Compute the incremental prediction R^2 (proportion of phenotypic variance explained by the PRS).
    This metric is computed by taking the R^2 of a model with covariates+PRS and subtracting from it
    the R^2 of a model with covariates alone covariates.

    :param true_val: The response value or phenotype (a numpy vector)
    :param pred_val: The predicted value or PRS (a numpy vector)
    :param covariates: A pandas table of covariates where the rows are ordered
    the same way as the predictions and response.
    :param return_all_r2: If True, return the R^2 values for the null and full models as well.

    :return: The incremental R^2 value
    """

    if covariates is None:
        add_intercept = False
        covariates = pd.DataFrame(np.ones((true_val.shape[0], 1)), columns=['const'])
    else:
        add_intercept = True

    null_result = fit_linear_model(true_val, covariates, add_intercept=add_intercept)
    full_result = fit_linear_model(true_val, covariates.assign(pred_val=pred_val),
                                   add_intercept=add_intercept)

    if return_all_r2:
        return {
            'Null_R2': null_result.rsquared,
            'Full_R2': full_result.rsquared,
            'Incremental_R2': full_result.rsquared - null_result.rsquared
        }
    else:
        return full_result.rsquared - null_result.rsquared


def partial_correlation(true_val, pred_val, covariates):
    """
    Compute the partial correlation between the phenotype `true_val` and the PRS `pred_val`
    by conditioning on a set of covariates. This metric is computed by first residualizing the
    phenotype and the PRS on a set of covariates and then computing the correlation coefficient
    between the residuals.

    :param true_val: The response value or phenotype (a numpy vector)
    :param pred_val: The predicted value or PRS (a numpy vector)
    :param covariates: A pandas table of covariates where the rows are ordered
    the same way as the predictions and response.

    :return: The partial correlation coefficient
    """

    true_response = fit_linear_model(true_val, covariates, add_intercept=True)
    pred_response = fit_linear_model(pred_val, covariates, add_intercept=True)

    return np.corrcoef(true_response.resid, pred_response.resid)[0, 1]

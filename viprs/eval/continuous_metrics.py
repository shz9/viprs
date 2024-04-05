import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


def r2_stats(r2_val, n):
    """
    Compute the confidence interval and p-value for a given R-squared (proportion of variance
     explained) value.

    This function and the formulas therein are based on the following paper
    by Momin et al. 2023: https://doi.org/10.1016/j.ajhg.2023.01.004 as well as
    the implementation in the R package `PRSmix`:
    https://github.com/buutrg/PRSmix/blob/main/R/get_PRS_acc.R#L63

    :param r2_val: The R^2 value to compute the confidence interval/p-value for.
    :param n: The sample size used to compute the R^2 value

    :return: A dictionary with the R^2 value, the lower and upper values of the confidence interval,
    the p-value, and the standard error of the R^2 metric.

    """

    assert 0. < r2_val < 1., "R^2 value must be between 0 and 1."

    # Compute the variance of the R^2 value:
    r2_var = (4. * r2_val * (1. - r2_val) ** 2 * (n - 2) ** 2) / ((n ** 2 - 1) * (n + 3))

    # Compute the standard errors for the R^2 value
    # as well as the lower and upper values for
    # the confidence interval:
    r2_se = np.sqrt(r2_var)
    lower_r2 = r2_val - 1.97 * r2_se
    upper_r2 = r2_val + 1.97 * r2_se

    # Compute the p-value assuming a Chi-squared distribution with 1 degree of freedom:
    pval = stats.chi2.sf((r2_val / r2_se) ** 2, df=1)

    return {
        'R2': r2,
        'Lower_R2': lower_r2,
        'Upper_R2': upper_r2,
        'P_Value': pval,
        'SE': r2_se,
    }


def r2(true_val, pred_val):
    """
    Compute the R^2 (proportion of variance explained) between
    the predictions or PRS `pred_val` and the phenotype `true_val`
    
    :param true_val: The response value or phenotype (a numpy vector)
    :param pred_val: The predicted value or PRS (a numpy vector)
    """
    _, _, r_val, _, _ = stats.linregress(pred_val, true_val)
    return r_val ** 2


def mse(true_val, pred_val):
    """
    Compute the mean squared error (MSE) between
    the predictions or PRS `pred_val` and the phenotype `true_val`
    
    :param true_val: The response value or phenotype (a numpy vector)
    :param pred_val: The predicted value or PRS (a numpy vector)
    """

    return np.mean((pred_val - true_val)**2)


def pearson_r(true_val, pred_val):
    """
    Compute the pearson correlation coefficient between
    the predictions or PRS `pred_val` and the phenotype `true_val`
    
    :param true_val: The response value or phenotype (a numpy vector)
    :param pred_val: The predicted value or PRS (a numpy vector)
    """
    return np.corrcoef(true_val, pred_val)[0, 1]


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
    """

    if covariates is None:
        covariates = pd.DataFrame(np.ones((true_val.shape[0], 1)), columns=['const'])
    else:
        covariates = sm.add_constant(covariates)

    null_result = sm.OLS(true_val, covariates).fit(disp=0)
    full_result = sm.OLS(true_val, covariates.assign(pred_val=pred_val)).fit(disp=0)

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
    """

    true_response = sm.OLS(true_val, sm.add_constant(covariates)).fit(disp=0)
    pred_response = sm.OLS(pred_val, sm.add_constant(covariates)).fit(disp=0)

    return np.corrcoef(true_response.resid, pred_response.resid)[0, 1]

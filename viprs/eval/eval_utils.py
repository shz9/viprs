import numpy as np


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

    from scipy import stats

    # Compute the p-value assuming a Chi-squared distribution with 1 degree of freedom:
    pval = stats.chi2.sf((r2_val / r2_se) ** 2, df=1)

    return {
        'R2': r2_val,
        'Lower_R2': lower_r2,
        'Upper_R2': upper_r2,
        'P_Value': pval,
        'SE': r2_se,
    }


def fit_linear_model(y, x, family='gaussian', link=None, add_intercept=False):
    """
    Fit a linear model to the data `x` and `y` and return the model object.

    :param y: The independent variable (a numpy vector)
    :param x: The design matrix (a pandas DataFrame)
    :param family: The family of the model. Must be either 'gaussian' or 'binomial'.
    :param link: The link function to use for the model. If None, the default link function.
    :param add_intercept: If True, add an intercept term to the model.
    """

    assert y.shape[0] == x.shape[0], ("The number of rows in the design matrix "
                                      "and the independent variable must match.")
    assert family in ('gaussian', 'binomial'), "The family must be either 'gaussian' or 'binomial'."
    if family == 'binomial':
        assert link in ('logit', 'probit', None), "The link function must be either 'logit', 'probit' or None."

    import statsmodels.api as sm

    if add_intercept:
        x = sm.add_constant(x)

    if family == 'gaussian':
        return sm.OLS(y, x).fit()
    elif family == 'binomial':
        if link == 'logit' or link is None:
            return sm.Logit(y, x).fit(disp=0)
        elif link == 'probit':
            return sm.Probit(y, x).fit(disp=0)

import numpy as np
from sklearn.metrics import (
    auc,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    f1_score
)
from .continuous_metrics import incremental_r2
from scipy.stats import norm
import statsmodels.api as sm
import pandas as pd


def roc_auc(true_val, pred_val):
    """
    Compute the area under the ROC (AUROC) for a model
     that maps from the PRS predictions to the binary phenotype.

    :param true_val: The response value or phenotype (a numpy binary vector with 0s and 1s)
    :param pred_val: The predicted value or PRS (a numpy vector)
    """
    return roc_auc_score(true_val, pred_val)


def pr_auc(true_val, pred_val):
    """
    Compute the area under the Precision-Recall curve for a model
    that maps from the PRS predictions to the binary phenotype.

    :param true_val: The response value or phenotype (a binary numpy vector with 0s and 1s)
    :param pred_val: The predicted value or PRS (a numpy vector)
    """
    precision, recall, thresholds = precision_recall_curve(true_val, pred_val)
    return auc(recall, precision)


def avg_precision(true_val, pred_val):
    """
    Compute the average precision between the PRS predictions and a binary.

    :param true_val: The response value or phenotype (a binary numpy vector with 0s and 1s)
    :param pred_val: The predicted value or PRS (a numpy vector)
    """
    return average_precision_score(true_val, pred_val)


def f1(true_val, pred_val):
    """
    Compute the F1 score between the PRS predictions and a binary phenotype.

    :param true_val: The response value or phenotype (a binary numpy vector with 0s and 1s)
    :param pred_val: The predicted value or PRS (a numpy vector)
    """
    return f1_score(true_val, pred_val)


def mcfadden_r2(true_val, pred_val, covariates=None):
    """
    Compute the McFadden pseudo-R^2 between the PRS predictions and a phenotype.
    If covariates are provided, we compute the incremental pseudo-R^2 by conditioning
    on the covariates.


    :param true_val: The response value or phenotype (a binary numpy vector with 0s and 1s)
    :param pred_val: The predicted value or PRS (a numpy vector)
    :param covariates: A pandas table of covariates where the rows are ordered
    the same way as the predictions and response.
    """

    if covariates is None:
        covariates = pd.DataFrame(np.ones((true_val.shape[0], 1)), columns=['const'])
    else:
        covariates = sm.add_constant(covariates)

    null_result = sm.Logit(true_val, covariates).fit(disp=0)
    full_result = sm.Logit(true_val, covariates.assign(pred_val=pred_val)).fit(disp=0)

    return 1. - (full_result.llf / null_result.llf)


def cox_snell_r2(true_val, pred_val, covariates=None):
    """
    Compute the Cox-Snell pseudo-R^2 between the PRS predictions and a binary phenotype.
    If covariates are provided, we compute the incremental pseudo-R^2 by conditioning
    on the covariates.

    :param true_val: The response value or phenotype (a binary numpy vector with 0s and 1s)
    :param pred_val: The predicted value or PRS (a numpy vector)
    :param covariates: A pandas table of covariates where the rows are ordered
    the same way as the predictions and response.
    """

    if covariates is None:
        covariates = pd.DataFrame(np.ones((true_val.shape[0], 1)), columns=['const'])
    else:
        covariates = sm.add_constant(covariates)

    null_result = sm.Logit(true_val, covariates).fit(disp=0)
    full_result = sm.Logit(true_val, covariates.assign(pred_val=pred_val)).fit(disp=0)
    n = true_val.shape[0]

    return 1. - np.exp(-2 * (full_result.llf - null_result.llf) / n)


def nagelkerke_r2(true_val, pred_val, covariates=None):
    """
    Compute the Nagelkerke pseudo-R^2 between the PRS predictions and a binary phenotype.
    If covariates are provided, we compute the incremental pseudo-R^2 by conditioning
    on the covariates.

    :param true_val: The response value or phenotype (a binary numpy vector with 0s and 1s)
    :param pred_val: The predicted value or PRS (a numpy vector)
    :param covariates: A pandas table of covariates where the rows are ordered
    the same way as the predictions and response.
    """

    if covariates is None:
        covariates = pd.DataFrame(np.ones((true_val.shape[0], 1)), columns=['const'])
    else:
        covariates = sm.add_constant(covariates)

    null_result = sm.Logit(true_val, covariates).fit(disp=0)
    full_result = sm.Logit(true_val, covariates.assign(pred_val=pred_val)).fit(disp=0)
    n = true_val.shape[0]

    # First compute the Cox & Snell R2:
    cox_snell = 1. - np.exp(-2 * (full_result.llf - null_result.llf) / n)

    # Then scale it by the maximum possible R2:
    return cox_snell / (1. - np.exp(2 * null_result.llf / n))


def liability_r2(true_val, pred_val, covariates=None, return_all_r2=False):
    """
    Compute the coefficient of determination (R^2) on the liability scale
    according to Lee et al. (2012) Gene. Epi.
    https://pubmed.ncbi.nlm.nih.gov/22714935/

    The R^2 liability is defined as:
    R_{liability}^2 = R2_{observed}*K*(K-1)/(z^2)

    where R_{observed}^2 is the R^2 on the observed scale and K is the sample prevalence
    and z is the "height of the normal density at the quantile for K".

    If covariates are provided, we compute the incremental pseudo-R^2 by conditioning
    on the covariates.

    :param true_val: The response value or phenotype (a binary numpy vector with 0s and 1s)
    :param pred_val: The predicted value or PRS (a numpy vector)
    :param covariates: A pandas table of covariates where the rows are ordered
    the same way as the predictions and response.
    :param return_all_r2: If True, return the null, full and incremental R2 values.
    """

    # First, obtain the incremental R2 on the observed scale:
    r2_obs = incremental_r2(true_val, pred_val, covariates, return_all_r2=return_all_r2)

    # Second, compute the prevalence and the standard normal quantile of the prevalence:

    k = np.mean(true_val)
    z2 = norm.pdf(norm.ppf(1.-k))**2
    mult_factor = k*(1. - k) / z2

    if return_all_r2:
        return {
            'Null_R2': r2_obs['Null_R2']*mult_factor,
            'Full_R2': r2_obs['Full_R2']*mult_factor,
            'Incremental_R2': r2_obs['Incremental_R2']*mult_factor
        }
    else:
        return r2_obs * mult_factor


def liability_probit_r2(true_val, pred_val, covariates=None, return_all_r2=False):
    """
    Compute the R^2 between the PRS predictions and a binary phenotype on the liability
    scale using the probit likelihood as outlined in Lee et al. (2012) Gene. Epi.
    https://pubmed.ncbi.nlm.nih.gov/22714935/

    The R^2 is defined as:
    R2_{probit} = Var(pred) / (Var(pred) + 1)

    Where Var(pred) is the variance of the predicted liability.

    If covariates are provided, we compute the incremental pseudo-R^2 by conditioning
    on the covariates.

    :param true_val: The response value or phenotype (a binary numpy vector with 0s and 1s)
    :param pred_val: The predicted value or PRS (a numpy vector)
    :param covariates: A pandas table of covariates where the rows are ordered
    the same way as the predictions and response.
    :param return_all_r2: If True, return the null, full and incremental R2 values.
    """

    if covariates is None:
        covariates = pd.DataFrame(np.ones((true_val.shape[0], 1)), columns=['const'])
    else:
        covariates = sm.add_constant(covariates)

    null_result = sm.Probit(true_val, covariates).fit(disp=0)
    full_result = sm.Probit(true_val, covariates.assign(pred_val=pred_val)).fit(disp=0)

    null_var = np.var(null_result.predict())
    null_r2 = null_var / (null_var + 1.)

    full_var = np.var(full_result.predict())
    full_r2 = full_var / (full_var + 1.)

    if return_all_r2:
        return {
            'Null_R2': null_r2,
            'Full_R2': full_r2,
            'Incremental_R2': full_r2 - null_r2
        }
    else:
        return full_r2 - null_r2


def liability_logit_r2(true_val, pred_val, covariates=None, return_all_r2=False):
    """
    Compute the R^2 between the PRS predictions and a binary phenotype on the liability
    scale using the logit likelihood as outlined in Lee et al. (2012) Gene. Epi.
    https://pubmed.ncbi.nlm.nih.gov/22714935/

    The R^2 is defined as:
    R2_{probit} = Var(pred) / (Var(pred) + pi^2 / 3)

    Where Var(pred) is the variance of the predicted liability.

    If covariates are provided, we compute the incremental pseudo-R^2 by conditioning
    on the covariates.

    :param true_val: The response value or phenotype (a binary numpy vector with 0s and 1s)
    :param pred_val: The predicted value or PRS (a numpy vector)
    :param covariates: A pandas table of covariates where the rows are ordered
    the same way as the predictions and response.
    :param return_all_r2: If True, return the null, full and incremental R2 values.
    """

    if covariates is None:
        covariates = pd.DataFrame(np.ones((true_val.shape[0], 1)), columns=['const'])
    else:
        covariates = sm.add_constant(covariates)

    null_result = sm.Probit(true_val, covariates).fit(disp=0)
    full_result = sm.Probit(true_val, covariates.assign(pred_val=pred_val)).fit(disp=0)

    null_var = np.var(null_result.predict())
    null_r2 = null_var / (null_var + (np.pi**2 / 3))

    full_var = np.var(full_result.predict())
    full_r2 = full_var / (full_var + (np.pi**2 / 3))

    if return_all_r2:
        return {
            'Null_R2': null_r2,
            'Full_R2': full_r2,
            'Incremental_R2': full_r2 - null_r2
        }
    else:
        return full_r2 - null_r2

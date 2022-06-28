import numpy as np
from scipy import stats
import statsmodels.api as sm
from sklearn.metrics import auc, roc_auc_score, average_precision_score, precision_recall_curve


def r2(pred_val, true_val):
    """
    Compute the R^2 (proportion of variance explained) between
    the predictions `pred_val` and the response `true_val`
    :param pred_val: The predicted value (a numpy vector)
    :param true_val: The response value (a numpy vector)
    """
    _, _, r_val, _, _ = stats.linregress(pred_val, true_val)
    return r_val ** 2


def mse(pred_val, true_val):
    """
    Compute the mean squared error (MSE) between the predictions `pred_val`
    and the response `true_val`.
    :param pred_val: The predicted value (a numpy vector)
    :param true_val: The response value (a numpy vector)
    """

    return np.sum((pred_val - true_val)**2)


def roc_auc(pred_val, true_val):
    """
    Compute the area under the ROC (AUROC) between the predictions and the response.
    :param pred_val: The predicted value (a numpy vector)
    :param true_val: The response value (a numpy vector)
    """
    return roc_auc_score(true_val, pred_val)


def pr_auc(pred_val, true_val):
    """
    Compute the area under the Precision-Recall curve.
    :param pred_val: The predicted value (a numpy vector)
    :param true_val: The response value (a binary numpy vector)
    """
    precision, recall, thresholds = precision_recall_curve(true_val, pred_val)
    return auc(recall, precision)


def avg_precision(pred_val, true_val):
    """
    Compute the average precision between the predictions and a response.
    :param pred_val: The predicted value (a numpy vector)
    :param true_val: The response value (a binary numpy vector)
    """
    return average_precision_score(true_val, pred_val),


def pearson_r(pred_val, true_val):
    """
    Compute the pearson correlation coefficient between the predictions and the response
    :param pred_val: The predicted value (a numpy vector)
    :param true_val: The response value (a numpy vector)
    """
    return np.corrcoef(pred_val, true_val)[0, 1]


def incremental_r2(pred_val, true_val, covariates):
    """
    Compute the incremental prediction R^2 by conditioning on a set of
    covariates.
    :param pred_val: The predicted value (a numpy vector)
    :param true_val: The response value (a numpy vector)
    :param covariates: A pandas table of covariates where the rows are ordered
    the same way as the predictions and response.
    """

    null_result = sm.OLS(true_val, sm.add_constant(covariates)).fit()
    full_result = sm.OLS(true_val, sm.add_constant(covariates.assign(pred_val=pred_val))).fit()

    return {
        'Null_R2': null_result.rsquared,
        'Full_R2': full_result.rsquared,
        'Incremental_R2': full_result.rsquared - null_result.rsquared
    }


def partial_correlation(pred_val, true_val, covariates):
    """
    Compute the partial correlation by conditioning on a set of covariates.
    :param pred_val: The predicted value (a numpy vector)
    :param true_val: The response value (a numpy vector)
    :param covariates: A pandas table of covariates where the rows are ordered
    the same way as the predictions and response.
    """

    true_response = sm.OLS(true_val, sm.add_constant(covariates)).fit()
    pred_response = sm.OLS(pred_val, sm.add_constant(covariates)).fit()

    return np.corrcoef(true_response.resid, pred_response.resid)[0, 1]

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


def compute_r2(pred_trait, true_trait):
    _, _, r_val, _, _ = stats.linregress(pred_trait, true_trait)
    return r_val ** 2


def compute_auc(pred_trait, true_trait):
    return roc_auc_score(true_trait, pred_trait)



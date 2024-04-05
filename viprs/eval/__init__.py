
from .binary_metrics import *
from .continuous_metrics import *

# Define a dictionary that maps evaluation metric names to their respective functions:
eval_metric_names = {
    'Pearson_R': pearson_r,
    'MSE': mse,
    'R2': r2,
    'Incremental_R2': incremental_r2,
    'Partial_Correlation': partial_correlation,
    'AUROC': roc_auc,
    'AUPRC': pr_auc,
    'Avg_Precision': avg_precision,
    'F1_Score': f1,
    'Liability_R2': liability_r2,
    'Liability_Probit_R2': liability_probit_r2,
    'Liability_Logit_R2': liability_logit_r2,
    'Nagelkerke_R2': nagelkerke_r2,
    'CoxSnell_R2': cox_snell_r2,
    'McFadden_R2': mcfadden_r2
}

# Define a list of metrics that can work with or require
# covariates to be computed:
eval_incremental_metrics = [
    'Incremental_R2',
    'Partial_Correlation',
    'Liability_R2',
    'Liability_Probit_R2',
    'Liability_Logit_R2',
    'Nagelkerke_R2',
    'CoxSnell_R2',
    'McFadden_R2'
]

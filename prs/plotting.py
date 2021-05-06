import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc, roc_auc_score


def plot_history(v, other_h2g=None):

    plt.scatter(np.arange(len(v.history['ELBO'])), v.history['ELBO'])
    plt.xlabel('Iteration')
    plt.ylabel('ELBO')
    plt.show()

    if 'pi' in v.history:
        plt.scatter(np.arange(len(v.history['pi'])), v.history['pi'])
        plt.axhline(1. - v.gdl.pis[0], ls='--')
        plt.xlabel('Iteration')
        plt.ylabel('$\\pi$')
        plt.show()

    for p in ['sigma_epsilon', 'sigma_beta']:
        if p in v.history:
            plt.scatter(np.arange(len(v.history[p])), v.history[p], label=p)
            plt.xlabel('Iteration')
            plt.ylabel(p)
            plt.show()

    if 'heritability' in v.history:
        plt.scatter(np.arange(len(v.history['heritability'])), v.history['heritability'])
        plt.axhline(v.gdl.h2g, ls='--', label='True $h_g^2$')

        if other_h2g is not None:
            color = iter(cm.rainbow(np.linspace(0, 1, len(other_h2g))))
            for k, v in other_h2g.items():
                plt.axhline(v['Estimate'], ls='--', c=next(color), label=k)

        plt.xlabel('Iteration')
        plt.ylabel('$h_g^2$')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()


def plot_heritability_boxplot(res, true_h2):
    # Plot the orbital period with horizontal boxes
    sns.boxplot(x="variable", y="value", data=res.melt(),
                whis=[0, 100], width=.6, palette="vlag")

    # Add in points to show each observation
    sns.stripplot(x="variable", y="value", data=res.melt(),
                  size=4, color=".3", linewidth=0)

    plt.axhline(true_h2, ls='--')
    plt.xlabel("Method")
    plt.ylabel('$h_g^2$ Estimates')
    plt.ylim([0., 1.])
    plt.show()


def plot_prediction_results(res, true_h2g, metric='Pooled R2'):
    rres = res.reset_index()
    plt.figure(figsize=(10, 5))
    sns.boxplot(x="index", y=metric, data=rres,
                whis=[0, 100], width=.6, palette="vlag")
    sns.stripplot(x="index", y=metric, data=rres,
                  size=4, color=".3", linewidth=0)

    plt.axhline(true_h2g, ls='--')
    plt.xlabel("Method")
    plt.ylabel(f'{metric} Estimates')
    plt.ylim([0., 1.])
    plt.show()


def plot_pr_curves(models, gs):

    for m in models:
        gt_y, pred_y = (np.where(gs.mixture_assignment[0])[1] > 0).astype(np.int), m.pip[0]
        precision, recall, _ = precision_recall_curve(gt_y, pred_y)
        plt.plot(recall, precision, marker='.', label=f'{type(m).__name__} (AUC = {auc(recall, precision):.3f})')

    plt.title('PR Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.show()


def plot_roc_curves(models, gs):

    for m in models:
        gt_y, pred_y = (np.where(gs.mixture_assignment[0])[1] > 0).astype(np.int), m.pip[0]
        fpr, tpr, _ = roc_curve(gt_y, pred_y)
        plt.plot(fpr, tpr, marker='.', label=f'{type(m).__name__} (AUC = {roc_auc_score(gt_y, pred_y):.3f})')

    ls_line = np.linspace(0., 1, 100)
    plt.plot(ls_line, ls_line, ls='--')

    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()


def plot_beta_correlation(models, gs):

    for m in models:
        plt.scatter(gs.betas[0], m.inf_beta[0], alpha=.5, marker='.',
                    label=f'{type(m).__name__} (Pearson R2 = {np.corrcoef(gs.betas[0], m.inf_beta[0])[0, 1]:.3f})')

    plt.xlabel('True Betas')
    plt.ylabel('Inferred Betas')
    plt.legend()
    plt.show()

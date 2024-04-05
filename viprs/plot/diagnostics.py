import pandas as pd
import seaborn as sns


def plot_history(prs_model, quantity=None):
    """
    This function plots the optimization history for various model parameters and/or objectives. For
    every iteration step, we generally save quantities such as the ELBO, the heritability, etc. For the purposes
    of debugging and checking model convergence, it is useful to visually observe the trajectory
    of these quantities as a function of training iteration.

    :param prs_model: A `VIPRS` (or its derived classes) object.
    :param quantity: The quantities to plot (e.g. `ELBO`, `heritability`, etc.).
    """

    if quantity is None:
        quantity = prs_model.history.keys()
    elif isinstance(quantity, str):
        quantity = [quantity]

    q_dfs = []

    for attr in quantity:

        df = pd.DataFrame({'Value': prs_model.history[attr]})
        df.reset_index(inplace=True)
        df.columns = ['Step', 'Value']
        df['Quantity'] = attr

        q_dfs.append(df)

    q_dfs = pd.concat(q_dfs)

    g = sns.relplot(
        data=q_dfs, x="Step", y="Value",
        row="Quantity",
        facet_kws={'sharey': False, 'sharex': True},
        kind="scatter",
        marker="."
    )

    return g

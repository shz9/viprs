import numpy as np

# Set up the logger:
import logging
logger = logging.getLogger(__name__)


def select_best_model(viprs_grid_model, validation_gdl=None, criterion='ELBO'):
    """
    From the grid of models that were fit to the data, select the best
    model according to the specified `criterion`. If the criterion is the ELBO,
    the model with the highest ELBO will be selected. If the criterion is
    validation or pseudo-validation, the model with the highest R^2 on the
    held-out validation set will be selected.

    :param viprs_grid_model: An instance of `VIPRSGrid` or `VIPRSGridPathwise` containing the fitted grid
    of VIPRS models.
    :param validation_gdl: An instance of `GWADataLoader` containing data from the validation set.
    :param criterion: The criterion for selecting the best model.
    Options are: (`ELBO`, `validation`, `pseudo_validation`)
    """

    assert criterion in ('ELBO', 'validation', 'pseudo_validation')

    if criterion == 'validation':
        assert validation_gdl is not None, "Validation GWADataLoader must be provided for validation criterion."
    elif criterion == 'pseudo_validation' and validation_gdl is None and viprs_grid_model.validation_std_beta is None:
        raise ValueError("Validation GWADataLoader or standardized betas from a validation set must be "
                         "initialized for the pseudo_validation criterion.")

    # Extract the models that converged successfully:
    models_converged = viprs_grid_model.valid_terminated_models
    best_model_idx = None

    if np.sum(models_converged) < 2:
        raise ValueError("Less than two models converged successfully. Cannot perform model selection.")
    else:

        if criterion == 'ELBO':
            elbo = viprs_grid_model.elbo()
            elbo[~models_converged] = -np.inf
            best_model_idx = np.argmax(elbo)
        elif criterion == 'validation':

            assert validation_gdl is not None
            assert validation_gdl.sample_table is not None
            assert validation_gdl.sample_table.phenotype is not None

            from viprs.eval.continuous_metrics import r2

            prs = viprs_grid_model.predict(test_gdl=validation_gdl)
            prs_r2 = np.array([r2(prs[:, i], validation_gdl.sample_table.phenotype)
                               for i in range(viprs_grid_model.n_models)])
            prs_r2[~models_converged] = -np.inf
            viprs_grid_model.validation_result['Validation_R2'] = prs_r2
            best_model_idx = np.argmax(prs_r2)
        elif criterion == 'pseudo_validation':

            pseudo_r2 = viprs_grid_model.pseudo_validate(validation_gdl)
            pseudo_r2[~models_converged] = -np.inf
            viprs_grid_model.validation_result['Pseudo_Validation_R2'] = pseudo_r2
            best_model_idx = np.argmax(np.nan_to_num(pseudo_r2, nan=0., neginf=0., posinf=0.))

    logger.info(f"> Based on the {criterion} criterion, selected model: {best_model_idx}")
    logger.info("> Model details:\n")
    logger.info(viprs_grid_model.validation_result.iloc[best_model_idx, :])

    # -----------------------------------------------------------------------
    # Update the variational parameters and their dependencies to only select the best model:
    for param in (viprs_grid_model.pip, viprs_grid_model.post_mean_beta, viprs_grid_model.post_var_beta,
                  viprs_grid_model.var_gamma, viprs_grid_model.var_mu, viprs_grid_model.var_tau,
                  viprs_grid_model.eta, viprs_grid_model.zeta, viprs_grid_model.q,
                  viprs_grid_model._log_var_tau):
        for c in param:
            param[c] = param[c][:, best_model_idx]

    # Update the eta diff:
    try:
        for c in viprs_grid_model.eta_diff:
            viprs_grid_model.eta_diff[c] = viprs_grid_model.eta_diff[c][:, best_model_idx]
    except IndexError:
        # Don't need to update this for the VIPRSGridPathwise model.
        pass

    # Update sigma_epsilon:
    viprs_grid_model.sigma_epsilon = viprs_grid_model.sigma_epsilon[best_model_idx]

    # Update sigma_g:
    viprs_grid_model._sigma_g = viprs_grid_model._sigma_g[best_model_idx]

    # Update sigma beta:
    if isinstance(viprs_grid_model.tau_beta, dict):
        for c in viprs_grid_model.tau_beta:
            viprs_grid_model.tau_beta[c] = viprs_grid_model.tau_beta[c][:, best_model_idx]
    else:
        viprs_grid_model.tau_beta = viprs_grid_model.tau_beta[best_model_idx]

    # Update pi

    if isinstance(viprs_grid_model.pi, dict):
        for c in viprs_grid_model.pi:
            viprs_grid_model.pi[c] = viprs_grid_model.pi[c][:, best_model_idx]
    else:
        viprs_grid_model.pi = viprs_grid_model.pi[best_model_idx]

    # -----------------------------------------------------------------------

    # Set the number of models to 1:
    viprs_grid_model.n_models = 1

    # Update the fixed parameters of the model:
    viprs_grid_model.set_fixed_params(
        viprs_grid_model.grid_table.iloc[best_model_idx].to_dict()
    )

    # -----------------------------------------------------------------------

    return viprs_grid_model


def bayesian_model_average(viprs_grid_model, normalization='softmax'):
    """
    Use Bayesian model averaging (BMA) to obtain a weighing scheme for the
     variational parameters of a grid of VIPRS models. The parameters of each model in the grid
     are assigned weights proportional to their final ELBO.

    :param viprs_grid_model: An instance of `VIPRSGrid` or `VIPRSGridPathwise` containing the fitted grid
    of VIPRS models.
    :param normalization: The normalization scheme for the final ELBOs.
    Options are (`softmax`, `sum`).
    :raises KeyError: If the normalization scheme is not recognized.
    """

    if viprs_grid_model.n_models < 2:
        return viprs_grid_model

    if np.sum(viprs_grid_model.valid_terminated_models) < 1:
        raise ValueError("No models converged successfully. "
                         "Cannot average models.")

    # Extract the models that converged successfully:
    models_to_keep = np.where(viprs_grid_model.valid_terminated_models)[0]

    elbos = viprs_grid_model.elbo()

    if normalization == 'softmax':
        from scipy.special import softmax
        weights = np.array(softmax(elbos))
    elif normalization == 'sum':
        weights = np.array(elbos)

        # Correction for negative ELBOs:
        weights = weights - weights.min() + 1.
        weights /= weights.sum()
    else:
        raise KeyError("Normalization scheme not recognized. "
                       "Valid options are: `softmax`, `sum`. "
                       "Got: {}".format(normalization))

    logger.info("Averaging PRS models with weights:", weights)

    # Average the model parameters:
    for param in (viprs_grid_model.var_gamma, viprs_grid_model.var_mu, viprs_grid_model.var_tau,
                  viprs_grid_model.q):
        for c in param:
            param[c] = (param[c][:, models_to_keep] * weights).sum(axis=1)

    viprs_grid_model.eta = viprs_grid_model.compute_eta()
    viprs_grid_model.zeta = viprs_grid_model.compute_zeta()

    # Update posterior moments:
    viprs_grid_model.update_posterior_moments()

    # Update the log of the variational tau parameters:
    viprs_grid_model._log_var_tau = {c: np.log(viprs_grid_model.var_tau[c])
                                     for c in viprs_grid_model.var_tau}

    # Update the hyperparameters based on the averaged weights
    import copy
    # TODO: double check to make sure this makes sense.
    fix_params_before = copy.deepcopy(viprs_grid_model.fix_params)
    viprs_grid_model.fix_params = {}
    viprs_grid_model.m_step()
    viprs_grid_model.fix_params = fix_params_before

    # -----------------------------------------------------------------------

    # Set the number of models to 1:
    viprs_grid_model.n_models = 1

    # -----------------------------------------------------------------------

    return viprs_grid_model

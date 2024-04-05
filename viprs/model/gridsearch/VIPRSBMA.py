
from .VIPRSGrid import VIPRSGrid
import numpy as np
from scipy.special import softmax


class VIPRSBMA(VIPRSGrid):
    """
    The `VIPRSBMA` class is an extension of the `VIPRSGrid` class that
    implements Bayesian model averaging for the `VIPRS` models in the grid.
    Bayesian model averaging is a technique that allows us to combine the
    results of multiple models by weighting them according to their evidence.
    In this context, we weigh the model by their final ELBO values.

    For more details on the BMA procedure implemented here, refer to the
    Supplementary material of:

    > Zabad S, Gravel S, Li Y. Fast and accurate Bayesian polygenic risk modeling with variational inference.
    Am J Hum Genet. 2023 May 4;110(5):741-761. doi: 10.1016/j.ajhg.2023.03.009.
    Epub 2023 Apr 7. PMID: 37030289; PMCID: PMC10183379.

    """

    def __init__(self,
                 gdl,
                 grid,
                 **kwargs):
        """

        Initialize the `VIPRSBMA` model.

        :param gdl: An instance of `GWADataLoader`
        :param grid: An instance of `HyperparameterGrid`
        :param kwargs: Additional keyword arguments for the VIPRS model
        """

        super().__init__(gdl, grid=grid, **kwargs)

    def average_models(self, normalization='softmax'):
        """
        Use Bayesian model averaging (BMA` to obtain final weights for each parameter.
        We average the weights by using the final ELBO for each model.
        
        :param normalization: The normalization scheme for the final ELBOs. Options are (`softmax`, `sum`). 
        :raises KeyError: If the normalization scheme is not recognized.
        """

        elbos = self.history['ELBO'][-1]

        if normalization == 'softmax':
            weights = np.array(softmax(elbos))
        elif normalization == 'sum':
            weights = np.array(elbos)

            # Correction for negative ELBOs:
            weights = weights - weights.min() + 1.
            weights /= weights.sum()
        else:
            raise KeyError("Normalization scheme not recognized. Valid options are: `softmax`, `sum`. "
                           "Got: {}".format(normalization))

        if int(self.verbose) > 1:
            print("Averaging PRS models with weights:", weights)

        for param in (self.pip, self.post_mean_beta, self.post_var_beta,
                      self.var_gamma, self.var_mu, self.var_tau,
                      self.eta, self.zeta, self.q):

            for c in param:
                param[c] = (param[c]*weights).sum(axis=1)

        return self

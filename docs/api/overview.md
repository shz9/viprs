## Models

* [BayesPRSModel](model/BayesPRSModel.md): A base class for all Bayesian PRS models.
* [VIPRS](model/VIPRS.md): Implementation of VIPRS with the "**spike-and-slab**" prior.
    *  Implementation of VIPRS with **other priors**:
        * [VIPRSMix](model/VIPRSMix.md): VIPRS with a sparse Gaussian mixture prior.
* **Hyperparameter Tuning**: Models/modules for performing hyperparameter search with `VIPRS` models.
    * [Hyperparameter grid](model/gridsearch/HyperparameterGrid.md): A utility class to help construct grids over model hyperparameters.
    * [HyperparameterSearch](model/gridsearch/HyperparameterSearch.md)
    * [VIPRSGrid](model/gridsearch/VIPRSGrid.md)
        * [VIPRSGridSearch](model/gridsearch/VIPRSGridSearch.md)
        * [VIPRSBMA](model/gridsearch/VIPRSBMA.md)
* **Baseline Models**:
    * [LDPredInf](model/LDPredInf.md): Implementation of the LDPred-inf model.

## Model Evaluation

* [Binary metrics](eval/binary_metrics.md): Evaluation metrics for binary (case-control) phenotypes.
* [Continuous metrics](eval/continuous_metrics.md): Evaluation metrics for continuous phenotypes.
* [Pseudo metrics](eval/pseudo_metrics.md): Evaluation metrics based on GWAS summary statistics.

## Utilities

* [Data utilities](utils/data_utils.md): Utilities for downloading and processing relevant data.
* [Compute utilities](utils/compute_utils.md): Utilities for computing various statistics / quantities over python data structures.
* [Exceptions](utils/exceptions.md): Custom exceptions used in the package.
* [OptimizeResult](utils/OptimizeResult.md): A class to store the result of an optimization routine.

## Plotting

* [Diagnostic plots](plot/diagnostics.md): Functions for plotting various quantities / results from VIPRS or other PRS models.

In addition to the python package interface, users may also opt to use `viprs` via commandline scripts.
The commandline interface is designed to be user-friendly and to provide a variety of options for the user to 
customize the inference process. 

When you install `viprs` using `pip`, the commandline scripts are automatically installed on your system and 
are available for use. The following scripts are meant to facilitate the entire pipeline of polygenic score inference, 
from fitting and estimating the posterior distribution of the variant effect sizes to predicting the PRS for a set of 
test individuals and evaluating the performance of the PRS predictions on held out samples.

* [`viprs_fit`](viprs_fit.md): This script is used to fit the variational PRS model to the GWAS summary statistics and to estimate the 
    posterior distribution of the variant effect sizes. The script provides a variety of options for the user to 
    customize the inference process, including the choice of prior distributions and the choice of 
    optimization algorithms.

* [`viprs_score`](viprs_score.md): This script is used to predict the PRS for a set of individuals using the 
    estimated variant effect sizes from the `viprs_fit` script. This is the script that generates the PRS per
    individual.

* [`viprs_evaluate`](viprs_evaluate.md): This script is used to evaluate the performance of the PRS predictions 
    using the PRS computed in the previous step. The script provides a variety of 
    options for the user to customize the evaluation process, including the choice of performance metrics and 
    the choice of evaluation datasets.


## TODO
-  Create a `nextflow` pipeline that runs all of the above steps in a single command.

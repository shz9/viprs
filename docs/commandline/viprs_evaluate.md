Evaluate Predictive Performance of PRS (`viprs_evaluate`)
---

The `viprs_evaluate` script is used to evaluate the performance of the PRS predictions using the PRS computed in 
the previous step. The script provides a variety of options for the user to customize the evaluation process, 
including the choice of performance metrics and the choice of evaluation datasets.

A full listing of the options available for the `viprs_evaluate` script can be found by running the 
following command in your terminal:

```bash
viprs_evaluate -h
```

Which outputs the following help message:

```bash

          **********************************************
                     _____                              
             ___   _____(_)________ ________________    
             __ | / /__  / ___  __ \__  ___/__  ___/    
             __ |/ / _  /  __  /_/ /_  /    _(__  )     
             _____/  /_/   _  .___/ /_/     /____/      
                           /_/                          
                                                        
          Variational Inference of Polygenic Risk Scores
            Version: 0.1.3 | Release date: April 2025   
              Author: Shadi Zabad, McGill University    
          **********************************************
          < Evaluate Prediction Accuracy of PRS Models >

usage: viprs_evaluate [-h] --prs-file PRS_FILE --phenotype-file PHENO_FILE [--phenotype-col PHENO_COL]
                      [--phenotype-likelihood {binomial,gaussian,infer}] [--keep KEEP] --output-file OUTPUT_FILE
                      [--metrics METRICS [METRICS ...]] [--covariates-file COVARIATES_FILE]
                      [--log-level {CRITICAL,WARNING,INFO,DEBUG,ERROR}]

Commandline arguments for evaluating polygenic scores

optional arguments:
  -h, --help            show this help message and exit
  --prs-file PRS_FILE   The path to the PRS file (expected format: FID IID PRS, tab-separated)
  --phenotype-file PHENO_FILE
                        The path to the phenotype file. The expected format is: FID IID phenotype (no header), tab-separated.
  --phenotype-col PHENO_COL
                        The column index for the phenotype in the phenotype file (0-based index).
  --phenotype-likelihood {binomial,gaussian,infer}
                        The phenotype likelihood ("gaussian" for continuous, "binomial" for case-control). If not set, will be inferred
                        automatically based on the phenotype file.
  --keep KEEP           A plink-style keep file to select a subset of individuals for the evaluation.
  --output-file OUTPUT_FILE
                        The output file where to store the evaluation metrics (with no extension).
  --metrics METRICS [METRICS ...]
                        The evaluation metrics to compute (default: all available metrics that are relevant for the phenotype). For a full
                        list of supported metrics, check the documentation.
  --covariates-file COVARIATES_FILE
                        A file with covariates for the samples included in the analysis. This tab-separated file should not have a header
                        and the first two columns should be the FID and IID of the samples.
  --log-level {CRITICAL,WARNING,INFO,DEBUG,ERROR}
                        The logging level for the console output.

```
Compute Polygenic Scores using inferred variant effect sizes (`viprs_score`)
---

The `viprs_score` script is used to compute the polygenic risk scores (PRS) for a set of individuals 
using the estimated variant effect sizes from the `viprs_fit` script. This is the script that generates 
the PRS per individual.

A full listing of the options available for the `viprs_score` script can be found by running the 
following command in your terminal:

```bash
viprs_score -h
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
          < Compute Polygenic Scores for Test Samples > 

usage: viprs_score [-h] -f FIT_FILES --bfile BED_FILES --output-file OUTPUT_FILE [--temp-dir TEMP_DIR] [--keep KEEP] [--extract EXTRACT]
                   [--backend {xarray,plink}] [--threads THREADS] [--compress] [--log-level {WARNING,CRITICAL,DEBUG,INFO,ERROR}]

Commandline arguments for computing polygenic scores

options:
  -h, --help            show this help message and exit
  -f FIT_FILES, --fit-files FIT_FILES
                        The path to the file(s) with the output parameter estimates from VIPRS. You may use a wildcard here if fit files are stored per-
                        chromosome (e.g. "prs/chr_*.fit")
  --bfile BED_FILES     The BED files containing the genotype data. You may use a wildcard here (e.g. "data/chr_*.bed")
  --output-file OUTPUT_FILE
                        The output file where to store the polygenic scores (with no extension).
  --temp-dir TEMP_DIR   The temporary directory where to store intermediate files.
  --keep KEEP           A plink-style keep file to select a subset of individuals for the test set.
  --extract EXTRACT     A plink-style extract file to select a subset of SNPs for scoring.
  --backend {xarray,plink}
                        The backend software used for computations with the genotype matrix.
  --threads THREADS     The number of threads to use for computations.
  --compress            Compress the output file
  --log-level {WARNING,CRITICAL,DEBUG,INFO,ERROR}
                        The logging level for the console output.

```
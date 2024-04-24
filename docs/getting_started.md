`viprs` is a `python` package for fitting Bayesian Polygenic Risk Score (PRS) models to summary statistics 
derived from Genome-wide Association Studies (GWASs). To showcase the interfaces and functionalities of the package 
as well as the data structures that power it, we will start with a simple example. 

!!! note 
    This example is designed to highlight the features of the package and the python API. If you'd like to 
    use the commandline interface, please refer to the [Command Line Scripts](commandline/overview.md) documentation.

Generally, summary statistics-based PRS methods require access to:

* GWAS summary statistics for the trait of interest 
* Linkage-Disequilibrium (LD) matrices from an appropriately-matched reference panel (e.g. 
from the 1KG dataset or UK Biobank). 

For the first item, we will use summary statistics for Standing Height (`EFO_0004339`) from the `fastGWA` 
[catalogue](https://yanglab.westlake.edu.cn/data/ukb_fastgwa/imp/pheno/50). 
For the second item, we will use genotype data on chromosome 22 for a subset of 378 European samples from the 
1KG project. This small dataset is shipped with the python package `magenpy`.

To start, let's import the required `python` packages:

```python linenums="1"
import magenpy as mgp
import viprs as vp
```

Then, we will use `magenpy` to read the 1KG genotype dataset and *automatically* match it with the GWAS 
summary statistics from `fastGWA`:

```python linenums="1"
# Load genotype and GWAS summary statistics data (chromosome 22):
gdl = mgp.GWADataLoader(bed_files=mgp.tgp_eur_data_path(),  # Path of the genotype data
                        sumstats_files=mgp.ukb_height_sumstats_path(),  # Path of the summary statistics
                        sumstats_format="fastGWA")  # Specify the format of the summary statistics
```

Once the genotype and summary statistics data are read by `magenpy`, we can go ahead and compute 
the LD (or SNP-by-SNP correlation) matrix:

```python linenums="1"
# Compute LD using the shrinkage estimator (Wen and Stephens 2010):
gdl.compute_ld("shrinkage",
               output_dir="temp",
               genetic_map_ne=11400, # effective population size (Ne)
               genetic_map_sample_size=183,
               threshold=1e-3)
```

Because of the small sample size of the reference panel, here we recommend using the `shrinkage` estimator 
for LD from Wen and Stephens (2010). The shrinkage estimator results in compact and sparse LD matrices that are 
more robust than the sample LD. The estimator requires access to information about the genetic map, such as 
the position of each SNP in centi Morgan, the effective population size, and the sample size used to 
estimate the genetic map.

Given the LD information from the reference panel, we can next fit the VIPRS model to the summary statistics data:

```python linenums="1"
# Initialize VIPRS, passing it the GWADataLoader object
v = vp.VIPRS(gdl)
# Invoke the .fit() method to obtain posterior estimates
v.fit()
```

Once the model converges, we can generate PRS estimates for height for the 1KG samples by simply 
invoking the `.predict()` method:

```python linenums="1"
v.predict()
```

```
array([ 0.01944202,  0.00597704,  0.07329462, ..., 0.06666187,  0.05251297,  0.00359018])
```
These are the polygenic scores for height for the European samples in the 1KG dataset! 

To examine posterior estimates for the model parameters, you can simply invoke the `.to_table()` method:

```python linenums="1"
v.to_table()
```

```
       CHR         SNP A1 A2       PIP          BETA      VAR_BETA
 0       22    rs131538  A  G  0.006107 -5.955517e-06  1.874619e-08
 1       22   rs9605903  C  T  0.005927  5.527188e-06  1.774252e-08
 2       22   rs5746647  G  T  0.005015  1.194178e-07  1.120063e-08
 3       22  rs16980739  T  C  0.008331 -1.335695e-05  3.717944e-08
 4       22   rs9605923  A  T  0.006181  6.334971e-06  1.979157e-08
 ...    ...         ... .. ..       ...           ...           ...
 15930   22   rs8137951  A  G  0.006367 -6.880591e-06  2.059650e-08
 15931   22   rs2301584  A  G  0.179406 -7.234545e-04  2.597197e-06
 15932   22   rs3810648  G  A  0.008000  1.222151e-05  3.399927e-08
 15933   22   rs2285395  A  G  0.005356  3.004282e-06  1.349082e-08
 15934   22  rs28729663  A  G  0.005350 -2.781053e-06  1.351239e-08

 [15935 rows x 7 columns]
```

Here, `PIP` is the **P**osterior **I**nclusion **P**robability under the variational density, while 
`BETA` and `VAR_BETA` are the posterior mean and variance for the effect size, respectively. 
For the purposes of prediction, we only need the `BETA` column. You can also examine the 
inferred hyperparameters of the model by invoking the `.to_theta_table()` method:

```python linenums="1"
v.to_theta_table()
```

```
           Parameter     Value
 0  Residual_variance  0.994231
 1       Heritability  0.005736
 2  Proportion_causal  0.015887
 3         sigma_beta  0.000021
```

Note that here, the SNP heritability only considers the contribution of variants on 
chromosome 22.
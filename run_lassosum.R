# Created by: szabad
# Created on: 10/25/20
# Following this tutorial:
# https://choishingwan.github.io/PRS-Tutorial/lassosum/

# ---------------------------------------------------------
# Libraries:

library(lassosum)
# Prefer to work with data.table as it speeds up file reading
library(data.table)
library(methods)
library(magrittr)
library(parallel)
library(optparse)

# ---------------------------------------------------------
# Options:

option_list <- list(
  make_option(c("-b", "--bed"), type="character", default=NULL,
              help="Path to bed files", metavar="character"),
  make_option(c("-s", "--sumstats"), type="character",
              help="Path to summary statistics file", metavar="character")
)

opt_parser <- OptionParser(option_list=option_list);
opt <- parse_args(opt_parser)


library(lassosum)
# Prefer to work with data.table as it speeds up file reading
library(data.table)
library(methods)
library(magrittr)
# For multi-threading, you can use the parallel package and
# invoke cl which is then passed to lassosum.pipeline
library(parallel)
# This will invoke 2 threads.
cl <- makeCluster(2)

dir <- "/Users/szabad/PycharmProjects/vemPRS/data/post-qc/"

sum.stat <- paste0(dir, "Height.QC.gz")
bfile <- paste0(dir, "EUR.QC")
# Read in and process the covariates
covariate <- fread(paste0(dir, "EUR.cov"))
pcs <- fread(paste0(dir, "EUR.eigenvec")) %>%
  setnames(., colnames(.), c("FID","IID", paste0("PC",1:6)))
# Need as.data.frame here as lassosum doesn't handle data.table
# covariates very well
cov <- merge(covariate, pcs)

# We will need the EUR.hg19 file provided by lassosum
# which are LD regions defined in Berisa and Pickrell (2015) for the European population and the hg19 genome.
ld.file <- paste0(dir, "EUR.hg19")
# output prefix
prefix <- "EUR"
# Read in the target phenotype file
target.pheno <- fread(paste0(dir, "EUR.height"))[,c("FID", "IID", "Height")]
# Read in the summary statistics
ss <- fread(sum.stat)
# Remove P-value = 0, which causes problem in the transformation
ss <- ss[!P == 0]
# Transform the P-values into correlation
cor <- p2cor(p = ss$P,
             n = ss$N,
             sign = log(ss$OR)
)
fam <- fread(paste0(bfile, ".fam"))
fam[,ID:=do.call(paste, c(.SD, sep=":")),.SDcols=c(1:2)]


# Run the lassosum pipeline
# The cluster parameter is used for multi-threading
# You can ignore that if you do not wish to perform multi-threaded processing
out <- lassosum.pipeline(
  cor = cor,
  chr = ss$CHR,
  pos = ss$BP,
  A1 = ss$A1,
  A2 = ss$A2,
  ref.bfile = bfile,
  test.bfile = bfile,
  #LDblocks = ld.file,
  cluster=cl
)
# Store the R2 results
target.res <- validate(out, pheno = target.pheno, covar=cov)
# Get the maximum R2
r2 <- max(target.res$validation.table$value)^2

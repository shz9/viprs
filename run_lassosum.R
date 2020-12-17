library(argparser)
library(lassosum)
library(data.table)
# For multi-threading, you can use the parallel package and 
# invoke cl which is then passed to lassosum.pipeline
#library(parallel)
# This will invoke 2 threads. 
#cl <- makeCluster(2)

p <- arg_parser("Run LASSOSUM PRS")
p <- add_argument(p, "sumstats", help="Summary statistics file")
p <- add_argument(p, "ref", help="Reference panel bed files (e.g. 1000 Genomes)")
p <- add_argument(p, "test", help="Test panel bed files")
#p <- add_argument(p, "targetpheno", help="Target phenotypes (used for cross-validation)")
p <- add_argument(p, "--remove", help="A file containing the FID/IID of individuals to remove from test panel")
p <- add_argument(p, "--ld", help="LD regions file to use", default="EUR.hg19")
p <- add_argument(p, "out", help="Output file prefix")

argv <- parse_args(p)

# Read in the target phenotype file
#target.pheno <- fread(argv$targetpheno)

# Read in the summary statistics

ss <- fread(argv$sumstats)
# Remove P-value = 0, which causes problem in the transformation
ss <- ss[!PVAL == 0]
# Transform the P-values into correlation
cor <- p2cor(p = ss$PVAL,
             n = ss$N,
             sign = ss$BETA)

if (is.na(argv$remove)){
  remove <- NULL
} else {
  remove <- argv$remove
}

# Run the lassosum pipeline

out <- lassosum.pipeline(
  cor = cor,
  chr = ss$CHR,
  snp = ss$SNP,
  pos = ss$BP,
  A1 = ss$A1,
  A2 = ss$A2,
  ref.bfile = argv$ref,
  test.bfile = argv$test,
  remove.test = remove,
  LDblocks = argv$ld,
  exclude.ambiguous = F
#  cluster=cl
)

# validate:
#target.res <- validate(out, pheno = target.pheno)
target.res <- pseudovalidate(out)

out_df <- out$sumstats
out_df$effectSize <- target.res$best.beta

write.table(out_df, paste0(argv$out, ".snpEffect"), row.names = F)


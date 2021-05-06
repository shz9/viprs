library(bigsnpr)

#p <- arg_parser("Run LDPred2 PRS")
#p <- add_argument(p, "sumstats", help="Summary statistics file")
#p <- add_argument(p, "ref", help="Reference panel bed files (e.g. 1000 Genomes)")
#p <- add_argument(p, "test", help="Test panel bed files")
#p <- add_argument(p, "targetpheno", help="Target phenotypes (used for cross-validation)")
#p <- add_argument(p, "--remove", help="A file containing the FID/IID of individuals to remove from test panel")
#p <- add_argument(p, "--ld", help="LD regions file to use", default="EUR.hg19")
#p <- add_argument(p, "out", help="Output file prefix")

#argv <- parse_args(p)

# --------------------

# Read the reference genotype data:
rds <- snp_readBed("/data/post-qc/EUR.QC.bed")
bfiles <- snp_attach(rds)

# Read the summary statistics:
sumstats <- bigreadr::fread2("/data/post-qc/Height.QC.gz")
# LDpred 2 require the header to follow the exact naming
names(sumstats) <-
  c("chr",
    "pos",
    "rsid",
    "a1",
    "a0",
    "n_eff",
    "beta_se",
    "p",
    "OR",
    "INFO",
    "MAF")
sumstats$beta <- log(sumstats$OR)

map <- bfiles$map[-(2:3)]
names(map) <- c("chr", "pos", "a0", "a1")
info_snp <- snp_match(sumstats, map)


genotype <- bfiles$genotypes
# Rename the data structures
CHR <- map$chr
POS <- map$pos

# --------------------
tmp <- tempfile(tmpdir = "/temp/LDPred2")
POS2 <- snp_asGeneticPos(bfiles$map$chromosome, bfiles$map$physical.pos,
                         dir = "/Users/szabad/PycharmProjects/data/1000-genomes-genetic-maps-master/interpolated_OMNI")
for (chr in 1:22) {
  # Extract SNPs that are included in the chromosome
  ind.chr <- which(info_snp$chr == chr)
  ind.chr2 <- info_snp$`_NUM_ID_`[ind.chr]
  # Calculate the LD
  corr0 <- snp_cor(
    genotype,
    ind.col = ind.chr2,
    infos.pos = POS2[ind.chr2],
    size = 3 / 1000
  )
  if (chr == 1) {
    ld <- Matrix::colSums(corr0^2)
    corr <- as_SFBM(corr0, tmp)
  } else {
    ld <- c(ld, Matrix::colSums(corr0^2))
    corr$add_columns(corr0, nrow(corr))
  }
}

# --------------------

df_beta <- info_snp[,c("beta", "beta_se", "n_eff", "_NUM_ID_")]
ldsc <- snp_ldsc(   ld, 
                    length(ld), 
                    chi2 = (df_beta$beta / df_beta$beta_se)^2,
                    sample_size = df_beta$n_eff, 
                    blocks = NULL)
h2_est <- ldsc[["h2"]]

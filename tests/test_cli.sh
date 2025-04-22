#!/bin/bash

if [[ -t 1 ]]; then
  set -e  # Enable exit on error, only in non-interactive sessions
fi

BFILE_PATH=$(python3 -c "import magenpy as mgp; print(mgp.tgp_eur_data_path())")
SUMSTATS_PATH=$(python3 -c "import magenpy as mgp; print(mgp.ukb_height_sumstats_path())")
LD_BLOCKS_PATH="https://bitbucket.org/nygcresearch/ldetect-data/raw/ac125e47bf7ff3e90be31f278a7b6a61daaba0dc/EUR/fourier_ls-all.bed"

# -------------------------------------------------------------------
# Use `magenpy_ld` cli script to estimate LD:

echo "> Estimating LD using the block estimator:"
magenpy_ld --estimator "block" \
           --bfile "$BFILE_PATH" \
           --ld-blocks "$LD_BLOCKS_PATH" \
           --output-dir "output/ld_block/"

# Check that there's a directory called "output/ld_block/chr_22/":
if [ ! -d "output/ld_block/chr_22/" ]; then
  echo "Error: The output directory was not created."
  exit 1
fi

# Check that the directory contains both `.zgroup` and `.zatrs` files:
if [ ! -f "output/ld_block/chr_22/.zgroup" ] || [ ! -f "output/ld_block/chr_22/.zattrs" ]; then
  echo "Error: The output directory does not contain the expected files."
  exit 1
fi

# -------------------------------------------------------------------
# Test the `viprs_fit` cli script:

echo -e "\n> Testing the VIPRS_EM model... \n"

viprs_fit -l "output/ld_block/chr_22/" \
          -s "$SUMSTATS_PATH" \
          --sumstats-format "fastgwa" \
          --output-dir "output/viprs_fit/" \
          --output-profiler-metrics

# Check that the output file exists:
if [ ! -f "output/viprs_fit/VIPRS_EM.fit.gz" ]; then
  echo "Error: The output file was not created."
  exit 1
fi

echo -e "\n> Testing the VIPRS_GS model... \n"

viprs_fit -l "output/ld_block/chr_22/" \
          -s "$SUMSTATS_PATH" \
          --sumstats-format "fastgwa" \
          --hyp-search "GS" \
          --pi-steps 10 \
          --output-dir "output/viprs_fit/" \
          --output-profiler-metrics

# Check that the output file exists:
if [ ! -f "output/viprs_fit/VIPRS_GS.fit.gz" ]; then
  echo "Error: The output file was not created."
  exit 1
fi

# -------------------------------------------------------------------
# Test the `viprs_score` cli script:

echo -e "\n> Testing the VIPRS scoring... \n"
viprs_score -f "output/viprs_fit/VIPRS_EM.fit.gz" \
            --bfile "$BFILE_PATH" \
            --output-file "output/viprs_score/scores" \
            --compress

# Check that the output file exists:
if [ ! -f "output/viprs_score/scores.prs.gz" ]; then
  echo "Error: The output file was not created."
  exit 1
fi

# -------------------------------------------------------------------
# Test the `viprs_evaluate` cli script:

# TODO: Expand this to include proper testing of evaluation pipeline
viprs_evaluate -h

# -------------------------------------------------------------------
# Clean up after computation:
rm -rf output/
rm -rf temp/

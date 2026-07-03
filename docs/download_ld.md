Linkage-Disequilibrium (LD) matrices, which record pairwise correlations between 
genetic variants, are required as input to the `VIPRS` model. To facilitate running the model 
on GWAS data from diverse ancestries, we computed LD matrices for 6 continental populations represented in 
the UK Biobank.

The sample sizes reported below are restricted to unrelated individuals in the UK Biobank. The matrices were
computed using the `block` LD estimator, where we only record pairwise correlations between variants in the
same LD block. The LD blocks are defined by [`LDetect`](https://bitbucket.org/nygcresearch/ldetect-data/src/master/).
The matrices were computed using the sister package [`magenpy`](https://shz9.github.io/magenpy/) and were then
quantized to `int8` data type for enhanced compressibility.

For European samples, we also provide LD matrices that record pairwise correlations for up to 18 million variants.
This matrix is available for download via [Zenodo](https://zenodo.org/records/14614207).

For more details on QC criteria, data preparation, etc., please consult our manuscript:

> Zabad, S., Haryan, C. A., Gravel, S., Misra, S., Li, Y. (2025). **Toward whole-genome inference of polygenic scores with fast and memory-efficient algorithms.** 
The American Journal of Human Genetics, 112(7), 1528-1546. https://doi.org/10.1016/j.ajhg.2025.05.002

!!! tip "New since July 2026: cloud-streamable LD matrices"
    VIPRS can now read the UK Biobank LD matrices directly from
    [Hugging Face](https://huggingface.co/datasets/shz9/ukb-ld/tree/main) using `hf://` paths.
    This means you can point VIPRS to the LD matrices without pre-downloading and extracting the archive files.

## Streaming LD matrices from Hugging Face 🤗

The LD matrices are available as chromosome-level zip files on Hugging Face. To use this feature,
install `huggingface_hub` in the environment where you run VIPRS:

```bash
python -m pip install huggingface_hub
```

You can then pass the appropriate `hf://` wildcard path to the `-l` / `--ld-panel` argument, for example:

```bash
viprs_fit \
  --ld-panel "hf://datasets/shz9/ukb-ld/EUR/chr_*.zip" \
  --sumstats path/to/sumstats.txt \
  --output-dir output/viprs_fit
```

The available ancestry-specific Hugging Face paths are:

| Code  | Ancestry group      | Sample size | Hugging Face LD path                         |
|:-----:|:--------------------|:-----------:|:---------------------------------------------|
| `EUR` | European            |   362446    | `hf://datasets/shz9/ukb-ld/EUR/chr_*.zip`    |
| `CSA` | Central/South Asian |    8284     | `hf://datasets/shz9/ukb-ld/CSA/chr_*.zip`    |
| `AFR` | African             |    6255     | `hf://datasets/shz9/ukb-ld/AFR/chr_*.zip`    |
| `EAS` | East Asian          |    2700     | `hf://datasets/shz9/ukb-ld/EAS/chr_*.zip`    |
| `MID` | Middle Eastern      |    1567     | `hf://datasets/shz9/ukb-ld/MID/chr_*.zip`    |
| `AMR` | Admixed American    |     987     | `hf://datasets/shz9/ukb-ld/AMR/chr_*.zip`    |

For a single chromosome, provide the exact chromosome path instead of the wildcard, for example
`hf://datasets/shz9/ukb-ld/EUR/chr_22.zip`.

## Downloading LD matrix archives

If you prefer to pre-download the matrices, the six ancestry groups and their corresponding archive links
are listed below:

| Code  | Ancestry group      | Sample size |                                                                         Download                                                                         |
|:-----:|:--------------------|:-----------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------:|
| `EUR` | European            |   362446    | [GitHub](https://github.com/shz9/viprs/releases/download/v0.1.2/EUR.tar.gz) or [Zenodo](https://zenodo.org/records/14614207/files/EUR.tar.gz?download=1) |
| `CSA` | Central/South Asian |    8284     | [GitHub](https://github.com/shz9/viprs/releases/download/v0.1.2/CSA.tar.gz) or [Zenodo](https://zenodo.org/records/14614207/files/CSA.tar.gz?download=1) |
| `AFR` | African             |    6255     | [GitHub](https://github.com/shz9/viprs/releases/download/v0.1.2/AFR.tar.gz) or [Zenodo](https://zenodo.org/records/14614207/files/AFR.tar.gz?download=1) |
| `EAS` | East Asian          |    2700     | [GitHub](https://github.com/shz9/viprs/releases/download/v0.1.2/EAS.tar.gz) or [Zenodo](https://zenodo.org/records/14614207/files/EAS.tar.gz?download=1) |
| `MID` | Middle Eastern      |    1567     | [GitHub](https://github.com/shz9/viprs/releases/download/v0.1.2/MID.tar.gz) or [Zenodo](https://zenodo.org/records/14614207/files/MID.tar.gz?download=1) |
| `AMR` | Admixed American    |     987     | [GitHub](https://github.com/shz9/viprs/releases/download/v0.1.2/AMR.tar.gz) or [Zenodo](https://zenodo.org/records/14614207/files/AMR.tar.gz?download=1) |


To access and use these matrices for downstream tasks, consult the codebase of [`magenpy`](https://shz9.github.io/magenpy/), our 
sister python package that implements specialized data structures for computing and processing large-scale LD matrices.

## Bash Script for downloading/extracting LD matrices

Here is a bash script that can be used to download and extract the LD matrices for all 6 populations. The script uses
the `GitHub` links provided above. Feel free to modify the script to suit your needs.

```bash
#!/bin/bash
output_dir="LD_matrices"
populations=("EUR" "CSA" "AFR" "EAS" "MID" "AMR")
extract=true

mkdir -p $output_dir

for pop in "${populations[@]}"
do
    echo "Downloading LD matrix for $pop"
    wget -O $output_dir/$pop.tar.gz "https://github.com/shz9/viprs/releases/download/v0.1.2/$pop.tar.gz"
    if [ "$extract" = true ]; then
        mkdir -p $output_dir/$pop
        tar -xf $output_dir/$pop.tar.gz -C $output_dir/$pop
    fi
done
```

Linkage-Disequilibrium (LD) matrices record pairwise correlations between genetic variants and are required as
input to the `VIPRS` model. We provide precomputed matrices from several cohorts and will add more reference
panels over time.

## Available reference panels

| Cohort | Populations | Genome build | Download |
|:-------|:-----------:|:------------:|:---------|
| [UK Biobank](#uk-biobank) | 6 | GRCh37/hg19 | [Zenodo](https://zenodo.org/records/14614207) or [Hugging Face](https://huggingface.co/datasets/shz9/ukb-ld/tree/main) |
| [HGDP + 1000 Genomes](#hgdp-1000-genomes) | 6 | **GRCh38/hg38** | [Zenodo](https://zenodo.org/records/24147521) or [Hugging Face](https://huggingface.co/datasets/shz9/hgdp-1kg-ld/tree/main) |

!!! warning "Match the genome build"
    The HGDP + 1000 Genomes matrices use **GRCh38/hg38 genomic positions**. Make sure that your GWAS summary
    statistics also use hg38 positions before fitting a model with these matrices. If necessary, lift over your
    summary statistics to hg38 first.

## Streaming matrices from Hugging Face 🤗

VIPRS can read the chromosome-level zip files directly from Hugging Face using `hf://` paths, without
pre-downloading and extracting the archive files. Install `huggingface_hub` in the environment where you run
VIPRS:

```bash
python -m pip install huggingface_hub
```

Then pass a cohort- and ancestry-specific wildcard path to the `-l` / `--ld-panel` argument. For example:

```bash
viprs_fit \
  --ld-panel "hf://datasets/shz9/hgdp-1kg-ld/EUR/chr_*.zip" \
  --sumstats path/to/hg38_sumstats.txt \
  --output-dir output/viprs_fit
```

For a single chromosome, replace the wildcard with the exact chromosome path, for example
`hf://datasets/shz9/hgdp-1kg-ld/EUR/chr_22.zip`.

## UK Biobank

These matrices were computed from unrelated UK Biobank participants in six continental ancestry groups. They
use the `block` LD estimator, which records correlations only between variants in the same LD block. The blocks
are defined by [`LDetect`](https://bitbucket.org/nygcresearch/ldetect-data/src/master/). The matrices were
computed with [`magenpy`](https://shz9.github.io/magenpy/) and quantized to the `int8` data type for enhanced
compressibility.

| Code | Ancestry group | Sample size | Hugging Face LD path | Archive download |
|:----:|:---------------|------------:|:---------------------|:-----------------|
| `EUR` | European | 362446 | `hf://datasets/shz9/ukb-ld/EUR/chr_*.zip` | [GitHub](https://github.com/shz9/viprs/releases/download/v0.1.2/EUR.tar.gz) or [Zenodo](https://zenodo.org/records/14614207/files/EUR.tar.gz?download=1) |
| `CSA` | Central/South Asian | 8284 | `hf://datasets/shz9/ukb-ld/CSA/chr_*.zip` | [GitHub](https://github.com/shz9/viprs/releases/download/v0.1.2/CSA.tar.gz) or [Zenodo](https://zenodo.org/records/14614207/files/CSA.tar.gz?download=1) |
| `AFR` | African | 6255 | `hf://datasets/shz9/ukb-ld/AFR/chr_*.zip` | [GitHub](https://github.com/shz9/viprs/releases/download/v0.1.2/AFR.tar.gz) or [Zenodo](https://zenodo.org/records/14614207/files/AFR.tar.gz?download=1) |
| `EAS` | East Asian | 2700 | `hf://datasets/shz9/ukb-ld/EAS/chr_*.zip` | [GitHub](https://github.com/shz9/viprs/releases/download/v0.1.2/EAS.tar.gz) or [Zenodo](https://zenodo.org/records/14614207/files/EAS.tar.gz?download=1) |
| `MID` | Middle Eastern | 1567 | `hf://datasets/shz9/ukb-ld/MID/chr_*.zip` | [GitHub](https://github.com/shz9/viprs/releases/download/v0.1.2/MID.tar.gz) or [Zenodo](https://zenodo.org/records/14614207/files/MID.tar.gz?download=1) |
| `AMR` | Admixed American | 987 | `hf://datasets/shz9/ukb-ld/AMR/chr_*.zip` | [GitHub](https://github.com/shz9/viprs/releases/download/v0.1.2/AMR.tar.gz) or [Zenodo](https://zenodo.org/records/14614207/files/AMR.tar.gz?download=1) |

For European samples, [Zenodo](https://zenodo.org/records/14614207) also provides whole-genome matrices that
record pairwise correlations for up to 18 million variants.

For details on quality control and data preparation, see:

> Zabad, S., Haryan, C. A., Gravel, S., Misra, S., Li, Y. (2025). **Toward whole-genome inference of polygenic scores with fast and memory-efficient algorithms.**
> *The American Journal of Human Genetics*, 112(7), 1528-1546. https://doi.org/10.1016/j.ajhg.2025.05.002

## HGDP + 1000 Genomes

These matrices were computed from unrelated participants in the joint
[Human Genome Diversity Project (HGDP) + 1000 Genomes Project dataset](https://gnomad.broadinstitute.org/downloads#v3-hgdp-1kg).
They contain pairwise correlations for up to 2,408,970 variants with minor allele frequency greater than 1%.
As with the UK Biobank matrices, they were computed with `magenpy`, use the `block` LD estimator, and were
quantized to `int8`. The ancestry groups follow the Pan-UK Biobank definitions.

!!! warning "These matrices use hg38"
    Variant positions and LD blocks in this panel use **GRCh38/hg38**, including the
    [GRCh38 LDetect block definitions](https://github.com/jmacdon/LDblocks_GRCh38). Do not use these matrices
    with summary statistics in GRCh37/hg19 coordinates without first lifting the summary statistics to hg38.

| Code | Ancestry group | Sample size | Hugging Face LD path | Archive download |
|:----:|:---------------|------------:|:---------------------|:-----------------|
| `EUR` | European | 657 | `hf://datasets/shz9/hgdp-1kg-ld/EUR/chr_*.zip` | [Zenodo](https://zenodo.org/records/24147521/files/EUR.tar.gz?download=1) |
| `CSA` | Central/South Asian | 647 | `hf://datasets/shz9/hgdp-1kg-ld/CSA/chr_*.zip` | [Zenodo](https://zenodo.org/records/24147521/files/CSA.tar.gz?download=1) |
| `AFR` | African | 654 | `hf://datasets/shz9/hgdp-1kg-ld/AFR/chr_*.zip` | [Zenodo](https://zenodo.org/records/24147521/files/AFR.tar.gz?download=1) |
| `EAS` | East Asian | 713 | `hf://datasets/shz9/hgdp-1kg-ld/EAS/chr_*.zip` | [Zenodo](https://zenodo.org/records/24147521/files/EAS.tar.gz?download=1) |
| `MID` | Middle Eastern | 135 | `hf://datasets/shz9/hgdp-1kg-ld/MID/chr_*.zip` | [Zenodo](https://zenodo.org/records/24147521/files/MID.tar.gz?download=1) |
| `AMR` | Admixed American | 366 | `hf://datasets/shz9/hgdp-1kg-ld/AMR/chr_*.zip` | [Zenodo](https://zenodo.org/records/24147521/files/AMR.tar.gz?download=1) |

The full collection is available from [Zenodo](https://zenodo.org/records/24147521), and the chromosome-level
archives are available from [Hugging Face](https://huggingface.co/datasets/shz9/hgdp-1kg-ld/tree/main). For
details on how this panel was generated, see the
[`1KG_HGDP_ld` codebase](https://github.com/shz9/1KG_HGDP_ld).

## Exploring and processing LD matrices

### Interactive exploration with the LD Browser

Use the [LD Browser](https://shz9.github.io/ld_browser/) to explore these matrices interactively without
installing software. You can select a hosted reference matrix or open your own chromosome-level matrix from:

- a local Zarr directory or zip archive; or
- cloud storage using an `hf://`, `s3://`, `gs://`, `gcs://`, or CORS-enabled HTTPS URI.

For example, enter the following URI in the browser to open chromosome 22 from the European HGDP + 1000
Genomes panel:

```text
hf://datasets/shz9/hgdp-1kg-ld/EUR/chr_22.zip
```

The browser provides regional LD heatmaps, pairwise-correlation queries, variant metadata, LD pruning, tagging
variant searches, and comparisons between populations. Results and heatmaps can be exported as CSV and PNG
files. Local files are read directly in your browser and are not uploaded.

!!! note
    The browser opens one chromosome-level matrix at a time. As elsewhere on this page, check the genome build
    before interpreting positions or comparing a matrix with other genomic data.

### Programmatic analysis with `magenpy`

For programmatic access, computation, and downstream processing, consult the documentation for
[`magenpy`](https://shz9.github.io/magenpy/). This sister Python package implements the specialized data
structures used to store and process these large-scale LD matrices.

## Downloading matrices with Python

Use `download_ld_matrix` to save archives locally. Specify an ancestry code;
Hugging Face and the UK Biobank panel are the defaults:

```python
from viprs.utils.data_utils import download_ld_matrix

paths = download_ld_matrix("LD_matrices", ancestry="EUR", chromosome=[21, 22])
# LD_matrices/ukb/EUR/chr_21.zip and chr_22.zip

# HGDP + 1000 Genomes uses GRCh38/hg38 positions:
paths = download_ld_matrix(
    "LD_matrices", ancestry="AFR", panel="hgdp-1kg", chromosome=22
)

# Zenodo supplies a complete ancestry archive, without chromosome selection:
paths = download_ld_matrix(
    "LD_matrices", ancestry="EUR", panel="hgdp-1kg", source="zenodo"
)
# LD_matrices/hgdp-1kg/EUR/EUR.tar.gz
```

The helper returns a list of local `pathlib.Path` objects. With Hugging Face,
omitting `chromosome` downloads all 22 autosomes. Downloaded ZIP files can be
passed directly to `magenpy`. Zenodo archives must be extracted before use, as
shown below. When `huggingface_hub` is installed, the helper automatically uses
its `hf_hub_download` interface. Otherwise, it streams downloads over HTTP.
The Hub interface stages files locally before moving them into place, without
keeping a second archive in the global Hub cache. Hub transfer failures are
reported rather than silently switching to HTTP. No Hugging Face package is required.
Existing files are reused unless `overwrite=True` is supplied. Interrupted
transfers do not replace completed files; retrying reuses completed chromosomes.

## Downloading matrices from Zenodo

To download a matrix archive from Zenodo, select the record for the desired cohort and set `population` to
one of the ancestry codes listed in the tables above:

- UK Biobank: `record_id=14614207`
- HGDP + 1000 Genomes: `record_id=24147521`

The following example downloads and extracts the European matrix from the HGDP + 1000 Genomes panel:

```bash
#!/bin/bash
record_id=24147521
population="EUR"
output_dir="LD_matrices"

mkdir -p "$output_dir/$population"

wget -O "$output_dir/$population.tar.gz" \
    "https://zenodo.org/records/$record_id/files/$population.tar.gz?download=1"

tar -xf "$output_dir/$population.tar.gz" -C "$output_dir/$population"
```

Change `record_id` to select a different cohort and `population` to select another ancestry group. To keep
the archive without extracting it, omit the final `tar` command.

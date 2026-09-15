# VIPRS continuous benchmark results

Accuracy is the mean test-set `Pseudo_R2` across five HEIGHT folds. Runtime is
the mean fitting `Total_WallClockTime` across those folds, in seconds. Benchmarks
run on GitHub-hosted `ubuntu-24.04` runners with Python 3.12 and one inference thread.
Pull requests publish their proposed table in the workflow summary and as an artifact;
pushes to the default branch also update this history.

| Commit ID | Date | Model | Configurations | Phenotype | Accuracy | Runtime | Sumstats | LD |
|---|---|---|---|---|---:|---:|---|---|
| [`2c3e868`](https://github.com/shz9/viprs/commit/2c3e86804e393e6046cc5acf2752ad31b14cc1a8) | 2026-09-15 | VIPRS |  | height | 0.336213 | 21.48 s | [HEIGHT](https://zenodo.org/records/14612130/files/HEIGHT.tar.gz) | [EUR](https://github.com/shz9/viprs/releases/download/v0.1.2/EUR.tar.gz) |
| [`2c3e868`](https://github.com/shz9/viprs/commit/2c3e86804e393e6046cc5acf2752ad31b14cc1a8) | 2026-09-15 | VIPRSMix(K=4) |  | height | 0.343416 | 42.39 s | [HEIGHT](https://zenodo.org/records/14612130/files/HEIGHT.tar.gz) | [EUR](https://github.com/shz9/viprs/releases/download/v0.1.2/EUR.tar.gz) |

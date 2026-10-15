# VIPRS continuous benchmark results

Accuracy is the mean test-set `Pseudo_R2` across five HEIGHT folds. Runtime is
the mean fitting `Total_WallClockTime` across those folds, in seconds. Benchmarks
run on GitHub-hosted `ubuntu-24.04` runners with Python 3.12 and one inference thread.
Pull requests publish their proposed table in the workflow summary and as an artifact;
pushes to the default branch also update this history.

| Commit ID | Date | Model | Configurations | Phenotype | Accuracy | Runtime | Sumstats | LD |
|---|---|---|---|---|---:|---:|---|---|
| [`1e98bc7`](https://github.com/shz9/viprs/commit/1e98bc77e816d5332bc01bbeb05bfa130871c022) | 2026-10-15 | VIPRS |  | height | 0.336218 | 15.01 s | [HEIGHT](https://zenodo.org/records/14612130/files/HEIGHT.tar.gz) | [EUR](https://github.com/shz9/viprs/releases/download/v0.1.2/EUR.tar.gz) |
| [`1e98bc7`](https://github.com/shz9/viprs/commit/1e98bc77e816d5332bc01bbeb05bfa130871c022) | 2026-10-15 | VIPRSMix(K=4) |  | height | 0.343416 | 60.71 s | [HEIGHT](https://zenodo.org/records/14612130/files/HEIGHT.tar.gz) | [EUR](https://github.com/shz9/viprs/releases/download/v0.1.2/EUR.tar.gz) |
| [`bbe3411`](https://github.com/shz9/viprs/commit/bbe34115d85b963136203306e5a2a47cc47a4191) | 2026-10-14 | VIPRS |  | height | 0.336213 | 21.00 s | [HEIGHT](https://zenodo.org/records/14612130/files/HEIGHT.tar.gz) | [EUR](https://github.com/shz9/viprs/releases/download/v0.1.2/EUR.tar.gz) |
| [`bbe3411`](https://github.com/shz9/viprs/commit/bbe34115d85b963136203306e5a2a47cc47a4191) | 2026-10-14 | VIPRSMix(K=4) |  | height | 0.343416 | 59.70 s | [HEIGHT](https://zenodo.org/records/14612130/files/HEIGHT.tar.gz) | [EUR](https://github.com/shz9/viprs/releases/download/v0.1.2/EUR.tar.gz) |
| [`2e3afff`](https://github.com/shz9/viprs/commit/2e3afffed04655ef920894a404121e3e7f3248f6) | 2026-10-13 | VIPRS |  | height | 0.336213 | 20.03 s | [HEIGHT](https://zenodo.org/records/14612130/files/HEIGHT.tar.gz) | [EUR](https://github.com/shz9/viprs/releases/download/v0.1.2/EUR.tar.gz) |
| [`2e3afff`](https://github.com/shz9/viprs/commit/2e3afffed04655ef920894a404121e3e7f3248f6) | 2026-10-13 | VIPRSMix(K=4) |  | height | 0.343416 | 39.62 s | [HEIGHT](https://zenodo.org/records/14612130/files/HEIGHT.tar.gz) | [EUR](https://github.com/shz9/viprs/releases/download/v0.1.2/EUR.tar.gz) |
| [`45ec390`](https://github.com/shz9/viprs/commit/45ec390da2b10f716b1308ee9070c54db9c8c18a) | 2026-10-13 | VIPRS |  | height | 0.336218 | 12.99 s | [HEIGHT](https://zenodo.org/records/14612130/files/HEIGHT.tar.gz) | [EUR](https://github.com/shz9/viprs/releases/download/v0.1.2/EUR.tar.gz) |
| [`45ec390`](https://github.com/shz9/viprs/commit/45ec390da2b10f716b1308ee9070c54db9c8c18a) | 2026-10-13 | VIPRSMix(K=4) |  | height | 0.343416 | 39.99 s | [HEIGHT](https://zenodo.org/records/14612130/files/HEIGHT.tar.gz) | [EUR](https://github.com/shz9/viprs/releases/download/v0.1.2/EUR.tar.gz) |
| [`f189850`](https://github.com/shz9/viprs/commit/f189850c4020902f156f5820726dc1c61a278a48) | 2026-10-13 | VIPRS |  | height | 0.336213 | 18.61 s | [HEIGHT](https://zenodo.org/records/14612130/files/HEIGHT.tar.gz) | [EUR](https://github.com/shz9/viprs/releases/download/v0.1.2/EUR.tar.gz) |
| [`f189850`](https://github.com/shz9/viprs/commit/f189850c4020902f156f5820726dc1c61a278a48) | 2026-10-13 | VIPRSMix(K=4) |  | height | 0.343416 | 43.21 s | [HEIGHT](https://zenodo.org/records/14612130/files/HEIGHT.tar.gz) | [EUR](https://github.com/shz9/viprs/releases/download/v0.1.2/EUR.tar.gz) |
| [`a90d9fe`](https://github.com/shz9/viprs/commit/a90d9fe22e5dcaf38c623b221a8368480139fb1e) | 2026-10-07 | VIPRS |  | height | 0.336213 | 20.55 s | [HEIGHT](https://zenodo.org/records/14612130/files/HEIGHT.tar.gz) | [EUR](https://github.com/shz9/viprs/releases/download/v0.1.2/EUR.tar.gz) |
| [`a90d9fe`](https://github.com/shz9/viprs/commit/a90d9fe22e5dcaf38c623b221a8368480139fb1e) | 2026-10-07 | VIPRSMix(K=4) |  | height | 0.343416 | 68.78 s | [HEIGHT](https://zenodo.org/records/14612130/files/HEIGHT.tar.gz) | [EUR](https://github.com/shz9/viprs/releases/download/v0.1.2/EUR.tar.gz) |
| [`2c3e868`](https://github.com/shz9/viprs/commit/2c3e86804e393e6046cc5acf2752ad31b14cc1a8) | 2026-09-15 | VIPRS |  | height | 0.336213 | 21.48 s | [HEIGHT](https://zenodo.org/records/14612130/files/HEIGHT.tar.gz) | [EUR](https://github.com/shz9/viprs/releases/download/v0.1.2/EUR.tar.gz) |
| [`2c3e868`](https://github.com/shz9/viprs/commit/2c3e86804e393e6046cc5acf2752ad31b14cc1a8) | 2026-09-15 | VIPRSMix(K=4) |  | height | 0.343416 | 42.39 s | [HEIGHT](https://zenodo.org/records/14612130/files/HEIGHT.tar.gz) | [EUR](https://github.com/shz9/viprs/releases/download/v0.1.2/EUR.tar.gz) |

# VIPRS continuous benchmark results

Accuracy is the mean test-set `Pseudo_R2` across five HEIGHT folds. Runtime is
the mean fitting `Total_WallClockTime` across those folds, in seconds. Benchmarks
run on GitHub-hosted `ubuntu-24.04` runners with Python 3.12 and one inference thread.
Pull requests publish their proposed table in the workflow summary and as an artifact;
pushes to the default branch also update this history.

| Commit ID | Date | Model | Configurations | Phenotype | Accuracy | Runtime | Sumstats | LD |
|---|---|---|---|---|---:|---:|---|---|

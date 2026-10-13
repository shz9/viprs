"""Small two-chromosome fixtures for integration tests; never application defaults."""
import json
from pathlib import Path
import zipfile

import numpy as np
from scipy.sparse import csr_matrix
from magenpy import LDMatrix


def make_multichrom_inputs(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    summaries, archives, stores = [], [], []
    for chromosome in (21, 22):
        size = 48
        matrix = 0.5 ** np.abs(np.arange(size)[:, None] - np.arange(size)[None, :])
        store = directory / f"chr{chromosome}.zarr"
        ld = LDMatrix.from_csr(csr_matrix(matrix), str(store), dtype="float64", overwrite=True)
        snps = [f"test_chr{chromosome}_{i}" for i in range(size)]
        for key, value in {"snps": snps, "a1": ["A"] * size, "a2": ["G"] * size,
                           "bp": np.arange(1, size + 1), "maf": np.full(size, 0.3),
                           "cm": np.arange(size) * .01, "ldscore": np.sum(matrix**2, axis=0)}.items():
            ld.set_metadata(key, np.asarray(value))
        ld.set_store_attr("Chromosome", chromosome)
        ld.set_store_attr("Sample size", 1000)
        effects = np.zeros(size)
        effects[[5, 20, 38]] = [.06, -.05, .04]
        z = 100 * (matrix @ effects) + .1 * np.sin(np.arange(size))
        summary = directory / f"chr{chromosome}.tsv"
        summary.write_text("CHR\tSNP\tPOS\tA1\tA2\tZ\tN\n" + "".join(
            f"{chromosome}\t{snp}\t{i+1}\tA\tG\t{z[i]}\t10000\n" for i, snp in enumerate(snps)))
        archive = directory / f"chr{chromosome}.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            for path in store.rglob("*"):
                if path.is_file():
                    handle.write(path, path.relative_to(directory))
        summaries.append(summary)
        archives.append(archive)
        stores.append(store)
    combined = directory / "combined.tsv"
    combined.write_text(summaries[0].read_text() + "\n".join(summaries[1].read_text().splitlines()[1:]) + "\n")
    combined_ld = directory / "combined.zip"
    with zipfile.ZipFile(combined_ld, "w") as handle:
        for store in stores:
            for path in store.rglob("*"):
                if path.is_file():
                    handle.write(path, path.relative_to(directory))
    return summaries, archives, combined, combined_ld


if __name__ == "__main__":
    make_multichrom_inputs(Path(__file__).resolve().parents[1] / "build/wasm/test-data/multi")

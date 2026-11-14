"""Exercise VIPRS entry points with chromosome labels normalized by magenpy."""
import runpy
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from magenpy import GWADataLoader
from viprs.model import BayesPRSModel
from viprs.eval.pseudo_metrics import _match_variant_stats
from tests.webapp.webapp_fixtures import make_multichrom_inputs


@pytest.fixture
def inputs(tmp_path):
    summaries, _, combined, _ = make_multichrom_inputs(tmp_path / "inputs")
    roots = [str(path.parent / f"chr{chromosome}.zarr") for path, chromosome in zip(summaries, (21, 22))]
    return roots, combined


@pytest.mark.parametrize("label", ["chr22", "chr:22", "chr_22", "022", "22.0"])
def test_cli_loads_chromosome_aliases(inputs, tmp_path, label):
    roots, summary = inputs
    table = pd.read_csv(summary, sep="\t")
    table["CHR"] = table["CHR"].astype(str).replace("22", label)
    source = tmp_path / "aliases.tsv"
    table.to_csv(source, sep="\t", index=False)
    cli = runpy.run_path(str(Path(__file__).parents[2] / "bin/viprs_fit"))
    args = Namespace(ld_dir=roots, temp_dir=str(tmp_path / "temp"), threads=1,
                     sumstats_path=str(source), sumstats_format="magenpy",
                     exclude_lrld=False, gwas_sample_size=None, genomewide=True, hyp_search="EM")
    loader = list(cli["init_data"](args))[0]["train"]
    try:
        assert set(loader.chromosomes) == {21, 22}
        assert loader.n_snps == 96
    finally:
        loader.cleanup()


def test_effect_tables_normalize_aliases_before_matching(inputs, tmp_path):
    roots, summary = inputs
    loader = GWADataLoader(ld_store_files=roots, sumstats_files=str(summary),
                           sumstats_format="magenpy", temp_dir=str(tmp_path / "temp"))
    try:
        model = BayesPRSModel(loader)
        parameters = loader.to_snp_table()
        parameters["BETA"] = np.linspace(-.01, .01, len(parameters))
        expected = model.harmonize_data(parameter_table=parameters)[1]
        expected_stats = _match_variant_stats(loader, parameters)
        aliases = parameters.copy()
        aliases["CHR"] = ["chr:21" if chromosome == 21 else "chr_22" for chromosome in aliases["CHR"]]
        # Include an unused category to guard chromosome discovery/grouping.
        aliases["CHR"] = pd.Categorical(aliases["CHR"], categories=["chr:21", "chr_22", "chrX"])
        actual = model.harmonize_data(parameter_table=aliases)[1]
        assert set(actual) == {21, 22}
        for chromosome in actual:
            np.testing.assert_allclose(actual[chromosome], expected[chromosome])
        for actual_stat, expected_stat in zip(_match_variant_stats(loader, aliases), expected_stats):
            np.testing.assert_allclose(actual_stat, expected_stat)
        assert str(aliases["CHR"].iloc[0]) == "chr:21"
    finally:
        loader.cleanup()

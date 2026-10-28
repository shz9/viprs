import asyncio

import pytest

from viprs.utils.data_utils import (
    gwas_catalog_directory,
    resolve_gwas_catalog,
    resolve_gwas_catalog_async,
)


def catalog_documents():
    directory = gwas_catalog_directory("GCST005195")
    return directory, {
        directory: (
            '<a href="harmonised/">harmonised</a>'
            '<a href="raw.tsv.gz">raw</a>'
        ),
        directory + "harmonised/": (
            '<a href="study.h.tsv.gz">harmonised</a>'
            '<a href="study.h.tsv.gz-meta.yaml">metadata</a>'
            '<a href="study-build37.f.tsv.gz">formatted build 37</a>'
        ),
        directory + "harmonised/study.h.tsv.gz-meta.yaml": (
            "genome_assembly: GRCh38\nis_harmonised: true\n"
        ),
    }


def test_catalog_accessions_resolve_to_their_thousand_study_directory():
    assert gwas_catalog_directory("gcst005195") == (
        "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/"
        "GCST005001-GCST006000/GCST005195/"
    )
    assert gwas_catalog_directory("GCST90000001").endswith(
        "GCST90000001-GCST90001000/GCST90000001/"
    )
    with pytest.raises(ValueError, match="such as GCST"):
        gwas_catalog_directory("005195")


def test_catalog_discovery_prefers_grch37_harmonised_files():
    directory, documents = catalog_documents()
    assert resolve_gwas_catalog("GCST005195", documents.__getitem__) == (
        directory + "harmonised/study-build37.f.tsv.gz"
    )


def test_catalog_discovery_uses_metadata_and_has_an_async_transport():
    directory = gwas_catalog_directory("GCST005195")
    documents = {
        directory: (
            '<a href="study.tsv.gz">data</a>'
            '<a href="study.tsv.gz-meta.yaml">metadata</a>'
        ),
        directory + "study.tsv.gz-meta.yaml": "genome_assembly: GRCh37\n",
    }

    async def read_text(url):
        return documents[url]

    assert asyncio.run(resolve_gwas_catalog_async("GCST005195", read_text)) == (
        directory + "study.tsv.gz"
    )


def test_catalog_discovery_rejects_unusable_studies():
    directory = gwas_catalog_directory("GCST005195")
    with pytest.raises(ValueError, match="no GRCh37/hg19"):
        resolve_gwas_catalog(
            "GCST005195",
            lambda url: '<a href="study-buildGRCh38.tsv.gz">data</a>'
            if url == directory
            else "",
        )

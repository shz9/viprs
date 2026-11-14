"""Test the browser's asynchronous transport around magenpy discovery."""
import asyncio
import runpy
from pathlib import Path

import pytest
from magenpy.utils.gwas_catalog import gwas_catalog_directory

resolve_browser_gwas_catalog = runpy.run_path(
    str(Path(__file__).parents[2] / "docs/webapp/inference.py")
)["resolve_browser_gwas_catalog"]


@pytest.mark.parametrize("metadata", ["genome_assembly: GRCh38", ConnectionError("unavailable")])
def test_discovery_uses_magenpy_priorities_and_fetches_each_document_once(metadata):
    directory = gwas_catalog_directory("GCST005195")
    documents = {
        directory: '<a href="harmonised/">harmonised</a><a href="raw-build37.tsv.gz">raw</a>',
        directory + "harmonised/": (
            '<a href="study.h.tsv.gz">harmonised</a>'
            '<a href="study.h.tsv.gz-meta.yaml">metadata</a>'
            '<a href="study-build37.f.tsv.gz">formatted build 37</a>'
        ),
        directory + "harmonised/study.h.tsv.gz-meta.yaml": metadata,
    }
    fetched = []

    async def read_text(url):
        fetched.append(url)
        value = documents[url]
        if isinstance(value, Exception):
            raise value
        return value

    assert asyncio.run(resolve_browser_gwas_catalog("GCST005195", read_text)) == (
        directory + "harmonised/study-build37.f.tsv.gz"
    )
    assert fetched == list(documents)


def test_discovery_accepts_grch37_metadata():
    directory = gwas_catalog_directory("GCST005195")
    documents = {
        directory: '<a href="study.tsv.gz">data</a><a href="study.tsv.gz-meta.yaml">metadata</a>',
        directory + "study.tsv.gz-meta.yaml": "genome_assembly: GRCh37\n",
    }

    async def read_text(url):
        return documents[url]

    assert asyncio.run(resolve_browser_gwas_catalog("GCST005195", read_text)) == directory + "study.tsv.gz"


def test_discovery_rejects_unusable_studies():
    async def read_text(url):
        return '<a href="study-buildGRCh38.tsv.gz">data</a>'

    with pytest.raises(ValueError, match="no GRCh37/hg19"):
        asyncio.run(resolve_browser_gwas_catalog("GCST005195", read_text))


def test_discovery_preserves_listing_failures():
    async def read_text(url):
        raise ConnectionError("listing unavailable")

    with pytest.raises(ConnectionError, match="listing unavailable"):
        asyncio.run(resolve_browser_gwas_catalog("GCST005195", read_text))


def test_discovery_validates_accessions_before_fetching():
    async def read_text(url):
        pytest.fail("Invalid accessions should not trigger browser requests")

    with pytest.raises(ValueError, match="such as GCST"):
        asyncio.run(resolve_browser_gwas_catalog("invalid", read_text))


@pytest.mark.parametrize("build,filename", [("hg19", "study-build37.f.tsv.gz"), ("hg38", "study-build38.f.tsv.gz")])
def test_discovery_targets_the_selected_ld_build(build, filename):
    directory = gwas_catalog_directory("GCST005195")
    documents = {
        directory: '<a href="harmonised/">harmonised</a>',
        directory + "harmonised/": '<a href="study-build37.f.tsv.gz">37</a><a href="study-build38.f.tsv.gz">38</a>',
    }
    async def read_text(url):
        return documents[url]
    assert asyncio.run(resolve_browser_gwas_catalog("GCST005195", read_text, build)) == directory + "harmonised/" + filename


def test_grch38_discovery_requires_matching_metadata():
    directory = gwas_catalog_directory("GCST005195")
    documents = {
        directory: '<a href="study.tsv.gz">data</a><a href="study.tsv.gz-meta.yaml">metadata</a>',
        directory + "study.tsv.gz-meta.yaml": 'genome_assembly: "hg38"',
    }
    async def read_text(url):
        return documents[url]
    assert asyncio.run(resolve_browser_gwas_catalog("GCST005195", read_text, "GRCh38")) == directory + "study.tsv.gz"
    documents[directory + "study.tsv.gz-meta.yaml"] = "genome_assembly: GRCh37"
    with pytest.raises(ValueError, match="no GRCh38/hg38"):
        asyncio.run(resolve_browser_gwas_catalog("GCST005195", read_text, "GRCh38"))

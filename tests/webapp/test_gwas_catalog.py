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


@pytest.mark.parametrize("build", ["GRCh37", "GRCh38"])
def test_lookup_reports_each_real_request_and_completion(build):
    directory = gwas_catalog_directory("GCST90079133")
    metadata_url = directory + "harmonised/study.h.tsv.gz-meta.yaml"
    documents = {
        directory: '<a href="harmonised/">harmonised</a>',
        directory + "harmonised/": '<a href="study.h.tsv.gz">data</a><a href="study.h.tsv.gz-meta.yaml">metadata</a>',
        metadata_url: f"genome_assembly: {build}",
    }
    events = []

    async def read_text(url):
        assert events[-1] == ("request", url)
        return documents[url]

    result = asyncio.run(resolve_browser_gwas_catalog(
        "GCST90079133", read_text, build, lambda event, url: events.append((event, url))
    ))
    assert result == directory + "harmonised/study.h.tsv.gz"
    assert events == [item for url in documents for item in [("request", url), ("received", url)]]


def test_failed_metadata_request_reports_failure_before_trying_next_candidate():
    directory = gwas_catalog_directory("GCST90079133")
    metadata_url = directory + "harmonised/study.h.tsv.gz-meta.yaml"
    documents = {
        directory: '<a href="harmonised/">harmonised</a>',
        directory + "harmonised/": '<a href="study.h.tsv.gz">data</a><a href="study.h.tsv.gz-meta.yaml">metadata</a><a href="study-build37.f.tsv.gz">fallback</a>',
    }
    events = []

    async def read_text(url):
        if url == metadata_url:
            raise ConnectionError("metadata unavailable")
        return documents[url]

    assert asyncio.run(resolve_browser_gwas_catalog(
        "GCST90079133", read_text, progress_callback=lambda event, url: events.append((event, url))
    )).endswith("study-build37.f.tsv.gz")
    assert events[-2:] == [("request", metadata_url), ("failed", metadata_url)]


def test_pending_lookup_reports_request_before_waiting_for_server():
    async def check():
        started = asyncio.Event()
        release = asyncio.Event()
        events = []

        async def read_text(url):
            started.set()
            await release.wait()
            return '<a href="study-build37.tsv.gz">data</a>'

        task = asyncio.create_task(resolve_browser_gwas_catalog(
            "GCST90079133", read_text, progress_callback=lambda event, url: events.append((event, url))
        ))
        await started.wait()
        assert [event for event, _ in events] == ["request"]
        release.set()
        await task
        assert [event for event, _ in events] == ["request", "received"]

    asyncio.run(check())


@pytest.mark.parametrize("selected,available", [("GRCh37", "GRCh38"), ("GRCh38", "GRCh37")])
def test_build_mismatch_recommends_a_matching_ld_matrix(selected, available):
    accession = "GCST90833078"
    directory = gwas_catalog_directory(accession)
    filename = accession + ".tsv.gz"
    documents = {
        directory: f'<a href="{filename}">data</a><a href="{filename}-meta.yaml">metadata</a>',
        directory + filename + "-meta.yaml": f"genome_assembly: {available}\n",
    }

    async def read_text(url):
        return documents[url]

    with pytest.raises(ValueError, match=f"Change the LD matrix to a compatible {available} panel"):
        asyncio.run(resolve_browser_gwas_catalog(accession, read_text, selected))
    assert asyncio.run(resolve_browser_gwas_catalog(accession, read_text, available)) == directory + filename


@pytest.mark.parametrize("build", ["GRCh37", "GRCh38"])
@pytest.mark.parametrize("stall_metadata", [False, True])
def test_stalled_requests_are_cancelled_and_reported(build, stall_metadata):
    async def check():
        directory = gwas_catalog_directory("GCST90833078")
        metadata_url = directory + "study.tsv.gz-meta.yaml"
        cancelled = []
        events = []

        async def read_text(url):
            if stall_metadata and url == directory:
                return '<a href="study.tsv.gz">data</a><a href="study.tsv.gz-meta.yaml">metadata</a>'
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.append(url)

        with pytest.raises((TimeoutError, ConnectionError), match="timed out after") as error:
            await asyncio.wait_for(resolve_browser_gwas_catalog(
                "GCST90833078", read_text, build,
                lambda event, url: events.append((event, url)), request_timeout=0.01,
            ), timeout=1)
        stalled_url = metadata_url if stall_metadata else directory
        assert cancelled == [stalled_url]
        assert events[-1] == ("failed", stalled_url)
        if stall_metadata:
            assert "Could not verify" in str(error.value)
            assert "no GRCh" not in str(error.value)

    asyncio.run(check())


@pytest.mark.parametrize("build,number", [("GRCh37", 37), ("GRCh38", 38)])
def test_metadata_timeout_can_fall_back_to_a_build_labelled_file(build, number):
    directory = gwas_catalog_directory("GCST005195")
    fallback = f"study-build{number}.f.tsv.gz"
    documents = {
        directory: '<a href="harmonised/">harmonised</a>',
        directory + "harmonised/": f'<a href="study.h.tsv.gz">data</a><a href="study.h.tsv.gz-meta.yaml">metadata</a><a href="{fallback}">fallback</a>',
    }

    async def read_text(url):
        if url.endswith("-meta.yaml"):
            await asyncio.Event().wait()
        return documents[url]

    assert asyncio.run(resolve_browser_gwas_catalog(
        "GCST005195", read_text, build, request_timeout=0.01,
    )) == directory + "harmonised/" + fallback


@pytest.mark.parametrize("stall_body", [False, True])
def test_browser_transport_aborts_stalled_fetch_and_response_body(monkeypatch, stall_body):
    import sys
    import types

    controllers = []

    class Controller:
        signal = object()

        @classmethod
        def new(cls):
            instance = cls()
            instance.aborted = False
            controllers.append(instance)
            return instance

        def abort(self):
            self.aborted = True

    class Response:
        status = 200

        async def string(self):
            await asyncio.Event().wait()

    async def pyfetch(url, **options):
        assert options["signal"] is controllers[-1].signal
        assert options["credentials"] == "omit"
        if not stall_body:
            await asyncio.Event().wait()
        return Response()

    monkeypatch.setitem(sys.modules, "js", types.SimpleNamespace(AbortController=Controller))
    monkeypatch.setitem(sys.modules, "pyodide.http", types.SimpleNamespace(pyfetch=pyfetch))
    with pytest.raises(TimeoutError, match="timed out after"):
        asyncio.run(resolve_browser_gwas_catalog("GCST005195", request_timeout=0.01))
    assert len(controllers) == 1
    assert controllers[0].aborted

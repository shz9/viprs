"""Hosted LD download contracts, using small in-memory HTTP responses."""
import io
import sys
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError

import numpy as np
import pytest

from viprs.utils import data_utils


@pytest.fixture(autouse=True)
def no_huggingface_hub(monkeypatch):
    # Exercise the HTTP fallback independently of the developer environment.
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)


class Response(io.BytesIO):
    def __init__(self, content=b"archive bytes", length=None):
        super().__init__(content)
        self.headers = {"Content-Length": str(len(content) if length is None else length)}


@pytest.mark.parametrize("panel,repository", [("ukb", "ukb-ld"), ("hgdp-1kg", "hgdp-1kg-ld")])
def test_huggingface_downloads_selected_chromosomes(tmp_path, monkeypatch, panel, repository):
    calls = []

    def fetch(url, timeout):
        calls.append(url)
        assert timeout == 60
        return Response()

    monkeypatch.setattr(data_utils, "urlopen", fetch)
    paths = data_utils.download_ld_matrix(tmp_path, ancestry="eur", panel=panel, chromosome=[np.int64(22), 21, 22])
    assert paths == [tmp_path / panel / "EUR" / f"chr_{c}.zip" for c in [22, 21]]
    assert calls == [f"https://huggingface.co/datasets/shz9/{repository}/resolve/main/EUR/chr_{c}.zip" for c in [22, 21]]
    assert all(path.read_bytes() == b"archive bytes" for path in paths)
    assert not list(tmp_path.rglob("*.part"))


def test_default_all_autosomes_and_single_chromosome(tmp_path, monkeypatch):
    monkeypatch.setattr(data_utils, "urlopen", lambda *args, **kwargs: Response())
    paths = data_utils.download_ld_matrix(tmp_path, ancestry="AFR")
    assert [path.name for path in paths] == [f"chr_{c}.zip" for c in range(1, 23)]
    assert data_utils.download_ld_matrix(tmp_path, ancestry="AFR", chromosome=22) == [paths[-1]]


@pytest.mark.parametrize("panel,record", [("ukb", "14614207"), ("hgdp-1kg", "24147521")])
def test_zenodo_downloads_whole_ancestry_archive(tmp_path, monkeypatch, panel, record):
    def fetch(url, **kwargs):
        assert url == f"https://zenodo.org/records/{record}/files/CSA.tar.gz?download=1"
        return Response()

    monkeypatch.setattr(data_utils, "urlopen", fetch)
    paths = data_utils.download_ld_matrix(tmp_path, ancestry="CSA", panel=panel, source="zenodo")
    assert paths == [tmp_path / panel / "CSA" / "CSA.tar.gz"]
    assert paths[0].read_bytes() == b"archive bytes"


@pytest.mark.parametrize("options", [
    {"ancestry": "INVALID"}, {"source": "invalid"}, {"panel": "invalid"},
    {"chromosome": 0}, {"chromosome": 23}, {"chromosome": True},
    {"chromosome": 1.5}, {"chromosome": "22"}, {"chromosome": []},
    {"chromosome": [1, False]}, {"chromosome": [1, "2"]},
    {"source": "zenodo", "chromosome": 22},
])
def test_validation_before_io(tmp_path, monkeypatch, options):
    monkeypatch.setattr(data_utils, "urlopen", lambda *a, **kw: pytest.fail("unexpected download"))
    with pytest.raises(ValueError):
        data_utils.download_ld_matrix(tmp_path / "absent", **({"ancestry": "EUR"} | options))
    assert not (tmp_path / "absent").exists()


def test_existing_files_and_explicit_overwrite(tmp_path, monkeypatch):
    calls = []

    def fetch(*args, **kwargs):
        calls.append(args)
        return Response()

    monkeypatch.setattr(data_utils, "urlopen", fetch)
    paths = data_utils.download_ld_matrix(tmp_path, ancestry="MID", chromosome=1)
    paths[0].write_bytes(b"existing")
    data_utils.download_ld_matrix(tmp_path, ancestry="MID", chromosome=1)
    assert len(calls) == 1
    assert paths[0].read_bytes() == b"existing"
    data_utils.download_ld_matrix(tmp_path, ancestry="MID", chromosome=1, overwrite=True)
    assert len(calls) == 2
    assert paths[0].read_bytes() == b"archive bytes"


@pytest.mark.parametrize("failure", ["http", "incomplete", "empty", "stream"])
def test_failure_preserves_existing_archive_and_cleans_temporary_files(tmp_path, monkeypatch, failure):
    path = tmp_path / "ukb" / "EUR" / "chr_22.zip"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"previous complete download")

    class BrokenResponse(Response):
        def read(self, *args):
            raise OSError("connection interrupted")

    def fetch(url, **kwargs):
        if failure == "http":
            raise HTTPError(url, 404, "Not found", {}, None)
        if failure == "stream":
            return BrokenResponse()
        return Response(b"partial", length=100) if failure == "incomplete" else Response(b"")

    monkeypatch.setattr(data_utils, "urlopen", fetch)
    with pytest.raises(OSError, match="ukb/EUR/chr_22.zip from huggingface"):
        data_utils.download_ld_matrix(tmp_path, ancestry="EUR", chromosome=22, overwrite=True)
    assert path.read_bytes() == b"previous complete download"
    assert not list(tmp_path.rglob("*.part"))


def test_failed_new_download_can_be_retried(tmp_path, monkeypatch):
    def fetch(url, **kwargs):
        return Response(b"partial", length=100) if "chr_2.zip" in url else Response()

    monkeypatch.setattr(data_utils, "urlopen", fetch)
    with pytest.raises(OSError):
        data_utils.download_ld_matrix(tmp_path, ancestry="EUR", chromosome=[1, 2])
    assert (tmp_path / "ukb/EUR/chr_1.zip").is_file()
    assert not (tmp_path / "ukb/EUR/chr_2.zip").exists()
    monkeypatch.setattr(data_utils, "urlopen", lambda *a, **kw: Response())
    assert len(data_utils.download_ld_matrix(tmp_path, ancestry="EUR", chromosome=[1, 2])) == 2


@pytest.mark.parametrize("panel,repository", [("ukb", "ukb-ld"), ("hgdp-1kg", "hgdp-1kg-ld")])
def test_hub_interface_when_installed(tmp_path, monkeypatch, panel, repository):
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        assert kwargs["repo_id"] == f"shz9/{repository}"
        assert kwargs["repo_type"] == "dataset"
        assert kwargs["revision"] == "main"
        staged = Path(kwargs["local_dir"]) / kwargs["filename"]
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged.write_bytes(b"Hub download")
        return str(staged)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))
    monkeypatch.setattr(data_utils, "urlopen", lambda *a, **kw: pytest.fail("unexpected HTTP fallback"))
    options = dict(ancestry="eur", panel=panel, chromosome=[22, 21, 22])
    paths = data_utils.download_ld_matrix(tmp_path, **options)
    assert [call["filename"] for call in calls] == ["EUR/chr_22.zip", "EUR/chr_21.zip"]
    assert all(call["force_download"] is False for call in calls)
    assert all(path.read_bytes() == b"Hub download" for path in paths)
    assert paths == [tmp_path / panel / "EUR" / f"chr_{c}.zip" for c in [22, 21]]
    assert data_utils.download_ld_matrix(tmp_path, **options) == paths
    assert len(calls) == 2
    data_utils.download_ld_matrix(tmp_path, overwrite=True, **options)
    assert len(calls) == 4
    assert all(call["force_download"] for call in calls[2:])
    assert all(not Path(call["local_dir"]).exists() for call in calls)


@pytest.mark.parametrize("empty", [False, True])
def test_hub_failure_is_atomic_and_does_not_fall_back(tmp_path, monkeypatch, empty):
    path = tmp_path / "ukb/EUR/chr_22.zip"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"existing")

    def download(**kwargs):
        staged = Path(kwargs["local_dir"]) / "partial"
        staged.write_bytes(b"")
        if not empty:
            raise OSError("Hub transfer interrupted")
        return str(staged)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))
    monkeypatch.setattr(data_utils, "urlopen", lambda *a, **kw: pytest.fail("unexpected HTTP fallback"))
    with pytest.raises(OSError, match="Could not download ukb/EUR/chr_22.zip"):
        data_utils.download_ld_matrix(tmp_path, ancestry="EUR", chromosome=22, overwrite=True)
    assert path.read_bytes() == b"existing"
    assert list(path.parent.iterdir()) == [path]


def test_zenodo_uses_http_even_with_hub_installed(tmp_path, monkeypatch):
    def download(**kwargs):
        pytest.fail("Zenodo should not use the Hub interface")

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))
    monkeypatch.setattr(data_utils, "urlopen", lambda *a, **kw: Response())
    paths = data_utils.download_ld_matrix(tmp_path, ancestry="EUR", source="zenodo")
    assert paths[0].read_bytes() == b"archive bytes"

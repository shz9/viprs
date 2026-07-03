from importlib.metadata import version

import viprs


def test_runtime_version_matches_package_metadata():
    assert viprs.__version__ == version("viprs")

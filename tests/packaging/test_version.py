from importlib.metadata import version

import viprs
from viprs.model import VIPRS, VIPRSMix


def test_runtime_version_matches_package_metadata():
    assert viprs.__version__ == version("viprs")


def test_top_level_model_exports_are_deliberately_minimal():
    assert viprs.VIPRS is VIPRS
    assert viprs.VIPRSMix is VIPRSMix
    assert not hasattr(viprs, "LDPredInf")
    assert not hasattr(viprs, "download_ld_matrix")


def test_ascii_logo_preserves_art_alignment_with_ascii_spaces():
    lines = viprs.make_ascii_logo(left_padding=3).splitlines()

    assert "\u2001" not in "\n".join(lines)
    assert all(line.startswith(" " * 3) for line in lines)
    art_left = lines[2].index("___")
    assert lines[1].index("_____") - art_left == 8
    assert lines[6].index("/_/") - art_left == 14

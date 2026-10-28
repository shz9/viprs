from ._version import __release_date__, __version__
from .model import VIPRS, VIPRSMix


def make_ascii_logo(desc=None, left_padding=None):
    """
    Generate an ASCII logo for the VIPRS package.
    :param desc: A string description to be added below the logo.
    :param left_padding: Padding to the left of the logo.

    :return: A string containing the ASCII logo.
    """

    logo = r"""
        _____
___   _____(_)________ ________________
__ | / /__  / ___  __ \__  ___/__  ___/
__ |/ / _  /  __  /_/ /_  /    _(__  )
_____/  /_/   _  .___/ /_/     /____/
              /_/
    """

    # Keep the logo on a single, predictable terminal column grid. U+2001
    # (EM QUAD) is not rendered at the same width as an ASCII space by every
    # terminal, despite both counting as one Python character. Also center the
    # ASCII art as one block so the relative indentation of its rows is kept.
    art_lines = logo.splitlines()
    while art_lines and not art_lines[0].strip():
        art_lines.pop(0)
    while art_lines and not art_lines[-1].strip():
        art_lines.pop()

    text_lines = [
        "Variational Inference of Polygenic Risk Scores",
        f"Version: {__version__} | Release date: {__release_date__}",
        "Author: Shadi Zabad, University of Oxford",
    ]

    # Find the maximum length of the lines
    art_width = max(len(line) for line in art_lines)
    max_len = max(art_width, *(len(line) for line in text_lines))
    if desc is not None:
        max_len = max(max_len, len(desc))

    # Right-pad the art before centering it to preserve its internal layout.
    lines = [line.ljust(art_width).center(max_len) for line in art_lines]
    lines.extend(line.center(max_len) for line in text_lines)

    # Add separators at the top and bottom
    lines.insert(0, "*" * max_len)
    lines.append("*" * max_len)

    if desc is not None:
        lines.append(desc.center(max_len))

    if left_padding is not None:
        for i, l in enumerate(lines):
            lines[i] = " " * left_padding + l

    return "\n".join(lines)

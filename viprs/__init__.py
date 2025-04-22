
__version__ = '0.1.3'
__release_date__ = 'April 2025'

from .model import VIPRS
from .utils.data_utils import *


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

    lines = logo.replace(' ', '\u2001').splitlines()[1:]
    lines.append("Variational Inference of Polygenic Risk Scores")
    lines.append(f"Version: {__version__} | Release date: {__release_date__}")
    lines.append("Author: Shadi Zabad, McGill University")

    # Find the maximum length of the lines
    max_len = max([len(l) for l in lines])
    if desc is not None:
        max_len = max(max_len, len(desc))

    # Pad the lines to the same length
    for i, l in enumerate(lines):
        lines[i] = l.center(max_len)

    # Add separators at the top and bottom
    lines.insert(0, '*' * max_len)
    lines.append('*' * max_len)

    if desc is not None:
        lines.append(desc.center(max_len))

    if left_padding is not None:
        for i, l in enumerate(lines):
            lines[i] = '\u2001' * left_padding + l

    return "\n".join(lines)

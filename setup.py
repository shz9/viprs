from setuptools import setup, Extension, find_packages
# from extension_helpers import add_openmp_flags_if_available
import numpy as np
import os

try:
    from Cython.Build import cythonize
except ImportError:
    cythonize = None

# ------------------------------------------------------
# Cython dependencies:


# https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#distributing-cython-modules
def no_cythonize(extensions, **_ignore):
    for extension in extensions:
        sources = []
        for sfile in extension.sources:
            path, ext = os.path.splitext(sfile)
            if ext in (".pyx", ".py"):
                if extension.language == "c++":
                    ext = ".cpp"
                else:
                    ext = ".c"
                sfile = path + ext
            sources.append(sfile)
        extension.sources[:] = sources
    return extensions


extensions = [
    Extension("viprs.utils.math_utils",
              ["viprs/utils/math_utils.pyx"],
              libraries=["m"],
              include_dirs=[np.get_include()]),
    Extension("viprs.utils.run_stats",
              ["viprs/utils/run_stats.pyx"],
              include_dirs=[np.get_include()]),
    Extension("viprs.model.PRSModel",
              ["viprs/model/PRSModel.pyx"],
              include_dirs=[np.get_include()]),
    Extension("viprs.model.VIPRS",
              ["viprs/model/VIPRS.pyx"],
              include_dirs=[np.get_include()]),
    Extension("viprs.model.VIPRSGrid",
              ["viprs/model/VIPRSGrid.pyx"],
              include_dirs=[np.get_include()]),
    Extension("viprs.model.VIPRSGridSearch",
              ["viprs/model/VIPRSGridSearch.pyx"],
              include_dirs=[np.get_include()]),
    Extension("viprs.model.VIPRSBMA",
              ["viprs/model/VIPRSBMA.pyx"],
              include_dirs=[np.get_include()]),
    Extension("viprs.model.VIPRSMix",
              ["viprs/model/VIPRSMix.pyx"],
              include_dirs=[np.get_include()]),
    Extension("viprs.model.VIPRSAlpha",
              ["viprs/model/VIPRSAlpha.pyx"],
              include_dirs=[np.get_include()]),
    Extension("viprs.model.VIPRSLDPred",
              ["viprs/model/VIPRSLDPred.pyx"],
              include_dirs=[np.get_include()])
]

"""
# Uncomment here in case you add openMP support:
# Add any extension that requires openMP here:
openmp_extensions = []  # ['viprs.model.VIPRSGrid']

for ext in extensions:
    if ext.name in openmp_extensions:
        add_openmp_flags_if_available(ext)
"""

if cythonize is not None:
    compiler_directives = {
        "language_level": 3,
        "embedsignature": True,
        'boundscheck': False,
        'wraparound': False,
        'nonecheck': False,
        'cdivision': True
    }
    extensions = cythonize(extensions, compiler_directives=compiler_directives)
else:
    extensions = no_cythonize(extensions)

# ------------------------------------------------------
# Read description/dependencies from file:

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt") as fp:
    install_requires = fp.read().strip().split("\n")

with open("requirements-optional.txt") as fp:
    opt_requires = fp.read().strip().split("\n")

# ------------------------------------------------------

setup(
    name="viprs",
    version="0.0.4",
    author="Shadi Zabad",
    author_email="shadi.zabad@mail.mcgill.ca",
    description="Variational Inference of Polygenic Risk Scores (VIPRS)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shz9/viprs",
    classifiers=[
        'Programming Language :: Python',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    package_dir={'': '.'},
    packages=find_packages(),
    package_data={'viprs': ['model/*.pxd', 'utils/*.pxd']},
    scripts=['bin/viprs_fit', 'bin/viprs_score', 'bin/viprs_evaluate'],
    install_requires=install_requires,
    extras_require={'full': opt_requires},
    ext_modules=extensions,
    zip_safe=False
)

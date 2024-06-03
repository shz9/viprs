from setuptools import setup, Extension, find_packages
from extension_helpers import add_openmp_flags_if_available
from extension_helpers._openmp_helpers import check_openmp_support
import pkgconfig
import numpy as np
import warnings
import os


try:
    from Cython.Build import cythonize
except ImportError:
    warnings.warn("Cython not found.")
    cythonize = None

# ------------------------------------------------------
# Find and set BLAS-related flags and paths:


def find_blas_libraries():
    """
    Find BLAS libraries on the system using pkg-config.
    This function will return the include directories (compiler flags)
    and the linker flags to enable building the C/C++/Cython extensions
    that require BLAS (or whose performance would be enhanced with BLAS).

    We use pkg-config (as encapsulated in the `pkgconfig` Python package)
    to perform this search. Note that we augment the pkg-config
    search path with the conda library path (if available) to
    enable linking against BLAS libraries installed via Conda.

    :return: A dictionary with the following keys:
        * 'found': A boolean indicating whether BLAS libraries were found.
        * 'include_dirs': A list of include directories (compiler flags).
        * 'extra_link_args': A list of linker flags.
        * 'define_macros': A list of macros to define.
        * 'libraries': A list of libraries to link against.
    """

    # STEP 0: Get the current pkg-config search path:
    current_pkg_config_path = os.getenv("PKG_CONFIG_PATH", "")

    # STEP 1: Augment the pkg-config search path with
    # the path of the current Conda environment (if exists).
    # This can leverage BLAS libraries installed via Conda.

    conda_path = os.getenv("CONDA_PREFIX")

    if conda_path is not None:
        conda_pkgconfig_path = os.path.join(conda_path, 'lib/pkgconfig')
        if os.path.isdir(conda_pkgconfig_path):
            current_pkg_config_path += ":" + conda_pkgconfig_path

    # STEP 2: Add the updated path to the environment variable:
    os.environ["PKG_CONFIG_PATH"] = current_pkg_config_path

    # STEP 3: Get all pkg-config packages and filter to
    # those that have "blas" in the name.
    blas_packages = [pkg for pkg in pkgconfig.list_all()
                     if "blas" in pkg]

    # First check: Make sure that compiler flags are defined and a
    # valid cblas.h header file exists in the include directory:
    if len(blas_packages) >= 1:

        blas_packages = [pkg for pkg in blas_packages
                         if pkgconfig.cflags(pkg) and
                         os.path.isfile(os.path.join(pkgconfig.variables(pkg)['includedir'], 'cblas.h'))]

    # If there remains more than one library after the previous
    # search and filtering steps, then apply some heuristics
    # to select the most relevant one:
    if len(blas_packages) > 1:
        # Check if the information about the most relevant library
        # can be inferred from numpy. Note that this interface from
        # numpy changes quite often between versions, so it's not
        # a reliable check. But in case it works on some systems,
        # we use it to link to the same library as numpy:
        try:
            for pkg in blas_packages:
                if pkg in np.__config__.get_info('blas_opt')['libraries']:
                    blas_packages = [pkg]
                    break
        except (KeyError, AttributeError):
            pass

    # If there are still multiple libraries, then apply some
    # additional heuristics (based on name matching) to select
    # the most relevant one. Some libraries (e.g. flexiblas) are published with support for 64bit
    # and they expose libraries for non-BLAS API (with the _api suffix).
    # Ignore these here if that is the case?
    if len(blas_packages) > 1:
        # Some libraries (e.g. flexiblas) are published with support for 64bit
        # and they expose libraries for non-BLAS API (with the _api suffix).
        # Ignore these here if that is the case?

        idx_to_remove = set()

        for pkg1 in blas_packages:
            if pkg1 != 'blas':
                for i, pkg2 in enumerate(blas_packages):
                    if pkg1 != pkg2 and pkg1 in pkg2:
                        idx_to_remove.add(i)

        blas_packages = [pkg for i, pkg in enumerate(blas_packages) if i not in idx_to_remove]

    # After applying all the heuristics, out of all the remaining libraries,
    # select the first one in the list. Not the greatest solution, maybe
    # down the line we can use the same BLAS order as numpy.
    if len(blas_packages) >= 1:
        final_blas_pkg = blas_packages[0]
    else:
        final_blas_pkg = None

    # STEP 4: If a relevant BLAS package was found, extract the flags
    # needed for building the Cython/C/C++ extensions:

    if final_blas_pkg is not None:
        blas_info = pkgconfig.parse(final_blas_pkg)
        blas_info['define_macros'] = [('HAVE_CBLAS', None)]
    else:
        blas_info = {
            'include_dirs': [],
            'library_dirs': [],
            'libraries': [],
            'define_macros': [],
        }
        warnings.warn("""
            ********************* WARNING *********************
            BLAS library header files not found on your system. 
            This may slow down some computations. If you are 
            using conda, we recommend installing BLAS libraries 
            beforehand.
            ********************* WARNING *********************
        """, stacklevel=2)

    return blas_info


blas_flags = find_blas_libraries()

# ------------------------------------------------------
# Build cython extensions:


def no_cythonize(cy_extensions, **_ignore):
    """
    Copied from:
    https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#distributing-cython-modules
    """

    for ext in cy_extensions:
        sources = []
        for s_file in ext.sources:
            path, ext = os.path.splitext(s_file)
            if ext in (".pyx", ".py"):
                if ext.language == "c++":
                    ext = ".cpp"
                else:
                    ext = ".c"
                s_file = path + ext
            sources.append(s_file)
        ext.sources[:] = sources

    return extensions


extensions = [
    Extension("viprs.utils.math_utils",
              ["viprs/utils/math_utils.pyx"],
              libraries=[[], ["m"]][os.name != 'nt'],  # Only include for non-Windows systems
              include_dirs=[np.get_include()],
              extra_compile_args=["-O3"]),
    Extension("viprs.model.vi.e_step",
              ["viprs/model/vi/e_step.pyx"],
              include_dirs=[np.get_include()],
              define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
              extra_compile_args=["-O3"]),
    Extension("viprs.model.vi.e_step_cpp",
              ["viprs/model/vi/e_step_cpp.pyx"],
              language="c++",
              libraries=blas_flags['libraries'],
              include_dirs=[np.get_include()] + blas_flags['include_dirs'],
              library_dirs=blas_flags['library_dirs'],
              define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")] + blas_flags['define_macros'],
              extra_compile_args=["-O3", "-std=c++17"])
]

if check_openmp_support():
    # Add any extension that requires openMP here:
    openmp_extensions = ['viprs.model.vi.e_step_cpp', 'viprs.model.vi.e_step']

    for omp_ext in extensions:
        if omp_ext.name in openmp_extensions:
            add_openmp_flags_if_available(omp_ext)
else:
    warnings.warn("""
        ******************** WARNING ********************
        OpenMP library not found on your system. This 
        means that some computations may be slower than 
        expected. It will preclude using multithreading 
        in the coordinate ascent optimization algorithm.
        ******************** WARNING ********************
    """)


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

with open("requirements-test.txt") as fp:
    test_requires = fp.read().strip().split("\n")

with open("requirements-docs.txt") as fp:
    doc_requires = fp.read().strip().split("\n")

# ------------------------------------------------------

setup(
    name="viprs",
    version="0.1.2",
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
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12'
    ],
    package_dir={'': '.'},
    packages=find_packages(),
    python_requires=">=3.8",
    package_data={'viprs': ['model/vi/*.pxd', 'utils/*.pxd']},
    scripts=['bin/viprs_fit', 'bin/viprs_score', 'bin/viprs_evaluate'],
    install_requires=install_requires,
    extras_require={'opt': opt_requires, 'test': test_requires, 'docs': doc_requires},
    ext_modules=extensions,
    zip_safe=False
)

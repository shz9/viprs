from setuptools import setup, Extension, find_packages
from extension_helpers import add_openmp_flags_if_available
from extension_helpers._openmp_helpers import check_openmp_support
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


def get_blas_include_dirs():
    """
    Get the include directories for the BLAS library from numpy build configuration.

    NOTE: np.distutils will be deprecated in future versions of numpy. Find alternative solutions
    to linking to BLAS libraries. Alternative solutions:

    * Use the `blas_opt` key in the `numpy.__config__.show()` output to get the include directories.
    * meson builder
    * ...
    """

    cblas_lib_path = None

    # Attempt (1): Getting the information from numpy distutils:
    try:
        from numpy.distutils.system_info import get_info
        cblas_lib_path = get_info('blas_opt')['include_dirs']
    except (AttributeError, KeyError, ImportError, ModuleNotFoundError):
        pass

    # Attempt (2): For newer versions of numpy, obtain information from np.show_config:
    if cblas_lib_path is None:
        try:
            cblas_lib_path = [np.show_config(mode='dicts')['Build Dependencies']['blas']['include directory']]
        except Exception:
            pass

    # Attempt (3): Obtain information from conda environment:
    if cblas_lib_path is None:
        # If not found, check if the library is present in the
        # conda environment:
        conda_path = os.getenv("CONDA_PREFIX")
        if conda_path is not None:
            # If the header file exists in the conda environment, use it:
            if os.path.isfile(os.path.join(conda_path, 'include', 'cblas.h')):
                cblas_lib_path = [os.path.join(conda_path, 'include')]

    # Attempt (4): Obtain information from environment variable:
    if cblas_lib_path is None:
        cblas_lib_path = os.getenv('BLAS_INCLUDE_DIR')

    # If the header file is not found, issue a warning:
    if (cblas_lib_path is None) or (not os.path.isfile(os.path.join(cblas_lib_path[0], 'cblas.h'))):
        # Ok, we give up...
        warnings.warn("""
            ******************** WARNING ********************
            BLAS library header files not found on your system. 
            This may slow down some computations. If the
            library is present on your system, please link to 
            it explicitly by setting the BLAS_INCLUDE_DIR 
            environment variable prior to installation.
        """)

        cblas_lib_path = []

    # Define macros based on whether CBLAS header exists
    macros = [('HAVE_CBLAS', None)] if len(cblas_lib_path) > 0 else []

    return len(cblas_lib_path) > 0, cblas_lib_path, macros


blas_found, blas_include, blas_macros = get_blas_include_dirs()

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
              libraries=[[], ["cblas"]][blas_found],
              include_dirs=[np.get_include()] + blas_include,
              define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")] + blas_macros,
              extra_compile_args=["-O3"])
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
    version="0.1.0",
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

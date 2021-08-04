from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy as np
import os
import platform

#if platform.system() == 'Darwin':
#    os.environ['CC'] = '/usr/local/opt/llvm/bin/clang++'

ext_modules = cythonize([
    Extension("prs.src.c_utils",
              ["prs/src/c_utils.pyx"],
              libraries=["m"],
              extra_compile_args=["-ffast-math"]),
    Extension("prs.src.run_stats",
              ["prs/src/run_stats.pyx"],
              libraries=["m"],
              extra_compile_args=["-ffast-math"]),
    Extension("prs.src.PRSModel",
              ["prs/src/PRSModel.pyx"],
              libraries=["m"],
              extra_compile_args=["-ffast-math"]),
    Extension("prs.src.VIPRS",
              ["prs/src/VIPRS.pyx"],
              libraries=["m"],
              extra_compile_args=["-ffast-math"]),
    Extension("prs.src.VIPRSSBayes",
              ["prs/src/VIPRSSBayes.pyx"],
              libraries=["m"],
              extra_compile_args=["-ffast-math"]),
    Extension("prs.src.GibbsPRS",
              ["prs/src/GibbsPRS.pyx"],
              libraries=["m"],
              extra_compile_args=["-ffast-math"]),
    Extension("prs.src.GibbsPRSSBayes",
              ["prs/src/GibbsPRSSBayes.pyx"],
              libraries=["m"],
              extra_compile_args=["-ffast-math"])
], language_level="3")

setup(name="prs",
      cmdclass={"build_ext": build_ext},
      ext_modules=ext_modules,
      include_dirs=[np.get_include()],
      compiler_directives={'boundscheck': False, 'wraparound': False,
                           'nonecheck': False, 'cdivision': True},
      script_args=["build_ext"],
      options={'build_ext': {'inplace': True, 'force': True}}
      )


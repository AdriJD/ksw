from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np
import os
from pathlib import Path

opj = os.path.join

path = str(Path(__file__).parent.absolute())

compile_opts = {
    'extra_compile_args': ['-std=c99',
                           '-fopenmp',
                           '-Wno-strict-aliasing',
                           '-g'],
    'extra_link_args': ['-fopenmp',
                        '-g']}

ext_modules = [Extension('ksw.radial_functional',
                        [opj(path, 'cython', 'radial_functional.pyx')],
                         libraries=['radial_functional'],
                         library_dirs=[opj(path, 'lib')],
                         include_dirs=[opj(path, 'include'),
                                       np.get_include()],
                         language_level=3,
                         **compile_opts)]

setup(package_dir={"ksw": "ksw"},
      ext_modules=cythonize(ext_modules))




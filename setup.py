from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np
import os
from pathlib import Path

opj = os.path.join

path = str(Path(__file__).parent.absolute())

compile_opts = {
    'extra_compile_args': ['-shared', '-std=gnu99', 
                           '-fopenmp',
                           '-Wno-strict-aliasing',
                           '-g']}

compiler_directives = {'language_level' : 3}

ext_modules = [Extension('ksw.radial_functional',
                        [opj(path, 'cython', 'radial_functional.pyx')],
                         libraries=['radial_functional'],
                         library_dirs=[opj(path, 'lib')],
                         include_dirs=[opj(path, 'include'),
                                       np.get_include()],
                         runtime_library_dirs=[opj(path, 'lib')],
                         **compile_opts)]

setup(name='ksw',
      packages=['ksw'],
      ext_modules=cythonize(ext_modules,
                            compiler_directives=compiler_directives))





import numpy as np
from setuptools import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize('ucb_score_cython.pyx'), extra_compile_args=['-O3'], include_dirs=[np.get_include()])

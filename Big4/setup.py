
from setuptools import setup
from Cython.Build import cythonize
import Cython.Compiler.Options
import numpy

Cython.Compiler.Options.annotate = True

setup(  name = ("sample_004_Big_cython.pyx"),
        ext_modules=cythonize("sample_004_Big_cython.pyx", compiler_directives={'language_level' : "3"}, annotate= True),
        include_dirs= [numpy.get_include()]
)

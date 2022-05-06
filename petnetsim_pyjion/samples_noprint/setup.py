
from setuptools import setup
from Cython.Build import cythonize
import Cython.Compiler.Options
import numpy


Cython.Compiler.Options.annotate = True

setup(  name = ("test_all"),
        ext_modules=cythonize("test_all.pyx", compiler_directives={'language_level' : "3"}, annotate= True),
        include_dirs= [numpy.get_include()]
)

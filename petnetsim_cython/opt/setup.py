
from setuptools import setup
from Cython.Build import cythonize
import Cython.Compiler.Options
import numpy

Cython.Compiler.Options.annotate = True

setup(  name = ("petnetsim_opt"),
        ext_modules=cythonize("petnetsim_opt.pyx", compiler_directives={'language_level' : "3"}, annotate= True),
        include_dirs= [numpy.get_include()]
)

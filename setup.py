from distutils.core import setup, Extension
import numpy as np

ext_modules = [ Extension('nanopre.hmm', sources = ['nanopre/hmm.cpp'],extra_compile_args=['-std=c++11'])]

setup(
	name = 'nanopre',
	version = '0.1',
	include_dirs = [np.get_include()],
	packages = ['nanopre'],
	ext_modules = ext_modules
)

from distutils.core import setup, Extension
import numpy as np

ext_modules = [ Extension('boostnano.hmm', sources = ['boostnano/hmm.cpp'],extra_compile_args=['-std=c++11'])]

setup(
	name = 'boostnano',
	version = '0.1',
	include_dirs = [np.get_include()],
	packages = ['boostnano'],
	ext_modules = ext_modules
)

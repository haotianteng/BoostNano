from setuptools import find_packages
from distutils.core import setup, Extension
import numpy as np

ext_modules = [ Extension('boostnano.hmm', sources = ['boostnano/hmm.cpp'])]
exec(open('boostnano/_version.py').read()) #readount the __version__ variable
setup(
	name = 'boostnano',
	version = __version__,
	include_dirs = [np.get_include()],
	ext_modules = ext_modules,
	packages=find_packages(),
    package_data={
        'boostnano': ['model/*'],  # Include all .txt files under the 'data' directory
    },
    include_package_data=True,
)

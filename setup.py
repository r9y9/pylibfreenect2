# coding: utf-8

from __future__ import with_statement, print_function, absolute_import

from setuptools import setup, find_packages, Extension
from distutils.version import LooseVersion

import numpy as np
import os
from glob import glob
from os.path import join


min_cython_ver = '0.19.0'
try:
    import Cython
    ver = Cython.__version__
    _CYTHON_INSTALLED = ver >= LooseVersion(min_cython_ver)
except ImportError:
    _CYTHON_INSTALLED = False

try:
    if not _CYTHON_INSTALLED:
        raise ImportError('No supported version of Cython installed.')
    from Cython.Distutils import build_ext
    cython = True
except ImportError:
    cython = False

if cython:
    ext = '.pyx'
    cmdclass = {'build_ext': build_ext}
else:
    ext = '.cpp'
    cmdclass = {}
    if not os.path.exists(join("pylibfreenect2", "libfreenect2" + ext)):
        raise RuntimeError("Cython is required to generate C++ codes.")

# define core cython module
ext_modules = [
    Extension(
        name="pylibfreenect2.libfreenect2",
        sources=[join("pylibfreenect2", "libfreenect2" + ext)],
        include_dirs=[np.get_include(), "/usr/local/include/libfreenect2"],
        library_dirs=["/usr/local/lib"],
        libraries=["freenect2"],
        extra_compile_args=[],
        language="c++"),
]

setup(
    name='pylibfreenect2',
    version='0.0.1-dev',
    description='A python wrapper for libfreenect2',
    author='Ryuichi Yamamoto',
    author_email='zryuichi@gmail.com',
    url='https://github.com/r9y9/pylibfreenect2',
    license='MIT',
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    install_requires=[
        'numpy >= 1.8.0',
        'six'
    ],
    tests_require=['nose', 'coverage'],
    extras_require={
        'docs': ['numpydoc', 'sphinx_rtd_theme', 'seaborn'],
        'test': ['nose'],
        'develop': ['cython >= ' + min_cython_ver],
    },
    classifiers=[
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Cython",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
    ],
    keywords=["pylibfreenect2", "libfreenect2", "freenect2"]
)

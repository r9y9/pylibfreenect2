# coding: utf-8

from __future__ import with_statement, print_function, absolute_import

from setuptools import setup, find_packages, Extension
from distutils.version import LooseVersion

import numpy as np
import os
from os.path import join
from subprocess import Popen, PIPE

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
    from Cython.Build import cythonize
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

# should be configurable
dylib_path = "/usr/local/lib/libfreenect2.dylib"


def has_opengl_backend():
    p = Popen("otool -L {} | grep OpenGL".format(dylib_path), stdin=PIPE,
              stdout=PIPE, stderr=PIPE, close_fds=True, shell=True)
    p.wait()
    return len(p.stdout.readlines()) > 0


def has_opencl_backend():
    p = Popen("otool -L {} | grep OpenCL".format(dylib_path), stdin=PIPE,
              stdout=PIPE, stderr=PIPE, close_fds=True, shell=True)
    p.wait()
    return len(p.stdout.readlines()) > 0

ext_modules = cythonize(
    [Extension(
        name="pylibfreenect2.libfreenect2",
        sources=[
            join("pylibfreenect2", "libfreenect2" + ext),
        ],
        include_dirs=[np.get_include(), "/usr/local/include/libfreenect2"],
        library_dirs=["/usr/local/lib"],
        libraries=["freenect2"],
        extra_compile_args=["-std=c++11", "-stdlib=libc++", "-mmacosx-version-min=10.8"],
        extra_link_args=[],
        language="c++")],
    compile_time_env={
        "LIBFREENECT2_WITH_OPENGL_SUPPORT": has_opengl_backend(),
        "LIBFREENECT2_WITH_OPENCL_SUPPORT": has_opencl_backend(),
    }
)


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
        'numpy >= 1.7.0',
        'six'
    ],
    tests_require=['nose', 'coverage'],
    extras_require={
        'docs': ['numpydoc', 'sphinx_rtd_theme', 'seaborn'],
        'test': ['nose'],
        'develop': ['cython >= ' + min_cython_ver],
    },
    classifiers=[
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

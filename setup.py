# coding: utf-8

from __future__ import with_statement, print_function, absolute_import

from setuptools import setup, find_packages, Extension
from distutils.version import LooseVersion

import platform

import numpy as np
import os
from os.path import join, exists
from subprocess import Popen, PIPE
import sys

libfreenect2_install_prefix = os.environ.get(
    "LIBFREENECT2_INSTALL_PREFIX", "/usr/local/")

libfreenect2_include_top = join(libfreenect2_install_prefix, "include")
libfreenect2_library_path = join(libfreenect2_install_prefix, "lib")
libfreenect2_configh_path = join(
    libfreenect2_include_top, "libfreenect2", "config.h")

if not exists(libfreenect2_configh_path):
    raise OSError("{}: is not found".format(libfreenect2_configh_path))

if platform.system() == "Windows":
    lib_candidates = list(filter(lambda l: l.startswith("freenect2."),
                                 os.listdir(join(libfreenect2_library_path))))
else:
    lib_candidates = list(filter(lambda l: l.startswith("libfreenect2."),
                                 os.listdir(join(libfreenect2_library_path))))

if len(lib_candidates) == 0:
    raise OSError("libfreenect2 library cannot be found")

min_cython_ver = '0.21.0'
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


def has_define_in_config(key, close_fds=None):
    if close_fds is None:
        if platform.system() == "Windows":
            close_fds = False
        else:
            close_fds = True

    if platform.system() == "Windows":
        lines = []
        with open(libfreenect2_configh_path, 'r') as f:
            for line in f:
                if key in line:
                    lines.append(line)
    else:
        p = Popen("cat {0} | grep {1}".format(libfreenect2_configh_path, key),
                  stdin=PIPE, stdout=PIPE, stderr=PIPE, close_fds=close_fds, shell=True)
        p.wait()
        lines = p.stdout.readlines()

    if sys.version_info.major >= 3 and not platform.system() == "Windows":
        return len(lines) == 1 and lines[0].startswith(b"#define")
    else:
        return len(lines) == 1 and lines[0].startswith("#define")


if platform.system() == "Darwin":
    extra_compile_args = ["-std=c++11", "-stdlib=libc++",
                          "-mmacosx-version-min=10.8"]
else:
    # should work with Ubuntu 14.04 with anaconda python3 instaleld
    extra_compile_args = ["-std=c++11"]

ext_modules = cythonize(
    [Extension(
        name="pylibfreenect2.libfreenect2",
        sources=[
            join("pylibfreenect2", "libfreenect2" + ext),
        ],
        include_dirs=[np.get_include(),
                      join(libfreenect2_include_top)],
        library_dirs=[libfreenect2_library_path],
        libraries=["freenect2"],
        extra_compile_args=extra_compile_args,
        extra_link_args=[],
        language="c++")],
    compile_time_env={
        "LIBFREENECT2_WITH_OPENGL_SUPPORT":
        has_define_in_config("LIBFREENECT2_WITH_OPENGL_SUPPORT"),
        "LIBFREENECT2_WITH_OPENCL_SUPPORT":
        has_define_in_config("LIBFREENECT2_WITH_OPENCL_SUPPORT"),
    }
)

install_requires = ['numpy >= 1.7.0']
if sys.version_info < (3, 4):
    install_requires.append('enum34')

setup(
    name='pylibfreenect2',
    version='0.1.3',
    description='A python interface for libfreenect2',
    author='Ryuichi Yamamoto',
    author_email='zryuichi@gmail.com',
    url='https://github.com/r9y9/pylibfreenect2',
    license='MIT',
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    install_requires=install_requires,
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

"""Build the pylibfreenect2 Cython extension.

Package metadata lives in pyproject.toml; this file only locates libfreenect2
and configures the extension. libfreenect2 is located, in order, from:

1. ``LIBFREENECT2_BUILD_DIR`` -- an uninstalled CMake build tree (e.g.
   ``libfreenect2/build``). Public headers are taken from
   ``LIBFREENECT2_SOURCE_DIR``/include (default: the build dir's parent),
   generated headers (config.h, export.h) and the library from the build dir.
2. ``LIBFREENECT2_INSTALL_PREFIX`` -- an installed prefix with include/ and lib/.
3. ``pkg-config freenect2``.
4. ``/usr/local``.

On non-Windows platforms the library directory is embedded as an RPATH, so no
``LD_LIBRARY_PATH`` is needed at runtime.
"""

import os
import platform
import re
import shlex
import shutil
import subprocess
from pathlib import Path

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

IS_WINDOWS = platform.system() == "Windows"
LIB_PREFIX = "freenect2." if IS_WINDOWS else "libfreenect2."


def _from_build_dir(build_dir):
    build_dir = Path(build_dir).expanduser().resolve()
    source_dir = Path(
        os.environ.get("LIBFREENECT2_SOURCE_DIR", build_dir.parent)
    ).expanduser().resolve()
    include_dirs = [source_dir / "include", build_dir]
    if not (source_dir / "include" / "libfreenect2" / "libfreenect2.hpp").exists():
        raise OSError(
            f"libfreenect2 headers not found under {source_dir / 'include'}; "
            "set LIBFREENECT2_SOURCE_DIR to the libfreenect2 source checkout"
        )
    return include_dirs, [build_dir / "lib"], "LIBFREENECT2_BUILD_DIR"


def _from_prefix(prefix, origin):
    prefix = Path(prefix).expanduser().resolve()
    return [prefix / "include"], [prefix / "lib"], origin


def _from_pkg_config():
    if shutil.which("pkg-config") is None:
        return None
    try:
        cflags = subprocess.run(
            ["pkg-config", "--cflags-only-I", "freenect2"],
            check=True, capture_output=True, text=True).stdout
        libs = subprocess.run(
            ["pkg-config", "--libs-only-L", "freenect2"],
            check=True, capture_output=True, text=True).stdout
    except subprocess.CalledProcessError:
        return None
    include_dirs = [Path(f[2:]) for f in shlex.split(cflags) if f.startswith("-I")]
    library_dirs = [Path(f[2:]) for f in shlex.split(libs) if f.startswith("-L")]
    if not include_dirs or not library_dirs:
        return None
    return include_dirs, library_dirs, "pkg-config"


def find_libfreenect2():
    if "LIBFREENECT2_BUILD_DIR" in os.environ:
        found = _from_build_dir(os.environ["LIBFREENECT2_BUILD_DIR"])
    elif "LIBFREENECT2_INSTALL_PREFIX" in os.environ:
        found = _from_prefix(os.environ["LIBFREENECT2_INSTALL_PREFIX"],
                             "LIBFREENECT2_INSTALL_PREFIX")
    else:
        found = _from_pkg_config() or _from_prefix("/usr/local", "default prefix")
    include_dirs, library_dirs, origin = found

    config_h = next((d / "libfreenect2" / "config.h" for d in include_dirs
                     if (d / "libfreenect2" / "config.h").exists()), None)
    if config_h is None:
        raise OSError(
            f"libfreenect2/config.h not found in {[str(d) for d in include_dirs]} "
            f"(from {origin}). Set LIBFREENECT2_BUILD_DIR to a libfreenect2 build "
            "tree or LIBFREENECT2_INSTALL_PREFIX to an install prefix."
        )
    if not any(p.name.startswith(LIB_PREFIX)
               for d in library_dirs if d.is_dir() for p in d.iterdir()):
        raise OSError(
            f"libfreenect2 library not found in {[str(d) for d in library_dirs]} "
            f"(from {origin}). Has libfreenect2 been built?"
        )
    return include_dirs, library_dirs, config_h


def has_define(config_h, key):
    pattern = re.compile(r"^\s*#\s*define\s+" + re.escape(key) + r"\b", re.M)
    return bool(pattern.search(config_h.read_text()))


include_dirs, library_dirs, config_h = find_libfreenect2()

if platform.system() == "Darwin":
    extra_compile_args = ["-std=c++11", "-stdlib=libc++"]
elif IS_WINDOWS:
    extra_compile_args = []
else:
    extra_compile_args = ["-std=c++11"]

ext_modules = cythonize(
    [Extension(
        name="pylibfreenect2.libfreenect2",
        sources=[os.path.join("pylibfreenect2", "libfreenect2.pyx")],
        include_dirs=[np.get_include()] + [str(d) for d in include_dirs],
        library_dirs=[str(d) for d in library_dirs],
        runtime_library_dirs=[] if IS_WINDOWS else [str(d) for d in library_dirs],
        libraries=["freenect2"],
        extra_compile_args=extra_compile_args,
        language="c++")],
    compile_time_env={
        key: has_define(config_h, key)
        for key in ("LIBFREENECT2_WITH_OPENGL_SUPPORT",
                    "LIBFREENECT2_WITH_OPENCL_SUPPORT",
                    "LIBFREENECT2_WITH_CUDA_SUPPORT")
    },
)

setup(ext_modules=ext_modules)

from setuptools import setup
from glob import glob

# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir

import sys
import numpy as np

__version__ = "0.0.1"

# The main interface is through Pybind11Extension.
# * You can add cxx_std=11/14/17, and then build_ext can be removed.
# * You can set include_pybind11=false to add the include directory yourself,
#   say from a submodule.
#
# Note:
#   Sort input source files if you glob sources to ensure bit-for-bit
#   reproducible builds (https://github.com/pybind/python_example/pull/53)

ext_modules = [
    Pybind11Extension("_onedal_py_dpc",
        sorted(glob("onedal/**/*.cpp", recursive=True)),
        define_macros = [('ONEDAL_DATA_PARALLEL', 1)],
        libraries = ["onedal_dpc", "onedal_core", "onedal_thread"],
        include_dirs = [np.get_include()],
        cxx_std = 17,
        ),
]

setup(
    name="onedal",
    version=__version__,
    author="",
    author_email="",
    url="",
    description="A python interface to oneDAL library",
    long_description="",
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    # Currently, build_ext only provides an optional "highest supported C++
    # level" feature, but in the future it may provide more features.
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)

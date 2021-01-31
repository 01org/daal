#===============================================================================
# Copyright 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

# Relative paths
set(DAL_ROOT_REL_PATH "../../..")
set(INC_REL_PATH "include")
set(LIB_REL_PATH "lib")
set(DLL_REL_PATH "redist")
set(SUB_DIR "intel64")

# Parse version info if possible
if (NOT "$ENV{DALROOT}" STREQUAL "")
    file(READ $ENV{DALROOT}/include/services/library_version_info.h DAL_VERSION_INFO)
    string(REGEX REPLACE ".*#define __INTEL_DAAL__ ([0-9]+).*" "\\1" DAL_VER_MAJOR "${DAL_VERSION_INFO}")
    string(REGEX REPLACE ".*#define __INTEL_DAAL_MINOR__ ([0-9]+).*" "\\1" DAL_VER_MINOR "${DAL_VERSION_INFO}")
    string(REGEX REPLACE ".*#define __INTEL_DAAL_UPDATE__ ([0-9]+).*" "\\1" DAL_VER_PATCH "${DAL_VERSION_INFO}")
    string(REGEX REPLACE ".*#define __INTEL_DAAL_MAJOR_BINARY__ ([0-9]+).*" "\\1" DAL_VER_MAJOR_BIN "${DAL_VERSION_INFO}")
    string(REGEX REPLACE ".*#define __INTEL_DAAL_MINOR_BINARY__ ([0-9]+).*" "\\1" DAL_VER_MINOR_BIN "${DAL_VERSION_INFO}")
    set(DAL_VERSION "${DAL_VER_MAJOR}.${DAL_VER_MINOR}.${DAL_VER_PATCH}.0")
    set(VERSIONS_SET TRUE)
else()
    set(VERSIONS_SET FALSE)
endif()

# Make a directory for result configs
if (NOT "$ENV{DALROOT}" STREQUAL "")
    set(INSTALL_DIR "$ENV{DALROOT}/lib/cmake/DAL-${DAL_VER_MAJOR}.${DAL_VER_MINOR}.${DAL_VER_PATCH}")
else()
    set(INSTALL_DIR "${CMAKE_CURRENT_LIST_DIR}/..")
endif()
get_filename_component(config_install_dir ${INSTALL_DIR} ABSOLUTE)
file(MAKE_DIRECTORY ${config_install_dir})

configure_file(${CMAKE_CURRENT_LIST_DIR}/../templates/DALConfig.cmake.in ${config_install_dir}/DALConfig.cmake @ONLY)
configure_file(${CMAKE_CURRENT_LIST_DIR}/../templates/DALConfigVersion.cmake.in ${config_install_dir}/DALConfigVersion.cmake @ONLY)

message(STATUS "DALConfig files were created in ${INSTALL_DIR}")
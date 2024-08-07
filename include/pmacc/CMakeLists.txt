#
# Copyright 2015-2023 Erik Zenker, Alexander Grund
#
# This file is part of PMacc.
#
# PMacc is free software: you can redistribute it and/or modify
# it under the terms of either the GNU General Public License or
# the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PMacc is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License and the GNU Lesser General Public License
# for more details.
#
# You should have received a copy of the GNU General Public License
# and the GNU Lesser General Public License along with PMacc.
# If not, see <http://www.gnu.org/licenses/>.
#

################################################################################
# PMacc tests
################################################################################

cmake_minimum_required(VERSION 3.25.0)
project(PMaccTest)

# set helper pathes to find libraries and packages
# Add specific hints
list(APPEND CMAKE_PREFIX_PATH "$ENV{MPI_ROOT}")
list(APPEND CMAKE_PREFIX_PATH "$ENV{CUDA_ROOT}")
list(APPEND CMAKE_PREFIX_PATH "$ENV{BOOST_ROOT}")
list(APPEND CMAKE_PREFIX_PATH "$ENV{VT_ROOT}")
# Add from environment after specific env vars
list(APPEND CMAKE_PREFIX_PATH "$ENV{CMAKE_PREFIX_PATH}")


################################################################################
# CMake policies
#
# Search in <PackageName>_ROOT:
#   https://cmake.org/cmake/help/v3.12/policy/CMP0074.html
################################################################################

if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif()


###############################################################################
# Language Flags
###############################################################################

# enforce C++17
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)


################################################################################
# Directory of this file.
################################################################################
set(PMACC_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Normalize the path (e.g. remove ../)
get_filename_component(PMACC_ROOT_DIR "${PMACC_ROOT_DIR}" ABSOLUTE)

################################################################################
# PMacc
################################################################################

find_package(PMacc REQUIRED CONFIG PATHS ${CMAKE_CURRENT_SOURCE_DIR})


###############################################################################
# Catch2
###############################################################################

add_subdirectory(${PMACC_ROOT_DIR}/../../thirdParty/catch2 ${CMAKE_BINARY_DIR}/catch2)


################################################################################
# MPI
################################################################################

# MPI is provided by pmacc but to execute the binaries via root additional flags must be given to the execution command
option(USE_MPI_AS_ROOT_USER "add --allow-run-as-root mpiexec used by ctest" OFF)

if(USE_MPI_AS_ROOT_USER)
    set(MPI_RUNTIME_FLAGS "--allow-run-as-root")
endif()

###############################################################################
# Targets
###############################################################################

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/test)

# CTest
enable_testing()

# Test cases
# Each *UT.cpp file is an independent executable with one or more test cases
file(GLOB_RECURSE TESTS test/*UT.cpp)
foreach(dim 2 3)
    foreach(testCaseFilepath ${TESTS})
        get_filename_component(testCaseFilename ${testCaseFilepath} NAME)
        string(REPLACE "UT.cpp" "" testCase ${testCaseFilename})
        set(testExe "${PROJECT_NAME}-${testCase}-${dim}D")
        alpaka_add_executable(${testExe} ${testCaseFilepath})
        target_compile_definitions(${testExe} PRIVATE TEST_DIM=${dim})
        target_link_libraries(${testExe} PUBLIC Catch2 Catch2WithMain)
        target_link_libraries(${testExe} PRIVATE pmacc::pmacc)
        add_test(NAME "${testCase}-${dim}D" COMMAND mpiexec ${MPI_RUNTIME_FLAGS} -n 1 ./${testExe})
    endforeach()
    string(REPLACE "-DTEST_DIM=${dim}" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endforeach()

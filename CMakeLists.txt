# Serene programming language.
#
#  Copyright (c) 2019-2021 Sameer Rahmani <lxsameer@gnu.org>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
cmake_minimum_required(VERSION 3.16)

# Project name and a few useful settings. Other commands can pick up the results
project(Serene
  VERSION 0.1.0
  DESCRIPTION "Serene language is a modern Lisp."
  LANGUAGES CXX C)

# Clangd command file
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

# Policies ==========================
cmake_policy(SET CMP0116 NEW)

# User Options ======================
option(CPP_20_SUPPORT "C++20 Support" OFF)
option(SERENE_BUILD_TESTING "Enable tests" OFF)
option(SERENE_ENABLE_BUILDID "Enable build id." OFF)
option(SERENE_ENABLE_THINLTO "Enable ThisLTO." ON)
option(SERENE_ENABLE_DOCS "Enable document generation" OFF)
option(SERENE_DISABLE_CCACHE "Disable automatic ccache integration" OFF)

# Only do these if this is the main project, and not if it is included through add_subdirectory
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  ## Settings =======================
  # specify the C++ standard
  if (CPP_20_SUPPORT)
    set(CMAKE_CXX_STANDARD 20)
  else()
    set(CMAKE_CXX_STANDARD 17)
  endif()

  set(CMAKE_CXX_STANDARD_REQUIRED True)

  # Setup the source locations
  set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
  set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)
  set(BIN_DIR ${CMAKE_SOURCE_DIR}/bin)

  set(CMAKE_CXX_CLANG_TIDY clang-tidy)
  # Let's ensure -std=c++xx instead of -std=g++xx
  set(CMAKE_CXX_EXTENSIONS OFF)

  set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/scripts/cmake")
  set(MemoryCheckCommand "valgrind")

  configure_file(${INCLUDE_DIR}/serene/config.h.in serene/config.h)

  # Let's nicely support folders in IDEs
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)

  # Setup the basic compiler flags
  add_compile_options(
    -Wall
    -Wextra
    -Werror
    -fno-rtti
    -fno-builtin-strlen

    # Dedicate a section to each function, so the linker
    # can do a better job on dead code elimination
    -ffunction-sections
    -fdata-sections

    $<$<CONFIG:DEBUG>:-g3>
    $<$<CONFIG:DEBUG>:-ggdb>
    # For the sake of debugging
    $<$<CONFIG:DEBUG>:-fno-inline>
    # To make the local ccache happy
    $<$<CONFIG:DEBUG>:-fdebug-prefix-map=${PROJECT_SOURCE_DIR}=.>

    $<$<CONFIG:DEBUG>:-fno-omit-frame-pointer>
    $<$<CONFIG:RELEASE>:-fomit-frame-pointer>

    $<$<CONFIG:RELEASE>:-O3>
    )

  add_link_options(
    # We enforce the lld linker
    -fuse-ld=lld
    $<$<CONFIG:DEBUG>:-fsanitize=address>

    # Do not link against shared libraries
    #--static
    )


  find_program(LLD_PROGRAM REQUIRED NAMES lld)
  find_program(MLIRTBLGEN_PROGRAM REQUIRED NAMES mlir-tblgen)

  if(SERENE_ENABLE_BUILDID)
    add_link_options(-Wl,--build-id)
  endif()


  if(SERENE_ENABLE_THINLTO)
  endif()


  # CCache support ==============================
  if(SERENE_DISABLE_CCACHE)
    message(STATUS "CCache support is disabled")
  else()

    find_program(CCACHE_PROGRAM ccache)

    if(CCACHE_PROGRAM)
      message(STATUS "Found CCache")
      set(SERENE_CCACHE_MAXSIZE "" CACHE STRING "Size of ccache")
      set(SERENE_CCACHE_DIR "" CACHE STRING "Directory to keep ccached data")
      set(SERENE_CCACHE_PARAMS "CCACHE_CPP2=yes CCACHE_HASHDIR=yes"
        CACHE STRING "Parameters to pass through to ccache")

      set(CCACHE_PROGRAM "${SERENE_CCACHE_PARAMS} ${CCACHE_PROGRAM}")

      if (SERENE_CCACHE_MAXSIZE)
        set(CCACHE_PROGRAM "CCACHE_MAXSIZE=${SERENE_CCACHE_MAXSIZE} ${CCACHE_PROGRAM}")
      endif()
      if (SERENE_CCACHE_DIR)
        set(CCACHE_PROGRAM "CCACHE_DIR=${SERENE_CCACHE_DIR} ${CCACHE_PROGRAM}")
      endif()
      set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ${CCACHE_PROGRAM})
    else()
      message(FATAL_ERROR "Unable to find the program ccache. Set SERENE_DISABLE_CCACHE to ON")
    endif()
  endif()


  # LLVM setup =========================================
  find_package(LLVM REQUIRED CONFIG)
  find_package(MLIR REQUIRED CONFIG)

  message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
  set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include(HandleLLVMOptions)



  include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
  separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
  add_definitions(${LLVM_DEFINITIONS_LIST})

  link_directories(${LLVM_BUILD_LIBRARY_DIR})
  add_definitions(${LLVM_DEFINITIONS})

  llvm_map_components_to_libnames(llvm_libs support core irreader)



  # Serene Setup ===================================
  include_directories(${PROJECT_SOURCE_DIR}/include)

  # We don't want the generated files from table gen
  # to be treated as local since the contain warnings
  include_directories(SYSTEM ${PROJECT_BINARY_DIR}/include)
  include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)

  # The compiled library code is here
  add_subdirectory(src/serene)
  # The executable code is here
  add_subdirectory(bin)
  add_subdirectory(include)

  # Testing only available if this is the main app
  # Emergency override SERENE_CMAKE_BUILD_TESTING provided as well
  if(SERENE_BUILD_TESTING)
    message("Build the test binary")
    add_subdirectory(src/tests)
  endif()

  if (SERENE_ENABLE_DOCS)
    # Docs only available if this is the main app
    find_package(Doxygen
      REQUIRED dot
      OPTIONAL_COMPONENTS dia)

    if(Doxygen_FOUND)
      add_subdirectory(docs)
    else()
      message(STATUS "Doxygen not found, not building docs")
    endif()
  endif()
endif()

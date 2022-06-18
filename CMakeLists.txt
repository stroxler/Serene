# Serene Programming Language
#
# Copyright (c) 2019-2022 Sameer Rahmani <lxsameer@gnu.org>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 2.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
cmake_minimum_required(VERSION 3.19)

# Project name and a few useful settings. Other commands can pick up the results
project(Serene
  VERSION 1.0.0
  DESCRIPTION "Serene language is a modern Lisp."
  LANGUAGES CXX C)

# Clangd command file
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

# Policies ==========================
cmake_policy(SET CMP0116 OLD)

# User Options ======================
option(CPP_20_SUPPORT "C++20 Support" OFF)
option(SERENE_BUILD_TESTING "Enable tests" OFF)
option(SERENE_ENABLE_BUILDID "Enable build id." OFF)
option(SERENE_ENABLE_THINLTO "Enable ThisLTO." ON)
option(SERENE_ENABLE_DOCS "Enable document generation" OFF)
option(SERENE_ENABLE_TIDY "Enable clang tidy check" OFF)
option(SERENE_DISABLE_CCACHE "Disable automatic ccache integration" OFF)
option(SERENE_ENABLE_DEVTOOLS "Enable the devtools build" OFF)

# libserene
option(SERENE_SHARED_LIB "Build libserene as a shared library" ON)
option(SERENE_SHOW_MLIR_DIALECTS "Print out a list of MLIR dialect libs" OFF)
option(SERENE_SHOW_MLIR_TRANSFORMERS "Print out a list of MLIR dialect transformer libs" OFF)
option(SERENE_SHOW_LLVM_LIBS "Print all the llvm libs available" OFF)

option(SERENE_WITH_MLIR_CL_OPTION "Add support for controlling MLIR via command line options" OFF)
option(SERENE_USE_COMPILER_RT "Use LLVM's compiler-rt" OFF)

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
  set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/libserene/include)

  if(SERENE_ENABLE_TIDY)
    find_program(CLANG_TIDY_PATH NAMES clang-tidy REQUIRED)
  endif()

  find_program(iwyu NAMES include-what-you-use iwyu REQUIRED)
  set(iwyu_path ${iwyu})
  # Let's ensure -std=c++xx instead of -std=g++xx
  set(CMAKE_CXX_EXTENSIONS OFF)

  set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/scripts/cmake")
  set(MemoryCheckCommand "valgrind")

  configure_file(${INCLUDE_DIR}/serene/config.h.in include/serene/config.h)

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
    $<$<CONFIG:DEBUG>:-fsanitize=address>

    $<$<CONFIG:RELEASE>:-O3>
    $<$<CONFIG:RELEASE>:-fmerge-all-constants>
    $<$<CONFIG:RELEASE>:-flto=thin>
    $<$<CONFIG:DEBUG>:-shared-libsan>
    )

  add_link_options(
    # We enforce the lld linker
    -fuse-ld=lld
    -Wl,-gc-sections
    $<$<CONFIG:DEBUG>:-shared-libsan>
    $<$<CONFIG:RELEASE>:-fsanitize-address-globals-dead-stripping>
    $<$<CONFIG:DEBUG>:-fsanitize=address>
    $<$<CONFIG:RELEASE>:-flto=thin>
    $<$<CONFIG:RELEASE>:-s>
    # Do not link against shared libraries
    #--static
    )

  if (SERENE_USE_COMPILER_RT)
    add_link_options(-rtlib=compiler-rt -stdlib=libc++ )
  endif()

  # Do we need to build serene as a shared lib? default is "yes"
  if(SERENE_SHARED_LIB)
    set(BUILD_SHARED_LIBS "${SERENE_SHARED_LIB}")
  endif()

  find_program(LLD_PROGRAM REQUIRED NAMES lld)
  find_program(MLIRTBLGEN_PROGRAM REQUIRED NAMES mlir-tblgen)

  if(SERENE_ENABLE_BUILDID)
    add_link_options(-Wl,--build-id)
  endif()

  #TODO: Setup the THINLTO on release
  if(SERENE_ENABLE_THINLTO)
  endif()

  include(GNUInstallDirs)
  Include(FetchContent)

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
      message(STATUS "Using CCACHE: ${CCACHE_PROGRAM}")
      set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ${CCACHE_PROGRAM})
    else()
      message(FATAL_ERROR "Unable to find the program ccache. Set SERENE_DISABLE_CCACHE to ON")
    endif()
  endif()


  if(SERENE_BUILD_TESTING)
    message(STATUS "Fetching Catch2 v3...")

    FetchContent_Declare(
      Catch2
      GIT_REPOSITORY https://github.com/catchorg/Catch2.git
      GIT_TAG        v3.0.0-preview4
      )
    FetchContent_MakeAvailable(Catch2)
    list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
  endif()

  # LLVM setup =========================================
  find_package(LLVM REQUIRED all-targets CONFIG)
  find_package(MLIR REQUIRED CONFIG)
  find_package(LLD REQUIRED CONFIG)

  message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using LLDConfig.cmake in: ${LLD_DIR}")

  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
  set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLD_CMAKE_DIR}")

  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include(HandleLLVMOptions)

  include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})

  separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
  add_definitions(${LLVM_DEFINITIONS_LIST})

  link_directories(${LLVM_BUILD_LIBRARY_DIR})
  add_definitions(${LLVM_DEFINITIONS})



  # function(llvm_libs_for output components)
  #   # A trick to extract variable number of args beside the first
  #   list(SUBLIST ARGV 1 -1 comps)

  #   # Generate a list of parameters to pass to llvm-config
  #   set(params "")
  #   foreach(lib ${comps})
  #     string(APPEND params "--libs ${lib} ")
  #   endforeach(lib)
  #   string(REPLACE ";" " " params_list "${params}")

  #   # Ask llvm-config for all the libs for the given components
  #   execute_process(COMMAND bash -c "llvm-config-15 ${params_list}"
  #     OUTPUT_VARIABLE llvm_libraries)

  #   string(REGEX REPLACE "\n$" "" llvm_libraries "${llvm_libraries}")
  #   string(REPLACE " " ";" llvm_libs ${llvm_libraries})

  #   # Convert the lib names to cmake target names
  #   set(targets "")
  #   foreach(lib ${llvm_libs})
  #     string(SUBSTRING ${lib} 2 100 target_name)
  #     list(APPEND targets "${target_name}")
  #   endforeach(lib)

  #   set(${output} ${targets} PARENT_SCOPE)
  # endfunction()

  # function(llvm_target_archs output)
  #   set(archs "")
  #   foreach(arch ${LLVM_TARGETS_TO_BUILD})
  #     string(TOLOWER ${arch} arch_name)
  #     list (APPEND archs "${arch_name}")
  #   endforeach(arch)

  #   set(${output} ${archs} PARENT_SCOPE)
  # endfunction()

  # llvm_target_archs(llvm_archs)
  # message(STATUS "-- LLVM is built against:: ${llvm_archs}")

  # llvm_libs_for(llvm_libs1
  #   ${llvm_archs}
  #   core
  #   support
  #   jitlink
  #   orcjit
  #   asmparser
  #   codegen)

  # message(STATUS "-- Libs to link:")
  # message(STATUS ${llvm_libs1})


  # set(arch_dependant_components "")
  # foreach(arch ${LLVM_TARGETS_TO_BUILD})
  #   list(APPEND arch_dependant_components "${arch}asmpa)
  # endforeach(arch)


  set(CONDITIONAL_COMPONENTS "")
  if(LLVM_USE_PERF)
    list(APPEND CONDITIONAL_COMPONENTS PerfJITEvents)
  endif()
  llvm_map_components_to_libnames(llvm_libs
    core
    support
    jitlink
    orcjit
    # aarch64asmparser
    # aarch64codegen
    # x86asmparser
    # x86codegen
    ExecutionEngine
    ${CONDITIONAL_COMPONENTS}
    ${LLVM_TARGETS_TO_BUILD}
    )

  message(STATUS "-- Libs to ----- link: ${LLVM_TARGETS_TO_BUILD}")
  message(STATUS ${llvm_libs})

  # Serene Setup ===================================

  # We don't want the generated files from table gen
  # to be treated as local since the contain warnings
  include_directories(SYSTEM ${PROJECT_BINARY_DIR}/include)
  #TODO: Can we move it to the libserene cmake somehow?
  include_directories(SYSTEM ${PROJECT_BINARY_DIR}/libserene/include/)

  include_directories(${PROJECT_BINARY_DIR}/include/)

  # Make sure that our source directory is on the current cmake module path so
  # that we can include cmake files from this directory.
  list(INSERT CMAKE_MODULE_PATH 0
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/"
    "${LLVM_COMMON_CMAKE_UTILS}/Modules"
    )

  # Hide all the symbols by default
  if(NOT DEFINED CMAKE_CXX_VISIBILITY_PRESET AND
      NOT DEFINED CMAKE_VISIBILITY_INLINES_HIDDEN)
    set(CMAKE_CXX_VISIBILITY_PRESET hidden)
    set(CMAKE_VISIBILITY_INLINES_HIDDEN YES)
  endif()

  include(tablegen-serene)
  # Create the tools we use to compile Serene
  add_subdirectory(serene-tblgen)

  # The compiled library code is here
  add_subdirectory(libserene)
  # The static library containing builtin special forms and functions
  add_subdirectory(core)
  # Binary tools of the compiler
  add_subdirectory(serenec)
  # add_subdirectory(serene-repl)

  # add_subdirectory(devtools)


  if (SERENE_ENABLE_DOCS)
    add_subdirectory(docs)
  endif()

  install(DIRECTORY ${PROJECT_BINARY_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING
    PATTERN *.h
    PATTERN *.td
    PATTERN "CMake*" EXCLUDE)

endif()

cmake_minimum_required(VERSION 3.16)

# Project name and a few useful settings. Other commands can pick up the results
project(Serene
  VERSION 0.1.0
  DESCRIPTION "Serene language is a modern Lisp."
  LANGUAGES CXX C)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
option(CPP_20_SUPPORT "C++20 Support" OFF)

# Only do these if this is the main project, and not if it is included through add_subdirectory
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)

  ## Settings -----------------------------------------
  # specify the C++ standard
  if (CPP_20_SUPPORT)
    set(CMAKE_CXX_STANDARD 20)
  else()
    set(CMAKE_CXX_STANDARD 17)
  endif()

  set(CMAKE_CXX_STANDARD_REQUIRED True)

  set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
  set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)
  set(BIN_DIR ${CMAKE_SOURCE_DIR}/bin)

  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")
  #set(CMAKE_CXX_LINK_EXECUTABLE "ld.lld")
  #set(CMAKE_C_LINK_EXECUTABLE "ld.lld")
  #set(LLVM_USE_LINKER "ld.lld")
  #set(LLVM_ENABLE_LLD ON)
  set(CMAKE_CXX_CLANG_TIDY clang-tidy)
  # Let's ensure -std=c++xx instead of -std=g++xx
  set(CMAKE_CXX_EXTENSIONS OFF)
  set(CMAKE_CXX_FLAGS_DEBUG
    "${CMAKE_CXX_FLAGS_DEBUG} -g -fno-omit-frame-pointer $ENV{ASAN_FLAG} -fno-builtin-strlen -ggdb -fno-inline ")
  set(CMAKE_LINKER_FLAGS_DEBUG
    "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer $ENV{ASAN_FLAG}")

  set(CMAKE_CXX_FLAGS_RELEASE
    "${CMAKE_CXX_FLAGS_RELEASE} -O3")

  set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/scripts/cmake")
  set(MemoryCheckCommand "valgrind")

  add_compile_options(-fno-rtti)

  configure_file(${INCLUDE_DIR}/config.h.in config.h)

  # Let's nicely support folders in IDEs
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)

  ## Options ------------------------------------------
  option(ENABLE_LOG "Enable logging" OFF)
  option(ENABLE_EXPR_LOG "Enable AExpr logging" OFF)
  option(ENABLE_READER_LOG "Enable reader logging" OFF)
  option(BUILD_TESTING "Enable tests" OFF)


  include(cotire)
  include(FetchContent)

  find_package(LLVM REQUIRED CONFIG)
  find_package(MLIR REQUIRED CONFIG)

  message(STATUS "Found LLVM ${LLVM_PAsCKAGE_VERSION}")
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

  include_directories(${LLVM_INCLUDE_DIRS})
  include_directories(${MLIR_INCLUDE_DIRS})
  include_directories(${PROJECT_SOURCE_DIR}/include)
  include_directories(${PROJECT_BINARY_DIR}/include)
  link_directories(${LLVM_BUILD_LIBRARY_DIR})
  add_definitions(${LLVM_DEFINITIONS})

  include_directories(${LLVM_INCLUDE_DIRS})
  add_definitions(${LLVM_DEFINITIONS})

  llvm_map_components_to_libnames(llvm_libs support core irreader)

  # The compiled library code is here
  add_subdirectory(src/serene)

  # The executable code is here
  add_subdirectory(bin)

  add_subdirectory(include)
  include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
  # Testing only available if this is the main app
  # Emergency override SERENE_CMAKE_BUILD_TESTING provided as well
  if(BUILD_TESTING)
    message("Build the test binary")
    add_subdirectory(src/tests)
  endif()

  if (CMAKE_BUILD_TYPE STREQUAL "Release")
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

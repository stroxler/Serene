cmake_minimum_required(VERSION 3.16)

# Project name and a few useful settings. Other commands can pick up the results
project(Serene
  VERSION 0.1.0
  DESCRIPTION "Serene language is a modern Lisp."
  LANGUAGES CXX)

# Only do these if this is the main project, and not if it is included through add_subdirectory
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)

    # specify the C++ standard
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED True)

    set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
    # set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)
    set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")
    set(CMAKE_CXX_CLANG_TIDY clang-tidy-10)
    # Let's ensure -std=c++xx instead of -std=g++xx
    set(CMAKE_CXX_EXTENSIONS OFF)
    set(CMAKE_CXX_FLAGS_DEBUG
      "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
    set(CMAKE_LINKER_FLAGS_DEBUG
      "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")

    set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

    # Let's nicely support folders in IDEs
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)

    # Testing only available if this is the main app
    # Note this needs to be done in the main CMakeLists
    # since it calls enable_testing, which must be in the
    # main CMakeLists.
    include(CTest)

    # Docs only available if this is the main app
    find_package(Doxygen)
    if(Doxygen_FOUND)
	add_subdirectory(docs)
    else()
	message(STATUS "Doxygen not found, not building docs")
    endif()
endif()

include(cotire)
include(FetchContent)

find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

llvm_map_components_to_libnames(llvm_libs support core irreader)

# Formatting library
# FetchContent_Declare(
#   fmtlib
#   GIT_REPOSITORY https://github.com/fmtlib/fmt.git
#   GIT_TAG        5.3.0
#   )
# FetchContent_MakeAvailable(fmtlib)


# The compiled library code is here
add_subdirectory(src)

# The executable code is here
add_subdirectory(bin)

# Testing only available if this is the main app
# Emergency override SERENE_CMAKE_BUILD_TESTING provided as well
if((CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME OR SERENE_CMAKE_BUILD_TESTING) AND BUILD_TESTING)
    add_subdirectory(tests)
endif()

#configure_file(${INCLUDE_DIR}/config.h.in config.h)


# target_link_libraries(serene ${llvm_libs})

# target_include_directories(serene SYSTEM PRIVATE $ENV{INCLUDE})
# target_include_directories(serene PRIVATE ${INCLUDE_DIR})
# target_include_directories(serene PUBLIC ${PROJECT_BINARY_DIR})
# install(TARGETS serene DESTINATION bin)
# install(FILES "${PROJECT_BINARY_DIR}/serene/config.h"
#   DESTINATION include
#   )

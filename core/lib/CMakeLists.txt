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

#TODO: To support MacOS look into cmake's public headers
# https://cmake.org/cmake/help/latest/prop_tgt/PUBLIC_HEADER.html

# Prevent any future RPATH issue on Posix
if(NOT APPLE)
  set(CMAKE_INSTALL_RPATH $ORIGIN)
endif()

add_library(core STATIC
  reader.cpp)

# Create an ALIAS target. This way if we mess up the name
# there will be an cmake error inseat of a linker error which is harder
# to understand. So any binary that wants to use serene has to
# use `Serene::core` alias instead
add_library(Serene::core ALIAS core)

set_target_properties(core PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
  CMAKE_PKG_NAME SereneCore
  # Warn on unused libs
  LINK_WHAT_YOU_USE TRUE
  CXX_INCLUDE_WHAT_YOU_USE "${iwyu_path}"
  C_INCLUDE_WHAT_YOU_USE "${iwyu_path}"
  # LTO support
  INTERPROCEDURAL_OPTIMIZATION TRUE)

if(SERENE_ENABLE_TIDY)
  set_target_properties(core PROPERTIES CXX_CLANG_TIDY ${CLANG_TIDY_PATH})
endif()

if (CPP_20_SUPPORT)
  target_compile_features(core PUBLIC cxx_std_20)
else()
  target_compile_features(core PUBLIC cxx_std_17)
endif()

#target_compile_options(core PUBLIC ${CMAKE_C_FLAGS} -emit-llvm)
# We need this directory, and users of our library will need it too
target_include_directories(core PUBLIC "$<BUILD_INTERFACE:${INCLUDE_DIR}>")
target_include_directories(core PUBLIC "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>")

# Generate the export.h
include(GenerateExportHeader)

generate_export_header(core EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/include/serene/core/export.h)


target_link_libraries(core PRIVATE)

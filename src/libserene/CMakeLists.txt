# Serene Programming Language
#
# Copyright (c) 2019-2021 Sameer Rahmani <lxsameer@gnu.org>
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

# Hide all the symbols by default
if(NOT DEFINED CMAKE_CXX_VISIBILITY_PRESET AND
    NOT DEFINED CMAKE_VISIBILITY_INLINES_HIDDEN)
  set(CMAKE_CXX_VISIBILITY_PRESET hidden)
  set(CMAKE_VISIBILITY_INLINES_HIDDEN YES)
endif()

# Prevent any future RPATH issue on Posix
if(NOT APPLE)
  set(CMAKE_INSTALL_RPATH $ORIGIN)
endif()

add_library(serene
  exprs/symbol.cpp
  exprs/list.cpp
  exprs/number.cpp
  exprs/expression.cpp
  exprs/def.cpp
  exprs/fn.cpp
  exprs/call.cpp

  serene.cpp
  context.cpp
  namespace.cpp
  jit.cpp
  source_mgr.cpp
  diagnostics.cpp

  # Reader
  reader/reader.cpp
  reader/semantics.cpp

  # Errors
  errors/error.cpp

  # IR
  slir/slir.cpp
  slir/dialect.cpp
  slir/value_op.cpp
  slir/generatable.cpp
  slir/utils.cpp
  slir/ops.cpp
  passes/slir_lowering.cpp
  passes/to_llvm_dialect.cpp
  )

# Create an ALIAS target. This way if we mess up the name
# there will be an cmake error inseat of a linker error which is harder
# to understand. So any binary that wants to use serene has to
# use `Serene::lib` alias instead
add_library(Serene::lib ALIAS serene)

set_target_properties(serene PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
  # Warn on unused libs
  LINK_WHAT_YOU_USE TRUE
  # LTO support
  INTERPROCEDURAL_OPTIMIZATION TRUE)

if(SERENE_ENABLE_TIDY)
  set_target_properties(serene PROPERTIES CXX_CLANG_TIDY ${CLANG_TIDY_PATH})
endif()

# Do we need to build serene as a shared lib? default is "yes"
if(SERENE_SHARED_LIB)

  # We need to use libsan as a shared lib on debug mode. The
  # target executable has to be built with `-fsanitize=address`
  # as well and it has to run with:
  # LD_PRELOAD=$(clang -print-file-name=libclang_rt.asan-x86_64.so)
  target_compile_options(serene PRIVATE
    $<$<CONFIG:DEBUG>:-shared-libsan>
    )
  target_link_options(
    serene PRIVATE
    $<$<CONFIG:DEBUG>:-shared-libsan>
    )
endif()


if (CPP_20_SUPPORT)
  target_compile_features(serene PUBLIC cxx_std_20)
else()
  target_compile_features(serene PUBLIC cxx_std_17)
endif()

# Generate the tablegen ODS files before this target
add_dependencies(serene SereneDialectGen)

# We need this directory, and users of our library will need it too
target_include_directories(serene PUBLIC "$<BUILD_INTERFACE:${INCLUDE_DIR}>")
target_include_directories(serene PUBLIC "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>")

# Generate the export.h
include(GenerateExportHeader)
generate_export_header(serene EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/include/serene/export.h)
target_compile_definitions(
  serene PUBLIC "$<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:SERENE_STATIC_DEFINE>")

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)


if(SERENE_SHOW_MLIR_DIALECTS)
  message(STATUS "MLIR Dialects to choose from:")
  foreach(lib ${dialect_libs})
    message(STATUS "\t${lib}")
  endforeach()
endif()


if(SERENE_SHOW_MLIR_TRANSFORMERS)
  message(STATUS "MLIR Dialects transformers to choose from:")
  foreach(lib ${conversion_libs})
    message(STATUS "\t${lib}")
  endforeach()
endif()

if(SERENE_SHOW_LLVM_LIBS)
  execute_process(COMMAND llvm-config --libs all
    OUTPUT_VARIABLE SERENE_LLVM_LIBS)
  message(STATUS "LLVM libs available:\n ${SERENE_LLVM_LIBS}")
endif()


set(serene_lib_dialects_in_use
  MLIRStandard)

set(serene_lib_transformers_in_use
  MLIRStandardToLLVM)

target_link_libraries(serene PRIVATE
  MLIRIR
  MLIRPass
  MLIRTransforms

  ${serene_lib_dialects_in_use}
  ${serene_lib_transformers_in_use}

  #TODO: Remove this lib, we're just using one func
  MLIRExecutionEngine

  # LLVM's JIT lib
  LLVMExecutionEngine
  LLVMOrcJIT
  MLIRLLVMToLLVMIRTranslation

  LLVMX86AsmParser

  ${llvm_libs})

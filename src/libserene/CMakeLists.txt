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

option(SERENE_SHOW_MLIR_DIALECTS "Print out a list of MLIR dialect libs" OFF)
option(SERENE_SHOW_MLIR_TRANSFORMERS "Print out a list of MLIR dialect transformer libs" OFF)
option(SERENE_SHOW_LLVM_LIBS "Print all the llvm libs available" OFF)


add_library(libserene OBJECT
  exprs/symbol.cpp
  exprs/list.cpp
  exprs/number.cpp
  exprs/expression.cpp
  exprs/def.cpp
  exprs/fn.cpp
  exprs/call.cpp

  context.cpp
  namespace.cpp
  jit.cpp
  source_mgr.cpp
  diagnostics.cpp

  # Reader
  reader/reader.cpp
  reader/location.cpp
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

add_dependencies(libserene SereneDialectGen)

if (CPP_20_SUPPORT)
  target_compile_features(libserene PUBLIC cxx_std_20)
else()
  target_compile_features(libserene PUBLIC cxx_std_17)
endif()


# We need this directory, and users of our library will need it too
target_include_directories(libserene PUBLIC ${INCLUDE_DIR})
target_include_directories(libserene PUBLIC ${PROJECT_BINARY_DIR})

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

set(static_deps
  MLIRIR
  MLIRPass
  MLIRTransforms

  ${serene_lib_dialects_in_use}
  ${serene_lib_transformers_in_use}

  # LLVM's JIT lib
  LLVMExecutionEngine
  LLVMOrcJIT
  MLIRLLVMToLLVMIRTranslation

  #TODO: Remove this lib, we're just using one func
  MLIRExecutionEngine

  ${llvm_libs})

set_target_properties(libserene PROPERTIES StaticDeps "${static_deps}")

# Create an ALIAS target. This way if we mess up the name
# there will be an cmake error inseat of a linker error which is harder
# to understand. So any binary that wants to use libserene has to
# use `Serene::lib` alias instead
add_library(Serene::lib ALIAS libserene)

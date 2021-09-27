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

# Make an automatic library - will be static or dynamic based on user setting
add_library(serene
  exprs/symbol.cpp
  exprs/list.cpp
  exprs/number.cpp
  exprs/expression.cpp
  exprs/def.cpp
  exprs/fn.cpp
  exprs/call.cpp

  context.cpp
  serene.cpp
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
#${HEADER_LIST}

# Make sure to generate files related to the dialects first
add_dependencies(serene SereneDialectGen)

if (CPP_20_SUPPORT)
  target_compile_features(serene PUBLIC cxx_std_20)
else()
  target_compile_features(serene PUBLIC cxx_std_17)
endif()


# We need this directory, and users of our library will need it too
target_include_directories(serene PUBLIC ${INCLUDE_DIR})
target_include_directories(serene PUBLIC ${PROJECT_BINARY_DIR})


source_group(TREE "${INCLUDE_DIR}" PREFIX "Header Files" FILES ${HEADER_LIST})
#target_precompile_headers(serene PRIVATE ${HEADER_LIST})
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
# This depends on (header only) boost
target_link_libraries(serene
  PUBLIC
    ${dialect_libs}
    ${conversion_libs}
    MLIRAnalysis
    MLIRCallInterfaces
    MLIRCastInterfaces
    MLIRExecutionEngine
    MLIRIR
    MLIRLLVMToLLVMIRTranslation
    MLIRParser
    MLIRPass
    MLIRSideEffectInterfaces
    MLIRTargetLLVMIRExport
    MLIRTransforms
    ${llvm_libs})

# target_precompile_headers(serene
#   PRIVATE
#   <serene_precompiles.h>
# )

set(HEADER_LIST
  "${INCLUDE_DIR}/serene/serene.h"
  "${INCLUDE_DIR}/serene/utils.h"
  "${INCLUDE_DIR}/serene/context.h"
  "${INCLUDE_DIR}/serene/environment.h"
  "${INCLUDE_DIR}/serene/traits.h"

  "${INCLUDE_DIR}/serene/exprs/expression.h"
  "${INCLUDE_DIR}/serene/exprs/symbol.h"
  "${INCLUDE_DIR}/serene/exprs/list.h"
  "${INCLUDE_DIR}/serene/exprs/number.h"
  "${INCLUDE_DIR}/serene/exprs/def.h"
  "${INCLUDE_DIR}/serene/exprs/fn.h"
  "${INCLUDE_DIR}/serene/exprs/traits.h"
  "${INCLUDE_DIR}/serene/exprs/call.h"

  # Reader
  "${INCLUDE_DIR}/serene/reader/reader.h"
  "${INCLUDE_DIR}/serene/reader/location.h"
  "${INCLUDE_DIR}/serene/reader/errors.h"
  "${INCLUDE_DIR}/serene/reader/semantics.h"
  "${INCLUDE_DIR}/serene/reader/traits.h"

  "${INCLUDE_DIR}/serene/errors.h"
  "${INCLUDE_DIR}/serene/errors/error.h"
  "${INCLUDE_DIR}/serene/errors/errc.h"
  "${INCLUDE_DIR}/serene/errors/constants.h"
  "${INCLUDE_DIR}/serene/errors/traits.h"


  "${INCLUDE_DIR}/serene/slir/slir.h"
  "${INCLUDE_DIR}/serene/slir/dialect.h"
  "${INCLUDE_DIR}/serene/slir/generatable.h"
  "${INCLUDE_DIR}/serene/slir/utils.h"
  "${INCLUDE_DIR}/serene/namespace.h"

  "${INCLUDE_DIR}/serene/passes.h")

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

  # Reader
  reader/reader.cpp
  reader/location.cpp
  reader/errors.cpp
  reader/semantics.cpp

  # Errors
  errors/error.cpp

  # IR
  slir/dialect.cpp
  slir/value_op.cpp
  slir/generatable.cpp
  slir/utils.cpp
  slir/ops.cpp
  passes/slir_lowering.cpp
  passes/to_llvm_dialect.cpp
  ${HEADER_LIST})


# Make sure to generate files related to the dialects first
add_dependencies(serene SereneDialectGen)


if (CPP_20_SUPPORT)
  target_compile_features(serene PUBLIC cxx_std_20)
else()
  target_compile_features(serene PUBLIC cxx_std_17)
endif()


# We need this directory, and users of our library will need it too
target_include_directories(serene PRIVATE ${INCLUDE_DIR})
target_include_directories(serene PUBLIC ${PROJECT_BINARY_DIR})


source_group(TREE "${INCLUDE_DIR}" PREFIX "Header Files" FILES ${HEADER_LIST})
#target_precompile_headers(serene PRIVATE ${HEADER_LIST})
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
# This depends on (header only) boost
target_link_libraries(serene
  PRIVATE
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
    ${llvm_libs}
    )

set(HEADER_LIST
  "${INCLUDE_DIR}/serene/serene.h"

  "${INCLUDE_DIR}/serene/exprs/expression.h"
  "${INCLUDE_DIR}/serene/exprs/symbol.h"
  "${INCLUDE_DIR}/serene/exprs/list.h"



  "${INCLUDE_DIR}/serene/expr.hpp"

  "${INCLUDE_DIR}/serene/number.hpp"
  "${INCLUDE_DIR}/serene/symbol.hpp"
  "${INCLUDE_DIR}/serene/list.hpp"
  "${INCLUDE_DIR}/serene/error.hpp"

  "${INCLUDE_DIR}/serene/state.hpp"
  # Reader
  "${INCLUDE_DIR}/serene/reader/reader.hpp"
  "${INCLUDE_DIR}/serene/reader/location.hpp"


  "${INCLUDE_DIR}/serene/sir/sir.hpp"
  "${INCLUDE_DIR}/serene/sir/dialect.hpp"
  "${INCLUDE_DIR}/serene/sir/generator.hpp"
  "${INCLUDE_DIR}/serene/namespace.hpp")

# Make an automatic library - will be static or dynamic based on user setting
add_library(serene
  exprs/symbol.cpp
  exprs/list.cpp

  serene.cpp
  symbol.cpp
  list.cpp
  namespace.cpp
  state.cpp
  error.cpp
  number.cpp

  # Reader
  reader/reader.cpp
  reader/location.cpp

  # IR
  sir/sir.cpp
  sir/dialect.cpp
  sir/value_op.cpp
  sir/generator.cpp
  ${HEADER_LIST})


# Make sure to generate files related to the dialects first
add_dependencies(serene SereneDialectGen)

# We need this directory, and users of our library will need it too
target_compile_features(serene PUBLIC cxx_std_14)
target_include_directories(serene PRIVATE ${INCLUDE_DIR})
target_include_directories(serene PUBLIC ${PROJECT_BINARY_DIR})

# This depends on (header only) boost
target_link_libraries(serene ${llvm_libs} fmt::fmt)
source_group(TREE "${INCLUDE_DIR}" PREFIX "Header Files" FILES ${HEADER_LIST})
#target_precompile_headers(serene PRIVATE ${HEADER_LIST})

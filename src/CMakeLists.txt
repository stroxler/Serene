# Optionally glob, but only for CMake 3.12 or later:
#file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${INCLUDE_DIR}/**/*.hpp")
#file(GLOB SOURCES_LIST CONFIGURE_DEPENDS "${SOURCE_DIR}/**/*.cpp")
set(HEADER_LIST
  "${INCLUDE_DIR}/serene/expr.hpp"
  "${INCLUDE_DIR}/serene/serene.hpp"
  "${INCLUDE_DIR}/serene/state.hpp"
  "${INCLUDE_DIR}/serene/sir/sir.hpp"
  "${INCLUDE_DIR}/serene/sir/dialect.hpp"
    "${INCLUDE_DIR}/serene/sir/generator.hpp"
  "${INCLUDE_DIR}/serene/namespace.hpp")

# Make an automatic library - will be static or dynamic based on user setting
add_library(lserene
  serene.cpp
  reader.cpp
  symbol.cpp
  list.cpp
  namespace.cpp
  state.cpp
  sir/sir.cpp
  sir/dialect.cpp
  sir/value_op.cpp
  sir/generator.cpp
  ${HEADER_LIST})

# Make sure to generate files related to the dialects first
add_dependencies(lserene SereneDialectGen)

# We need this directory, and users of our library will need it too
target_compile_features(lserene PUBLIC cxx_std_14)
target_include_directories(lserene PRIVATE ${INCLUDE_DIR})
target_include_directories(lserene PUBLIC ${PROJECT_BINARY_DIR})

# This depends on (header only) boost
target_link_libraries(lserene ${llvm_libs} fmt::fmt)
source_group(TREE "${INCLUDE_DIR}" PREFIX "Header Files" FILES ${HEADER_LIST})

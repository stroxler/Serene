# Optionally glob, but only for CMake 3.12 or later:
#file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${INCLUDE_DIR}/**/*.hpp")
#file(GLOB SOURCES_LIST CONFIGURE_DEPENDS "${SOURCE_DIR}/**/*.cpp")
set(HEADER_LIST
  "${INCLUDE_DIR}/serene/expr.hpp"
  "${INCLUDE_DIR}/serene/serene.hpp")
# Make an automatic library - will be static or dynamic based on user setting
add_library(lserene
  serene.cpp
  reader.cpp
  symbol.cpp
  list.cpp
  ${HEADER_LIST})


# We need this directory, and users of our library will need it too
target_compile_features(lserene PUBLIC cxx_std_20)
target_include_directories(lserene PRIVATE ${INCLUDE_DIR})
target_include_directories(lserene PUBLIC ${PROJECT_BINARY_DIR})

# This depends on (header only) boost
target_link_libraries(lserene ${llvm_libs} fmt::fmt)
source_group(TREE "${INCLUDE_DIR}" PREFIX "Header Files" FILES ${HEADER_LIST})

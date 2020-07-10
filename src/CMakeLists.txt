
# Note that headers are optional, and do not affect add_library, but they will not
# show up in IDEs unless they are listed in add_library.

# Optionally glob, but only for CMake 3.12 or later:
# file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${ModernCMakeExample_SOURCE_DIR}/include/modern/*.hpp")
set(HEADER_LIST "${INCLUDE_DIR}/serene/serene.hpp")

# Make an automatic library - will be static or dynamic based on user setting
add_library(libserene libserene.cpp ${HEADER_LIST})

# We need this directory, and users of our library will need it too
target_include_directories(libserene PUBLIC "${INCLUDE_DIR}")
target_compile_features(libserene PUBLIC cxx_std_20)

# This depends on (header only) boost
target_link_libraries(libserene ${llvm_libs})


source_group(TREE "${INCLUDE_DIR}" PREFIX "Header Files" FILES ${HEADER_LIST})

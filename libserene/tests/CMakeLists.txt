# Catch2 should be installed system wide
find_package(Catch2 REQUIRED)

# Tests need to be added as executables first
add_executable(tests serenetests.cpp)
add_dependencies(tests SereneDialectGen)
add_dependencies(tests serene)
target_link_libraries(tests PRIVATE
  serene
  ${llvm_libs}
  MLIRAnalysis
  MLIRIR
  MLIRParser
  MLIRSideEffectInterfaces
  MLIRTransforms
  )

target_compile_features(tests PRIVATE cxx_std_17)

# Should be linked to the main library, as well as the Catch2 testing library
target_link_libraries(tests PUBLIC serene Catch2::Catch2)
# target_include_directories(serene SYSTEM PRIVATE $ENV{INCLUDE})
# target_include_directories(serene PUBLIC ${INCLUDE_DIR})

# If you register a test, then ctest and make test will run it.
# You can also run examples and check the output, as well.
# add_test(NAME testlibtest serene testlib) # Command can be a target
include(CTest)
include(Catch)
catch_discover_tests(tests)

# Catch2 should be installed system wide
#find_package(Catch2 3 REQUIRED)

# Tests need to be added as executables first
add_executable(libsereneTests serenetests.cpp)

add_dependencies(libsereneTests SereneDialectGen)
add_dependencies(libsereneTests serene)

target_link_libraries(libsereneTests PRIVATE
  serene
  ${llvm_libs}
  MLIRAnalysis
  MLIRIR
  MLIRParser
  MLIRSideEffectInterfaces
  MLIRTransforms

  Catch2::Catch2WithMain
  )

target_compile_features(libsereneTests PRIVATE cxx_std_17)

# target_include_directories(serene SYSTEM PRIVATE $ENV{INCLUDE})
# target_include_directories(serene PUBLIC ${INCLUDE_DIR})

include(CTest)
include(Catch)
catch_discover_tests(libsereneTests)

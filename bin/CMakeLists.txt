add_executable(serene serene.cpp)
# Make sure to generate files related to the dialects first
add_dependencies(serene SereneDialectGen)

target_compile_features(serene PRIVATE cxx_std_14)

target_link_libraries(serene PRIVATE
  lserene
  ${llvm_libs}
  fmt::fmt
  MLIRAnalysis
  MLIRIR
  MLIRParser
  MLIRSideEffectInterfaces
  MLIRTransforms
  )


target_include_directories(serene SYSTEM PRIVATE $ENV{INCLUDE})
target_include_directories(serene PRIVATE ${INCLUDE_DIR})

install(TARGETS serene DESTINATION bin)
install(FILES "${PROJECT_BINARY_DIR}/config.h" DESTINATION include)
cotire(serene)

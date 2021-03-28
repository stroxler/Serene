add_executable(serenec serene.cpp)
# Make sure to generate files related to the dialects first
add_dependencies(serenec SereneDialectGen)

target_compile_features(serenec PRIVATE cxx_std_14)

target_link_libraries(serenec PRIVATE
  serene
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

install(TARGETS serenec DESTINATION bin)
install(FILES "${PROJECT_BINARY_DIR}/config.h" DESTINATION include)
cotire(serenec)

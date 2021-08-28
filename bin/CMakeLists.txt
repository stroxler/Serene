add_executable(serenec serene.cpp)

# Make sure to generate files related to the dialects first
#add_dependencies(serenec SereneDialectGen)

if (CPP_20_SUPPORT)
  target_compile_features(serenec PRIVATE cxx_std_20)
else()
  target_compile_features(serenec PRIVATE cxx_std_17)
endif()




target_link_libraries(serenec PRIVATE
  serene
  ${llvm_libs}
  MLIRAnalysis
  MLIRIR
  MLIRParser
  MLIRSideEffectInterfaces
  MLIRTransforms
  LLVMX86AsmParser
  LLVMOption
  #clangTooling
  clangDriver
  clangBasic
  clangdSupport
  clangFrontend
  clangEdit
  clangLex
  )


target_include_directories(serene SYSTEM PRIVATE $ENV{INCLUDE})
target_include_directories(serene PRIVATE ${INCLUDE_DIR})

install(TARGETS serenec DESTINATION bin)
install(FILES "${PROJECT_BINARY_DIR}/config.h" DESTINATION include)

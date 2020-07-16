add_executable(serene serene.cpp)
target_compile_features(serene PRIVATE cxx_std_20)

target_link_libraries(serene PRIVATE
  lserene
  ${llvm_libs}
  fmt::fmt
  )

target_include_directories(serene SYSTEM PRIVATE $ENV{INCLUDE})
target_include_directories(serene PRIVATE ${INCLUDE_DIR})
target_include_directories(serene PUBLIC ${PROJECT_BINARY_DIR})
install(TARGETS serene DESTINATION bin)
install(FILES "${PROJECT_BINARY_DIR}/serene/config.h"
  DESTINATION include
  )
cotire(serene)

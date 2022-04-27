set(LLVM_TARGET_DEFINITIONS passes.td)

mlir_tablegen(passes.h.inc -gen-pass-decls)

if(SERENE_ENABLE_DOCS)
  mlir_tablegen(passes.md.inc -gen-pass-doc)
endif()

add_public_tablegen_target(SerenePassGen)

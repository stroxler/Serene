set(LLVM_TARGET_DEFINITIONS dialect.td)
mlir_tablegen(ops.h.inc -gen-op-decls)
mlir_tablegen(ops.cpp.inc -gen-op-defs)

mlir_tablegen(attrs.h.inc -gen-attrdef-decls)
mlir_tablegen(attrs.cpp.inc -gen-attrdef-defs)
mlir_tablegen(types.h.inc -gen-typedef-decls )
mlir_tablegen(types.cpp.inc -gen-typedef-defs)

mlir_tablegen(dialect.h.inc -gen-dialect-decls)
mlir_tablegen(dialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(SereneDialectGen)

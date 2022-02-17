set(LLVM_TARGET_DEFINITIONS errors.td)

serene_tablegen(errs.h.inc -errors-backend)

add_public_tablegen_target(SereneErrorGen)

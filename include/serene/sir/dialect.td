#ifndef SERENE_DIALECT
#define SERENE_DIALECT

include "mlir/IR/OpBase.td"
//include "mlir/Interfaces/SideEffectInterfaces.td"

def Serene_Dialect : Dialect {
    let name = "serene";
    let cppNamespace = "::serene::sir";
}

#endif // SERENE_DIALECT

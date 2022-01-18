#ifndef SERENE_DIALECT
#define SERENE_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/Interfaces/VectorInterfaces.td"


// Dialect definition. It will directly generate the SereneDialect class
def Serene_Dialect : Dialect {
    let name = "serene";
    let cppNamespace = "::serene::slir";
    let summary = "Primary IR of serene language.";
    let description = [{
        This dialect tries to map the special forms of a lisp into
        IR level operations.
    }];
    let useDefaultTypePrinterParser = 1;
}

// Base class for Serene dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class Serene_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Serene_Dialect, mnemonic, traits>;


// All of the types will extend this class.
class Serene_Type<string name> : TypeDef<Serene_Dialect, name> { }


def SereneSymbol : Serene_Type<"Symbol"> {
    let mnemonic = "symbol";

    let summary = "A Lisp symbol type";

    let description = [{
        A Lisp symbol type
    }];

    let parameters = (ins "std::string":$ns, "std::string":$name);
    let assemblyFormat = "`<` $ns `,` $name `>`";
}




def ValueOp: Serene_Op<"value"> {

    let summary = "This operation represent a value";
    let description = [{
        ValueOp
    }];

    let arguments = (ins I64Attr:$value);
    let results = (outs I64);

    //let verifier = [{ return serene::sir::verify(*this); }];

    let builders = [
        OpBuilder<(ins "int":$value), [{
        // Build from fix 64 bit int
        build(odsBuilder, odsState, odsBuilder.getI64Type(), (uint64_t) value);
        }]>,

    ];
}

// Def
def DefOp: Serene_Op<"def"> {

    let summary = "This operation defines a global binding";
    let description = [{
        DefOp
    }];

    let arguments = (ins TypeAttr:$type,
                         StrAttr:$name,
                         OptionalAttr<StrAttr>:$sym_visibility);

    let results = (outs SereneSymbol);
}


// TODO: Add the FunctionLike trait here and include its header file in dialect.h
def FnOp: Serene_Op<"fn", [
  AffineScope, AutomaticAllocationScope,
  IsolatedFromAbove,
]> {

    let summary = "This operation is just a place holder for a function";
    let description = [{
        A place holder for an anonymous function. For example consider an expression
        like `(def a (fn (x) x))`, in this case we don't immediately create an anonymous
        function since we need to set the name and create the function later.
    }];

    let arguments = (ins StrAttr:$name,
                         DictionaryAttr:$args,
                         OptionalAttr<StrAttr>:$sym_visibility);

    let regions = (region AnyRegion:$body);
    let results = (outs I64);

}

def ReturnOp: Serene_Op<"return", [NoSideEffect, HasParent<"FnOp">,
                                   ReturnLike, Terminator]> {

    let summary = "This operation marks the return value of a function";
    let description = [{
        ReturnOp
    }];

    let arguments = (ins AnyType:$operand);
    let assemblyFormat =
          [{ attr-dict $operand  `:` type($operand) }];
}

#endif // SERENE_DIALECT

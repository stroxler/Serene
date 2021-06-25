#ifndef SERENE_DIALECT
#define SERENE_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"


// Dialect definition. It will directly generate the SereneDialect class
def Serene_Dialect : Dialect {
    let name = "serene";
    let cppNamespace = "::serene::slir";
    let summary = "Primary IR of serene language.";
    let description = [{
        This dialect tries to map the special forms of a lisp into
        IR level operations.
    }];
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


// def SymbolType : Serene_Type<"Symbol"> {

//   let mnemonic = "symbol";

//   let summary = "A typical Lisp symbol";

//   let description = [{
//       A symbol is just a name and nothing more. Just a name
//       to give to a value or to use it as it is.
//   }];

//   //  let cppNamespace = "::serene::sir";
//   let parameters = (ins "std::string":$name);

//   // We define the printer inline.
//   let printer = [{
//     $_printer << "Symbol<" << getImpl()->name << ">";
//   }];

//   // The parser is defined here also.
//   let parser = [{
//     if ($_parser.parseLess())
//       return Type();

//     std::string name;
//     if ($_parser.parseInteger(name))
//       return Type();

//     return get($_ctxt, name);
//   }];
// }

def ValueOp: Serene_Op<"value"> {

    let summary = "This operation represent a value";
    let description = [{
        some description
    }];

    let arguments = (ins I64Attr:$value);
    let results = (outs I64);

    // let verifier = [{ return serene::sir::verify(*this); }];

    let builders = [
        OpBuilder<(ins "int":$value), [{
        // Build from fix 64 bit int
        build(odsBuilder, odsState, odsBuilder.getI64Type(), (uint64_t) value);
        }]>,

    ];
}

def FnOp: Serene_Op<"fn", [
  AffineScope, AutomaticAllocationScope,
  IsolatedFromAbove
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
    let results = (outs NoneType);

  // let builders = [
  //     OpBuilder<(ins
  //       "llvm::StringRef":$name, "mlir::FunctionType":$type,
  //       CArg<"llvm::ArrayRef<mlir::NamedAttribute>", "{}">:$attrs,
  //       CArg<"llvm::ArrayRef<mlir::DictionaryAttr>", "{}">:$argAttrs)
  // >];

  // let extraClassDeclaration = [{
  //   static FnOp create(mlir::Location location, llvm::StringRef name, mlir::FunctionType type,
  //                      llvm::ArrayRef<mlir::NamedAttribute> attrs = {});
    // static FnOp create(mlir::Location location, llvm::StringRef name, mlir::FunctionType type,
    //                    mlir::Operation::dialect_attr_range attrs);
    // static FnOp create(mlir::Location location, llvm::StringRef name, mlir::FunctionType type,
    //                    llvm::ArrayRef<mlir::NamedAttribute> attrs,
    //                    llvm::ArrayRef<mlir::DictionaryAttr> argAttrs);
    // }];
}
#endif // SERENE_DIALECT

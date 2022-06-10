/*
 * Serene Programming Language
 *
 * Copyright (c) 2019-2022 Sameer Rahmani <lxsameer@gnu.org>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 2.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef SERENE_DIALECT_OPS_TD
#define SERENE_DIALECT_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/OpAsmInterface.td"

// Base class for Serene dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class Serene_Op<string mnemonic, list<Trait> traits = []> :
    Op<Serene_Dialect, mnemonic, traits>;


// serene.value
def ValueOp : Serene_Op<"value", [
    ConstantLike, NoSideEffect,
    TypesMatchWith<
    "result and attribute have the same type",
    "value", "result", "$_self">]> {

  let summary = "This operation represent a compile time value";
  let description = [{
    The `value` operation produces an SSA value equal to value
    specified by an attribute. This is the way Serene marks constant
    compile time values.

    Example:

    ```
    // Integer constant
    %1 = serene.value 42 : i32

    // Equivalent generic form
    %1 = "serene.value"() {value = 42 : i32} : () -> i32
    ```
  }];

  let arguments = (ins AnyAttr:$value);
  let results = (outs AnyType:$result);

  let builders = [
    OpBuilder<(ins "mlir::Attribute":$value),
    [{ build($_builder, $_state, value.getType(), value); }]>,
    OpBuilder<(ins "mlir::Attribute":$value, "mlir::Type":$type),
    [{ build($_builder, $_state, type, value); }]>,
  ];

  let extraClassDeclaration = [{
    /// Whether the constant op can be constructed with a particular value and
    /// type.
    static bool isBuildableWith(mlir::Attribute value, mlir::Type type);
  }];

  // Need to define the ::fold() method to make value op foldable
  // let hasFolder = 1;
  let assemblyFormat = "attr-dict $value";
  let hasFolder = 1;
}

// serene.intern
def InternOp : Serene_Op<"intern", [
    NoSideEffect
]> {

  let summary = "This operation is the runtime contructor for symbol type";
  let description = [{
    The `intern` operation produces an SSA value that holds a value of type
    symbol at runtime.

    Example:

    ```

    %ns = serene.string "some.ns"
    %name = serene.string "symbol_name"
    %1 = serene.intern  %ns %name

    // Equivalent generic form
    %1 = "serene.symbol"(%ns, %name){} : (!serene.string, !serene.string) -> !serene.ptr<serene.symbol>
    ```
  }];

  let arguments = (ins StringType:$ns, StringType:$name);
  let results = (outs Ptr<SymbolType>:$result);

  let assemblyFormat = "attr-dict $ns $name";
}

// serene.symbol
def SymbolOp : Serene_Op<"symbol", [
    NoSideEffect, ConstantLike,
]> {

  let summary = "This operation is the compile time contructor for symbol type";
  let description = [{
    The `symbol` operation produces an SSA value that holds a value of type
    symbol at compile time.

    Example:

    ```

    %1 = serene.symbol "some.ns" "symbol_name"

    // Equivalent generic form
    %1 = "serene.symbol"() {ns = "some.ns", name = "symbol_name"} : () -> i32
    ```
  }];

  let arguments = (ins SymbolAttr:$value);
  let results = (outs Ptr<SymbolType>:$result);

  let assemblyFormat = "attr-dict $value";
  let hasFolder = 1;
}

// serene.convert
def ConvertOp : Serene_Op<"convert", [
    NoSideEffect
]> {

  let summary = "This operation converts a symbol to the equivelant llvm type";
  let description = [{
    This operation converts a symbol to the equivelant llvm type
  }];

  let arguments = (ins AnyType:$value);

  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    $value attr-dict `:` functional-type($value, results)
  }];

}

// ============================================================================
// Define Family
// ============================================================================

// serene.define
def DefineOp: Serene_Op<"define", [Symbol]> {
    let summary = "This operation defines a global binding in the current namespace";
    let description = [{
        `define` defines a global binding in the current namespace. It always return a
        symbol type.

        Examples:

        ```mlir
        %foo = "serene.define"(%0){name = "foo"}: (i64) -> !serene.symbol

        // compact form
        %bar = serene.define "bar", %0 : i64
        ```
    }];

    let arguments = (ins SymbolNameAttr:$sym_name, AnyType:$value,
                         OptionalAttr<BoolAttr>:$is_top_level,
                         OptionalAttr<StrAttr>:$sym_visibility);

    //let results = (outs SereneSymbol);
    let assemblyFormat = "attr-dict $sym_name `,` $value `:` type($value)";
}

// serene.define_constant
def DefineConstantOp: Serene_Op<"define_constant", [Symbol]> {
    let summary = "This operation defines a constant global binding in the current namespace";
    let description = [{
        `define` defines a constant global binding in the current namespace. It always return a
        symbol type.

        Examples:

        ```mlir
        %foo = "serene.define_constant"(%0){name = "foo"}: (i64) -> !serene.symbol

        // compact form
        %bar = serene.define_constant "bar", %0 : i64
        ```
    }];

    let arguments = (ins SymbolNameAttr:$sym_name, AnyAttr:$value,
                         OptionalAttr<BoolAttr>:$is_top_level,
                         OptionalAttr<StrAttr>:$sym_visibility);

    //    let results = (outs SereneSymbol);
    let assemblyFormat = "attr-dict $sym_name $value";
}

// serene.set_value
def SetValueOp: Serene_Op<"set_value", []> {

    let summary = "Sets the value of a SLIRS global varible.";
    let description = [{
        This operation sets the value of a SLIR global variable. This operation
        is NOT threadsafe.

    }];

    let arguments = (ins SymbolRefAttr:$var,
                         AnyType:$value);

    let assemblyFormat = "attr-dict $var `,` $value `:` type($value)";
}

// serene.fn
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
                         TypeAttr:$return_type,
                         OptionalAttr<StrAttr>:$sym_visibility);

    let regions = (region VariadicRegion<AnyRegion>:$bodies);
    let results = (outs FnType);
}


// serene.ret
def ReturnOp: Serene_Op<"ret", [
  NoSideEffect, HasParent<"FnOp">,
  ReturnLike, Terminator
  //  MemRefsNormalizable
]> {

    let summary = "This operation marks the return value of a function";
    let description = [{
        `ret` marks the return value of a function body.
        Example:

        ```mlir
        serene.ret %foo : i32
        ```
    }];

    let arguments = (ins Variadic<AnyType>:$operands);

    let builders = [OpBuilder<(ins), [{
        build($_builder, $_state, llvm::None);
    }]>];

    let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
    // TODO: Turn on the verifier for `ret`
    // let hasVerifier = 1;
}


// serene.call
def CallOp : Serene_Op<"call",
    [MemRefsNormalizable]> {
  let summary = "call operation";
  let description = [{
    The `serene.call` operation represents a direct call to a function that is
    within the same symbol scope as the call. The operands and result types of
    the call must match the specified function type. The callee is encoded as a
    symbol reference attribute named "callee".

    Example:

    ```mlir
    %2 = serene.call @my_add(%0, %1) : (f32, f32) -> f32
    ```
  }];

  let arguments = (ins FnType:$fn,
                       Variadic<AnyType>:$args);
  let results = (outs Variadic<AnyType>);

  let assemblyFormat = [{
    $fn `(` $args `)` attr-dict `:` functional-type($args, results)
  }];
}


// TODO: Do we need to have a NS type?
def NsOp : Serene_Op<"ns", [
    AffineScope, IsolatedFromAbove, NoRegionArguments, SymbolTable, Symbol,
    OpAsmOpInterface
] # GraphRegionNoTerminator.traits> {
  let summary = "This operation represents a Serene namespace.";
  let description = [{
    The `serene.ns` operation represents a namespace that will eventually
    lowers to a `mlir::ModuleOp`.

    Example:

    ```mlir
    serene.ns @some.ns {
    ....
    }
    ```
  }];

  // TODO: Create a new Attr type that is an array to represent
  //       required namespaces
  let arguments = (ins SymbolNameAttr:$sym_name,
                       OptionalAttr<StrAttr>:$sym_visibility);

  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = "$sym_name attr-dict-with-keyword $body";

  let builders = [
    OpBuilder<(ins CArg<"llvm::Optional<llvm::StringRef>", "{}">:$name)>
  ];

  let extraClassDeclaration = [{
    /// Construct a namespace from the given location with an optional name.
    //  static NsOp create(slir::reader::LocationRange loc, Optional<StringRef> name = llvm::None);

    /// Return the name of this module if present.
    llvm::StringRef getName() { return sym_name(); }

    //===------------------------------------------------------------------===//
    // SymbolOpInterface Methods
    //===------------------------------------------------------------------===//

    /// A ModuleOp may optionally define a symbol.
    bool isOptionalSymbol() { return false; }

    //===------------------------------------------------------------------===//
    // DataLayoutOpInterface Methods
    //===------------------------------------------------------------------===//
    mlir::DataLayoutSpecInterface getDataLayoutSpec();

    //===------------------------------------------------------------------===//
    // OpAsmOpInterface Methods
    //===------------------------------------------------------------------===//
    static ::llvm::StringRef getDefaultDialect() {
      return "builtin";
    }
  }];
}

#endif // SERENE_DIALECT_OPS

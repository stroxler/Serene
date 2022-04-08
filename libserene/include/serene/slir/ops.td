/* -*- C++ -*-
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

#ifndef SERENE_DIALECT_OPS
#define SERENE_DIALECT_OPS

include "mlir/IR/OpBase.td"

// Base class for Serene dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class Serene_Op<string mnemonic, list<Trait> traits = []> :
    Op<Serene_Dialect, mnemonic, traits>;

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
}


// Def
def DefOp: Serene_Op<"def"> {

    let summary = "This operation defines a global binding in the current namespace";
    let description = [{
        `def` defines a global binding in the current namespace. It always return a
        symbol type.

        Examples:

        ```mlir
        %foo = "serene.def"(%0){name = "foo"}: (i64) -> !serene.symbol

        // compact form
        %bar = serene.def "bar", %0 : i64
        ```
    }];

    let arguments = (ins StrAttr:$name,
                         AnyType:$value,
                         OptionalAttr<StrAttr>:$sym_visibility);

    let results = (outs SereneSymbol);
    let assemblyFormat = "attr-dict $name `,` $value `:` type($value)";
}

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
    let results = (outs SereneFn);
}

def ReturnOp: Serene_Op<"ret", [NoSideEffect, HasParent<"FnOp">,
                                   ReturnLike, Terminator]> {

    let summary = "This operation marks the return value of a function";
    let description = [{
        ReturnOp
    }];

    let arguments = (ins AnyType:$operand);
    let assemblyFormat =
          [{ attr-dict $operand  `:` type($operand) }];
}


#endif // SERENE_DIALECT_OPS

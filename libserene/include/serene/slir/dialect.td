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

#ifndef SERENE_DIALECT
#define SERENE_DIALECT


include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
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

    // All the types have to define a mnemonic
    let useDefaultTypePrinterParser = 1;

    // All the attributes have to define a mnemonic
    let useDefaultAttributePrinterParser = 1;
}

include "types.td"
include "ops.td"

// ============================================================================
// Old operations that are we need to replace
// ============================================================================
def Value1Op: Serene_Op<"value1"> {
    let summary = "This operation represent a compile time value";
    let description = [{
        ValueOp represent a value in compile time. For example:

        ```mlir
        %0 = "serene.value"(){value = 3} : () -> i64
        ```
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

def Fn1Op: Serene_Op<"fn1", [
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
    let results = (outs SereneFn);

}

def Return1Op: Serene_Op<"return", [NoSideEffect, HasParent<"Fn1Op">,
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

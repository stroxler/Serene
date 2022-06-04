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

#ifndef SERENE_DIALECT_TYPES_TD
#define SERENE_DIALECT_TYPES_TD

include "mlir/IR/OpBase.td"

// All of the types will extend this class.
class Serene_Type<string name> : TypeDef<Serene_Dialect, name> { }

def AnyPtr : Type<CPred<"$_self.isa<PtrType>()">,
                          "Serene pointer type", "Ptr">;

class Ptr<Type type> : Type<
        And<[AnyPtr.predicate,
             Or<[CPred<"$_self.cast<PtrType>().isOpaque()">,
                 SubstLeaves<
                     "$_self",
                     "$_self.cast<PtrType>().getPointeeType()",
                     type.predicate>]>]>,
               "A pointer to type " # type.summary,
               "PtrType">,
    // We need Ptr to be buildable coz we need to be able to infer
    // the type out of it when we use Ptr<T> as the result type
    // of an operation
    SameBuildabilityAs<type, "PtrType::get(" # type # "::get($_builder.getContext()))"> {
   Type pointeeType = type;
}


def PtrType : Serene_Type<"Ptr"> {
   let mnemonic = "ptr";
    let summary = "A pointer to a value of type T.";

    let description = [{
        A pointer to a value of type T. For example

        %0 = serene.address-of %1 : (!serene.symbol) -> !ptr<serene.symbol>
    }];

    let parameters = (ins "mlir::Type":$pointeeType, DefaultValuedParameter<"unsigned", "0">:$addressSpace);
    let genAccessors = 1;
    let assemblyFormat = "`<` qualified($pointeeType) `>`";

    let extraClassDeclaration = [{
       /// Gets or creates an instance of pointer type pointing to an
       /// object of `pointee` type in the given address space. The pointer type is
       /// created in the same context as `pointee`. If the pointee is not provided,
       /// creates an opaque pointer in the given context and address space.
       static PtrType get(mlir::MLIRContext *context, unsigned addressSpace = 0);
       static PtrType get(mlir::Type pointee, unsigned addressSpace = 0);

       bool isOpaque() const;
    }];
}

def StringType : Serene_Type<"String"> {
    let mnemonic = "string";
    let summary = "A simple string type";

    let description = [{
        A simple string type that contains the following structure:

        i8*:     buffer; pointer to the character buffer
        i32:     length; the number of chars in the buffer
    }];
}


def SymbolType : Serene_Type<"Symbol"> {
    let mnemonic = "symbol";

    let summary = "A Lisp symbol type";

    let description = [{
        A Lisp symbol type
    }];
    // let parameters = (ins "std::string":$ns, "std::string":$name);
}

def FnType : Serene_Type<"Fn"> {
    let mnemonic = "fn";

    let summary = "Function type";

    let description = [{
        Function type represent any function, anonymous or named with multiple
        bodies.
    }];

    // TODO: do we need to know about number of bodies and the signature
    // of each one?
    // let parameters = (ins "std::string":$ns, "std::string":$name);
}

def Serene_Type : AnyTypeOf<[
    PtrType,
    SymbolType,
    StringType,
    FnType
]>;

#endif // SERENE_DIALECT_TYPES_TD

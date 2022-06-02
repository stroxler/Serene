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

#ifndef SERENE_DIALECT_TYPES
#define SERENE_DIALECT_TYPES

// All of the types will extend this class.
class Serene_Type<string name> : TypeDef<Serene_Dialect, name> { }

def SereneString : Serene_Type<"String"> {
    let mnemonic = "string";
    let summary = "A simple string type";

    let description = [{
        A simple string type that contains the following structure:

        i8*:     buffer; pointer to the character buffer
        i32:     length; the number of chars in the buffer
    }];
}


def SereneSymbol : Serene_Type<"Symbol"> {
    let mnemonic = "symbol";

    let summary = "A Lisp symbol type";

    let description = [{
        A Lisp symbol type
    }];
    // let parameters = (ins "std::string":$ns, "std::string":$name);
}

def SereneFn : Serene_Type<"Fn"> {
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

#endif // SERENE_DIALECT_TYPES

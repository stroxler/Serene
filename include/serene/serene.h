/* -*- C++ -*-
 * Serene Programming Language
 *
 * Copyright (c) 2019-2021 Sameer Rahmani <lxsameer@gnu.org>
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

#ifndef SERENE_SERENE_H
#define SERENE_SERENE_H

#include "serene/config.h"
#include "serene/context.h"
#include "serene/export.h"
#include "serene/exprs/expression.h"
#include "serene/source_mgr.h"

namespace serene {

/// Clinet applications have to call this function before any interaction with
/// the Serene's compiler API.
SERENE_EXPORT void initCompiler();

/// Register the global CLI options of the serene compiler. If the client
/// application needs to setup the compilers options automatically use this
/// function in conjunction with `applySereneCLOptions`.
SERENE_EXPORT void registerSereneCLOptions();

/// Applies the global compiler options on the give \p SereneContext. This
/// function has to be called after `llvm::cl::ParseCommandLineOptions`.
SERENE_EXPORT void applySereneCLOptions(SereneContext &ctx);

/// Reads the the given \p input as a Serene source code in the given
/// \c SereneContext \p ctx and returns the possible AST tree of the input or an
/// error otherwise.
///
/// In case of an error Serene will throw the error messages vis the diagnostic
/// engine as well and the error that this function returns will be the generic
/// error message.
///
/// Be aware than this function reads the input in the context of the current
/// namespace. So for example if the input is somthing like:
///
/// (ns example.code) ....
///
/// and the current ns is `user` then if there is a syntax error in the input
/// the error will be reported under the `user` ns. This is logical because
/// this function reads the code and not evaluate it. the `ns` form has to be
/// evaluated in order to change the ns.
SERENE_EXPORT exprs::MaybeAst read(SereneContext &ctx, std::string &input);

/// Evaluates the given AST form \p input in the given \c SereneContext \p ctx
/// and retuns a new expression as the result or a possible error.
///
/// In case of an error Serene will throw the error messages vis the diagnostic
/// engine as well and the error that this function returns will be the
/// generic error message.
SERENE_EXPORT exprs::MaybeNode eval(SereneContext &ctx, exprs::Ast &input);

// TODO: Return a Serene String type instead of the std::string
// TODO: Create an overload to get a stream instead of the result string
/// Prints the given AST form \p input in the given \c SereneContext \p ctx
/// into the given \p result.
/// Note: print is a lisp action. Don't confuse it with a print function such
/// as `println`.
SERENE_EXPORT void print(SereneContext &ctx, const exprs::Ast &input,
                         std::string &result);

} // namespace serene
#endif

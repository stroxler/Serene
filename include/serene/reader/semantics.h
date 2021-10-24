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

#ifndef SERENE_READER_SEMANTICS_H
#define SERENE_READER_SEMANTICS_H

#include "serene/context.h"
#include "serene/errors/error.h"
#include "serene/exprs/expression.h"

namespace serene::reader {
using AnalyzeResult = Result<exprs::Ast, std::vector<errors::ErrorPtr>>;
/// The entry point to the Semantic analysis phase. It calls the `analyze`
/// method of each node in the given AST and creates a new AST that contains a
/// more comprehensive set of nodes in a semantically correct AST. If the
/// `analyze` method of a node return a `nullptr` value as the `success` result
/// (Checkout the `Result` type in `utils.h`) then the original node will be
/// used instead. Also please note that in **Serene** Semantic errors
/// represented as AST nodes as well. So you should expect an `analyze` method
/// of a node to return a `Result<node>::Success(Error...)` in case of a
/// semantic error.
///
/// \param ctx The semantic analysis context
/// \param inputAst The raw AST to analyze and possibly rewrite.
AnalyzeResult analyze(serene::SereneContext &ctx, exprs::Ast &inputAst);
}; // namespace serene::reader

#endif

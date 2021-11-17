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

#ifndef SERENE_SEMANTICS_H
#define SERENE_SEMANTICS_H

#include "serene/environment.h"
#include "serene/errors/error.h"
#include "serene/utils.h"

#include <llvm/ADT/StringRef.h>

#include <memory>

namespace serene {

namespace exprs {
class Expression;
using Node = std::shared_ptr<Expression>;
using Ast  = std::vector<Node>;
}; // namespace exprs

class Namespace;
using SemanticEnv = Environment<std::string, exprs::Node>;
namespace semantics {

using AnalyzeResult = Result<exprs::Ast, std::vector<errors::ErrorPtr>>;

/// This struct represent the state necessary for each analysis job.
struct AnalysisState {
  Namespace &ns;
  SemanticEnv &env;

  AnalysisState(Namespace &ns, SemanticEnv &env) : ns(ns), env(env){};

  std::unique_ptr<AnalysisState> moveToNewEnv();
};

/// Create an new `AnalysisState` by forwarding all parameters off this
/// function directly to the ctor of `AnalysisState` and returns a
/// unique pointer to the state.
template <typename... Args>
std::unique_ptr<AnalysisState> makeAnalysisState(Args &&...args) {
  return std::make_unique<AnalysisState>(std::forward<Args>(args)...);
};

/// The entry point to the Semantic analysis phase. It calls the `analyze`
/// method of each node in the given \p form and creates a new AST that
/// contains a more comprehensive set of nodes in a semantically correct
/// AST. If the `analyze` method of a node return a `nullptr` value as the
/// `success` result (Checkout the `Result` type in `utils.h`) then the
/// original node will be used instead. Any possible error will return as
/// the `error` case of the `Result` type.
/// \param state The semantic analysis state that keep track of the envs.
/// \param form  The actual AST in question.
AnalyzeResult analyze(AnalysisState &state, exprs::Ast &forms);

} // namespace semantics
} // namespace serene
#endif
